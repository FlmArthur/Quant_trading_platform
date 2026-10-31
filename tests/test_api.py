import json
import pytest
from fastapi.testclient import TestClient
from backend import store,data
from backend.api import app
from backend.schemas import ExperimentConfig
from backend.worker import main as run_worker


client=TestClient(app)


def test_upload_quality_and_aliases():
    text='日期,收盘,成交量\n'+'\n'.join(f'2024-01-{i+1:02d},3500,100' for i in range(20))
    r=client.post('/api/v1/datasets/import',data=dict(name='中文文件',symbol='TEST',frequency='1d'),files={'file':('data.csv',text.encode(),'text/csv')})
    assert r.status_code==200,r.text
    assert r.json()['rows']==20
    invalid=text+'\n2024-01-01,3501,100'
    r=client.post('/api/v1/datasets/import',data=dict(name='重复',symbol='TEST',frequency='1d'),files={'file':('data.csv',invalid.encode(),'text/csv')})
    assert r.status_code==400


def test_origin_boundary():
    r=client.post('/api/v1/reports',headers={'Origin':'https://untrusted.example'},json={})
    assert r.status_code==403


def test_full_research_to_trade_and_report():
    # Seed datasets before other tests create arbitrary uploads.
    if not any(d['synthetic'] for d in store.listing('dataset')):
        for d in store.listing('dataset'):
            with store.connect() as db:db.execute('DELETE FROM records WHERE id=?',(d['id'],))
        data.bootstrap()
    d=next(d for d in store.listing('dataset') if d['frequency']=='5m')
    c=ExperimentConfig(dataset_id=d['id'],epochs=1,hidden=8,channels=4).model_dump()
    r=client.post('/api/v1/experiments',json=c,headers={'Idempotency-Key':'api-flow-neural'})
    assert r.status_code==200,r.text
    id=r.json()['id']
    duplicate=client.post('/api/v1/experiments',json=c,headers={'Idempotency-Key':'api-flow-neural'})
    assert duplicate.json()['id']==id
    run_worker(id)
    run=client.get('/api/v1/experiments/'+id).json();assert run['status']=='COMPLETED',run.get('error')
    snapshots=client.get('/api/v1/experiments/'+id+'/snapshots').json();assert snapshots['snapshot']['hidden']
    explained=client.post('/api/v1/experiments/'+id+'/explain',json=dict(checkpoint='direction-1',mask=0))
    assert explained.status_code==200,explained.text
    assert explained.json()['conv2'][0]==[0.]*4
    p=client.post('/api/v1/strategies',json=dict(run_id=id,name='端到端策略'));assert p.status_code==200,p.text
    sid=p.json()['id'];body=dict(strategy_id=sid,dataset_id=d['id'],initial_cash=1000000)
    forbidden=client.post('/api/v1/sessions',json=dict(**body,start_index=0));assert forbidden.status_code==400
    bt=client.post('/api/v1/backtests',json=body);assert bt.status_code==200,bt.text
    backtest=bt.json();created=client.post('/api/v1/sessions',json=body).json()
    while True:
        response=client.post('/api/v1/sessions/'+created['id']+'/step',json=dict(bars=100));assert response.status_code==200,response.text
        s=response.json()
        if s['state']['status']=='COMPLETED':break
    assert s['state']['equity']==backtest['state']['equity']
    assert s['metrics']['fees']==backtest['metrics']['fees']
    assert len(s['state']['orders'])==len(backtest['state']['orders'])
    report=client.post('/api/v1/reports',json=dict(run_id=id,backtest_id=backtest['id'])).json()
    html=client.get('/api/v1/reports/'+report['id']+'/download')
    assert html.status_code==200 and d['hash'] in html.text
    assert client.get('/api/v1/backtests/'+backtest['id']+'/trades.csv').status_code==200


def test_manual_orders_are_idempotent():
    d=next(d for d in store.listing('dataset') if d['frequency']=='5m')
    r=client.post('/api/v1/sessions',json=dict(dataset_id=d['id'],control='manual')).json();id=r['id']
    client.post(f'/api/v1/sessions/{id}/step',json=dict(bars=1))
    one=client.post(f'/api/v1/sessions/{id}/orders',json=dict(target=1,key='same-order-123'))
    two=client.post(f'/api/v1/sessions/{id}/orders',json=dict(target=1,key='same-order-123'))
    assert one.json()['state']['pending']==two.json()['state']['pending']
    s=client.post(f'/api/v1/sessions/{id}/step',json=dict(bars=1)).json()
    assert len(s['state']['orders'])==1


def test_worker_pauses_at_checkpoint_and_resumes(monkeypatch):
    import threading,time
    d=next(d for d in store.listing('dataset') if d['frequency']=='5m' and d['rows']>2000)
    c=ExperimentConfig(dataset_id=d['id'],epochs=2,objective='direction',hidden=8,channels=4).model_dump()
    run=client.post('/api/v1/experiments',json=c).json();id=run['id']
    original_event=store.event;requested=False
    def observe(run_id,type,payload):
        nonlocal requested
        event=original_event(run_id,type,payload)
        if run_id==id and type=='train.epoch_end' and not requested:
            store.update(id,command='pause');requested=True
        return event
    monkeypatch.setattr(store,'event',observe)
    thread=threading.Thread(target=run_worker,args=(id,),daemon=True);thread.start()
    for _ in range(100):
        if store.get(id)['status']=='PAUSED':break
        time.sleep(.03)
    assert store.get(id)['status']=='PAUSED'
    assert (store.DATA/'runs'/id/'resume.pt').exists()
    response=client.post(f'/api/v1/experiments/{id}/controls',json=dict(action='resume'))
    assert response.status_code==200
    thread.join(5)
    assert not thread.is_alive() and store.get(id)['status']=='COMPLETED'
    events=client.get(f'/api/v1/experiments/{id}/history').json()
    cursor=events[-2]['event_id']
    streamed=client.get(f'/api/v1/experiments/{id}/events?after={cursor}')
    assert f'id: {cursor}\n' not in streamed.text
    assert f'id: {events[-1]["event_id"]}\n' in streamed.text
