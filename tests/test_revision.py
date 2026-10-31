import copy
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend import data,store
from backend.api import app
from trading.assessment import assess
from trading.engine import create_state,step

client=TestClient(app)


def test_minute_history_request_cannot_silently_save_recent_window(monkeypatch):
    import akshare
    frame=pd.DataFrame(dict(datetime=pd.date_range('2026-10-01 09:35',periods=1023,freq='5min'),close=np.arange(1023)+3500))
    monkeypatch.setattr(akshare,'futures_zh_minute_sina',lambda **kwargs:frame)
    c=dict(kind='futures_minute',symbol='IF0',period='5',start='2023-01-01',end='',name='long')
    before=len(store.listing('dataset'))
    with pytest.raises(ValueError,match='请求区间未覆盖'):data.acquire(c)
    assert len(store.listing('dataset'))==before
    partial=data.acquire(dict(**c,allow_partial=True))
    assert partial['rows']==1023 and partial['coverage_status']=='PARTIAL'
    assert any('未覆盖完整' in w for w in partial['warnings'])


def test_delete_and_restore_preserve_dependency_files():
    original=next(d for d in store.listing('dataset') if d['frequency']=='5m')
    frame,_=data.load(original['id']);d=data.save_dataset(frame,'可恢复测试','DELETE','5m','测试')
    r=client.delete('/api/v1/datasets/'+d['id']);assert r.status_code==200
    assert d['id'] not in [x['id'] for x in client.get('/api/v1/datasets').json()]
    # Existing references can still read frozen data after deletion.
    assert len(data.load(d['id'])[0])==len(frame)
    assert d['id'] in [x['id'] for x in client.get('/api/v1/trash').json()]
    assert client.post('/api/v1/trash/'+d['id']+'/restore').status_code==200
    assert d['id'] in [x['id'] for x in client.get('/api/v1/datasets').json()]
    run=store.put('experiment',dict(id=store.uid('run_'),kind='training',status='RUNNING',config={'dataset_id':d['id']}))
    assert client.delete('/api/v1/experiments/'+run['id']).status_code==400
    assert client.delete('/api/v1/datasets/'+d['id']).status_code==400
    store.update(run['id'],status='CANCELLED')
    assert client.delete('/api/v1/experiments/'+run['id']).status_code==200
    assert run['id'] not in [x['id'] for x in client.get('/api/v1/experiments').json()]


def test_historical_start_date_and_frozen_model_cutoff():
    from backend.api import session_inputs
    d=next(d for d in store.listing('dataset') if d['frequency']=='5m')
    frame,_=data.load(d['id']);start=frame.timestamp.iloc[200][:10]
    r=client.post('/api/v1/sessions',json=dict(dataset_id=d['id'],control='manual',start_date=start))
    assert r.status_code==200,r.text
    state=r.json()['state'];assert frame.timestamp.iloc[state['start_index']].startswith(start)
    # Model's training date is checked even if the user explicitly chooses a past date.
    strategy=store.put('strategy',dict(id=store.uid('strategy_'),symbol=d['symbol'],frequency=d['frequency'],training_cutoff=frame.timestamp.iloc[300],run_id='revision-run',artifact_hashes={}))
    store.put('experiment',dict(id='revision-run',kind='training',status='COMPLETED',config={},result={'artifacts':{}}))
    with pytest.raises(ValueError,match='未来信息'):
        session_inputs(dict(dataset_id=d['id'],strategy_id=strategy['id'],mode='research',start_date=frame.timestamp.iloc[100][:10]))
    assert client.delete('/api/v1/strategies/'+strategy['id']).status_code==200
    with pytest.raises(ValueError,match='策略已删除'):
        session_inputs(dict(dataset_id=d['id'],strategy_id=strategy['id'],mode='research'))
    assert client.post('/api/v1/trash/'+strategy['id']+'/restore').status_code==200


def test_accuracy_waits_until_future_is_revealed_and_matches_training_label():
    frame=pd.DataFrame(dict(timestamp=pd.date_range('2024-01-02 09:35',periods=9,freq='5min').astype(str),close=[100,99,101,102,104,105,108,107,106]))
    prediction=dict(id='pred',origin_index=1,as_of=frame.timestamp.iloc[1],direction=1,confidence=.8,horizon=4,
        raw_duration_hours=4*5/60,evaluation_horizon=4,segment_error=100,min_segment=3,explanation={'turning':{'candidate':True}})
    state=dict(index=4,observations=[prediction],signals=[])
    before=assess(state,frame,'5m');assert before['pending']==1 and before['direction_accuracy'] is None
    mutated=frame.copy();mutated.loc[5:,'close']*=10
    assert assess(state,mutated,'5m')==before
    state['index']=5
    result=assess(state,frame,'5m');assert result['resolved']==1 and result['direction_accuracy']==1
    assert result['duration_mae_hours']==pytest.approx(0)
    assert result['calibration'][0]['count']==1
    # Observations from later bars must never affect the summary of an earlier replay cursor.
    hidden=copy.deepcopy(prediction);hidden.update(origin_index=7,as_of=frame.timestamp.iloc[7])
    state['observations'].append(hidden)
    assert assess(state,frame,'5m')['total']==1


def test_historical_feed_chunks_mapping_and_rejects_missing_days(monkeypatch):
    from backend import history_provider as h
    days=pd.bdate_range('2023-01-02',periods=12);contracts=['IF2301.CFX']*6+['IF2302.CFX']*6
    mapping=pd.DataFrame(dict(trade_date=days.strftime('%Y%m%d'),mapping_ts_code=contracts,ts_code='IF.CFX'))
    calls=[]
    def query(name,params):
        calls.append((name,params))
        if name=='fut_mapping':return mapping
        selected=days[(days>=pd.Timestamp(params['start_date']))&(days<=pd.Timestamp(params['end_date']))]
        times=[d+pd.Timedelta(hours=9,minutes=35+5*i) for d in selected for i in range(3)]
        return pd.DataFrame(dict(trade_time=times,close=3500.,open=3500.,vol=100))
    monkeypatch.setattr(h,'query',query);monkeypatch.setattr(h.time,'sleep',lambda _:None)
    c=dict(provider='tushare',kind='futures_minute',start='2023-01-01',end='2023-01-17',symbol='IF0',period='5',name='历史')
    result=data.acquire(c)
    frame,_=data.load(result['id'])
    assert result['rows']==36 and frame.contract_id.nunique()==2
    assert not frame.timestamp.duplicated().any()
    assert all(pd.Timestamp(p['end_date'])-pd.Timestamp(p['start_date'])<pd.Timedelta(days=7) for name,p in calls if name=='ft_mins')
    original=query
    def missing(name,params):
        f=original(name,params)
        return f if name=='fut_mapping' else f[pd.to_datetime(f.trade_time).dt.date!=days[3].date()]
    monkeypatch.setattr(h,'query',missing)
    with pytest.raises(ValueError,match='缺少分钟数据'):data.acquire(c)


def test_separate_duration_model_and_live_network_values():
    # Reuse completed, real test model artifacts from the end-to-end test.
    source=next(r for r in store.listing('experiment') if r.get('result',{}).get('artifacts',{}).get('duration') and r['config'].get('method')=='hybrid')
    response=client.post('/api/v1/strategies',json=dict(run_id=source['id'],duration_run_id=source['id'],name='方向与时长解释'))
    assert response.status_code==200,response.text
    s=response.json();assert s['duration_artifact_hashes']
    from trading.predictor import Predictor
    frame,meta=data.load(s['dataset_id']);predictor=Predictor(s,frame,meta)
    index=next(i for i in range(len(frame)) if frame.timestamp.iloc[i]>s['training_cutoff'])
    prediction=predictor.predict(index)
    assert 0<=prediction['confidence']<=1
    assert len(prediction['explanation']['direction_network']['input'])==4
    assert len(prediction['explanation']['duration_network']['gates'])==4
    future=frame.copy();future.loc[index+1:,'close']*=5
    changed=Predictor(s,future,meta).predict(index)
    assert changed==prediction
    r=client.post('/api/v1/sessions',json=dict(strategy_id=s['id'],dataset_id=s['dataset_id']))
    session=r.json();advance=client.post('/api/v1/sessions/'+session['id']+'/step',json=dict(bars=30))
    assert advance.status_code==200,advance.text
    result=advance.json();assert len(result['state']['observations'])==30
    assert result['latest_prediction']['explanation']['direction_model']==source['id']
    assert result['evaluation']['resolved']>0


def test_legacy_partial_metadata_is_marked_without_changing_prices():
    frame=pd.DataFrame(dict(timestamp=pd.date_range('2026-10-01 09:35',periods=25,freq='5min'),close=3500))
    meta=data.save_dataset(frame,'旧的2023请求','IF0','5m','AKShare / Sina')
    task=store.put('experiment',dict(id=store.uid('run_'),kind='acquisition',status='COMPLETED',config=dict(start='2023-01-01',end='2026-10-24'),result=meta))
    data.audit_legacy_acquisitions()
    updated=store.get(meta['id'],'dataset')
    assert updated['coverage_status']=='PARTIAL' and updated['hash']==meta['hash']
    assert store.get(task['id'])['coverage_warning']
    data.audit_legacy_acquisitions()
    assert len(store.get(meta['id'])['warnings'])==len(updated['warnings'])
