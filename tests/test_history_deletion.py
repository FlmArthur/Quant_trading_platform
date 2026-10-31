from fastapi.testclient import TestClient
from backend import store,data
from backend.api import app,PREDICTORS

client=TestClient(app)


def test_backtest_and_session_history_delete_restore_and_reference_integrity():
    # Use disposable history so existing user results are never touched.
    dataset=next(d for d in store.listing('dataset') if d['frequency']=='5m')
    response=client.post('/api/v1/sessions',json=dict(dataset_id=dataset['id'],control='manual'))
    assert response.status_code==200,response.text
    session=response.json();id=session['id']
    client.post(f'/api/v1/sessions/{id}/step',json={'bars':1})
    frozen=store.get(id)['state']
    PREDICTORS[id]=object()
    assert client.delete(f'/api/v1/sessions/{id}').status_code==200
    assert id not in [s['id'] for s in client.get('/api/v1/sessions').json()]
    assert id not in PREDICTORS
    assert client.post(f'/api/v1/sessions/{id}/step',json={'bars':1}).status_code==400
    assert client.post(f'/api/v1/sessions/{id}/orders',json={'target':1,'key':'deleted-order-key'}).status_code==400
    assert store.get(id)['state']==frozen
    assert id in [r['id'] for r in client.get('/api/v1/trash').json()]
    assert client.post(f'/api/v1/trash/{id}/restore').status_code==200
    restored=client.post(f'/api/v1/sessions/{id}/step',json={'bars':1})
    assert restored.status_code==200 and restored.json()['state']['index']==frozen['index']+1
    bt=store.put('backtest',dict(id=store.uid('bt_'),config={'dataset_id':dataset['id']},created_at=store.now(),state={'orders':[]},metrics={'total_return':.01}))
    snapshot=store.put('report',dict(id=store.uid('report_'),backtest_id=bt['id'],name='保留引用'))
    assert client.delete('/api/v1/backtests/'+bt['id']).status_code==200
    assert bt['id'] not in [r['id'] for r in client.get('/api/v1/backtests').json()]
    assert store.get(snapshot['id'])['backtest_id']==bt['id']
    assert store.get(bt['id'])['metrics']['total_return']==.01
    trash=client.get('/api/v1/trash').json()
    assert next(r for r in trash if r['id']==bt['id'])['kind']=='backtest'
    assert client.post('/api/v1/trash/'+bt['id']+'/restore').status_code==200
    assert bt['id'] in [r['id'] for r in client.get('/api/v1/backtests').json()]
