import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from backend import data,store
from backend.api import app
from backend.schemas import ExperimentConfig
from research.trends import prepare
from trading.engine import create_state,step


def test_akshare_filter_excludes_next_midnight(monkeypatch):
    import akshare
    frame=pd.DataFrame(dict(date=pd.bdate_range('2024-01-02',periods=40),close=np.arange(40)+3000))
    monkeypatch.setattr(akshare,'stock_zh_index_daily',lambda symbol:frame)
    end=frame.date.iloc[29].strftime('%Y-%m-%d')
    result=data.acquire(dict(kind='index_daily',symbol='sh000300',name='边界验证',end=end))
    assert result['rows']==30
    assert result['end'].startswith(end)
    assert not any('连续合约' in w for w in result['warnings'])


def test_slice_is_immutable_and_exclusive_of_following_day():
    d=next(d for d in store.listing('dataset') if d['frequency']=='5m')
    client=TestClient(app)
    r=client.post('/api/v1/datasets/'+d['id']+'/slice',json=dict(name='研究切片',start='2024-01-02',end='2024-01-03'))
    assert r.status_code==200,r.text
    s=r.json();assert s['end']<'2024-01-04'
    assert s['parent_hash']==d['hash'] and s['id']!=d['id']
    assert data.load(d['id'])[1]['hash']==d['hash']


def test_explicit_date_splits():
    d=next(d for d in store.listing('dataset') if d['frequency']=='5m' and d['rows']>2000)
    c=ExperimentConfig(dataset_id=d['id'],train_end_date='2024-02-08',validation_end_date='2024-02-23').model_dump()
    frame,meta=data.load(d['id']);samples,p=prepare(frame,c,meta['frequency'])
    assert frame.timestamp.iloc[p['train_end']].startswith('2024-02-08')
    assert frame.timestamp.iloc[p['validation_end']].startswith('2024-02-23')


def test_research_ignores_futures_multiplier():
    c=dict(mode='research',control='manual',initial_cash=10000.,fee_rate=.001,close_today_rate=.002,slippage=0.,multiplier=300,margin_rate=.12)
    frame=pd.DataFrame(dict(timestamp=['2024-01-02T09:35','2024-01-02T09:40','2024-01-02T09:45'],open=[100.,101.,102.],close=[100.,102.,103.]))
    s=create_state(c,frame,0);step(s,frame,c);s['pending']=dict(target=1,reason='test');step(s,frame,c)
    assert s['position']==1 and s['orders'][0]['status']=='FILLED'
    assert s['equity']<10100


def test_offline_paper_samples_use_next_full_segment_and_cannot_publish():
    d=next(d for d in store.listing('dataset') if d['frequency']=='5m' and d['rows']>2000)
    frame,meta=data.load(d['id']);c=ExperimentConfig(dataset_id=d['id'],sampling_protocol='paper_offline').model_dump()
    samples,protocol=prepare(frame,c,meta['frequency'])
    assert protocol['research_only']
    assert all(len(s['segments'])==4 and s['label_end']>s['index'] for s in samples)
    assert all(s['duration']==(s['label_end']-s['index']+1)*5/60 for s in samples)
    run=dict(id=store.uid('offline_'),kind='training',status='COMPLETED',config=c,result={'protocol':protocol})
    store.put('experiment',run)
    r=TestClient(app).post('/api/v1/strategies',json={'run_id':run['id'],'name':'不能发布'})
    assert r.status_code==400 and '仅用于研究' in r.text


def test_gp_rejects_normalization_from_future_date():
    from research.gp import factors,rolling_start
    import pytest
    d=next(d for d in store.listing('dataset') if d['frequency']=='1d' and d['rows']>700)
    frame,meta=data.load(d['id'])
    c=ExperimentConfig(dataset_id=d['id'],method='stgp',base_date=frame.timestamp.iloc[500][:10],train_ratio=.5).model_dump()
    f=factors(frame,c)
    with pytest.raises(ValueError,match='未来信息'):rolling_start(frame,c,f.notna().all(axis=1).to_numpy())
