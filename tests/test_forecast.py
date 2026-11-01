from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from fastapi.testclient import TestClient
from backend import data, store
from backend.api import app
from trading.forecast import completed_prefix, make_forecast, with_outcome


def test_daily_forecast_cutoff_and_no_future_prices():
    frame=pd.DataFrame({'timestamp':['2026-01-02T00:00:00','2026-01-05T00:00:00','2026-01-06T00:00:00'],'close':[100,101,900]})
    zone=ZoneInfo('Asia/Shanghai')
    assert len(completed_prefix(frame,datetime(2026,1,5,14,59,tzinfo=zone)))==1
    assert len(completed_prefix(frame,datetime(2026,1,5,15,0,tzinfo=zone)))==2


def test_forecast_snapshot_and_outcome():
    dates=pd.bdate_range('2020-01-01',periods=110)
    frame=pd.DataFrame({'timestamp':dates,'close':[100+i*.2+(i%7) for i in range(110)]})
    original=data.save_dataset(frame.iloc[:100], '测试预测', 'TESTFORECAST','1d','test')
    later=data.save_dataset(frame.iloc[:101], '后续行情', 'TESTFORECAST','1d','test')
    client=TestClient(app)
    strategy=client.post('/api/v1/strategies/macd',json={'dataset_id':original['id'],'fast':3,'slow':6,'signal':3}).json()
    body={'dataset_id':original['id'],'strategy_ids':[strategy['id']]}
    result=client.post('/api/v1/forecasts',json=body)
    assert result.status_code==200, result.text
    saved=result.json()['results'][0]
    assert saved['reference']['count']==60
    assert saved['confidence'] is None
    assert client.post('/api/v1/forecasts',json=body).json()['results'][0]['id']==saved['id']
    view=with_outcome(saved,[later])
    assert view['outcome']['date'][:10]==dates[100].strftime('%Y-%m-%d')
    assert abs(view['outcome']['daily_return']-(frame.close.iloc[100]/frame.close.iloc[99]-1))<1e-12
    p=client.post('/api/v1/forecast-schedules',json={'strategy_id':strategy['id'],'enabled':True})
    assert p.status_code==200 and p.json()['enabled']
    assert client.post('/api/v1/forecast-schedules',json={'strategy_id':strategy['id'],'enabled':False}).json()['enabled'] is False


def test_automatic_calendar_and_successful_idempotent_forecast():
    from trading.auto_forecast import calendar_dates, run_once
    zone=ZoneInfo('Asia/Shanghai')
    calendar=pd.bdate_range('2025-01-01','2025-01-08')
    assert calendar_dates(datetime(2025,1,6,15,29,tzinfo=zone),calendar)==('2025-01-03','2025-01-06')
    assert calendar_dates(datetime(2025,1,6,15,30,tzinfo=zone),calendar)==('2025-01-06','2025-01-07')
    dates=pd.bdate_range('2024-08-01','2025-01-06')
    frame=pd.DataFrame({'timestamp':dates,'close':[100+i%17 for i in range(len(dates))]})
    dataset=data.save_dataset(frame,'自动测试','sh000300','1d','test')
    client=TestClient(app)
    strategy=client.post('/api/v1/strategies/macd',json={'dataset_id':dataset['id']}).json()
    plan=client.post('/api/v1/forecast-schedules',json={'strategy_id':strategy['id']}).json()
    now=datetime(2025,1,6,16,tzinfo=zone)
    def fetch(symbol,expected):
        assert symbol=='sh000300' and expected=='2025-01-06'
        return dataset
    run_once([plan],calendar=calendar,now=now,fetcher=fetch)
    plan=store.get(plan['id']); assert plan['last_error'] is None
    result=store.get(plan['last_forecast_id']);assert result['target_date']=='2025-01-07'
    assert result['as_of'].startswith('2025-01-06')
    run_once([plan],calendar=calendar,now=now,fetcher=fetch)
    assert store.get(plan['id'])['last_forecast_id']==result['id']


def test_automatic_failure_does_not_create_stale_forecast():
    from trading.auto_forecast import run_once
    dates=pd.bdate_range('2024-01-01',periods=100)
    dataset=data.save_dataset(pd.DataFrame({'timestamp':dates,'close':range(100,200)}),'过期行情','sh000300','1d','test')
    strategy=TestClient(app).post('/api/v1/strategies/macd',json={'dataset_id':dataset['id']}).json()
    plan=TestClient(app).post('/api/v1/forecast-schedules',json={'strategy_id':strategy['id']}).json()
    def fail(symbol,expected):raise ValueError('行情尚未更新')
    run_once([plan],calendar=pd.bdate_range('2025-01-01','2025-01-08'),now=datetime(2025,1,6,16,tzinfo=ZoneInfo('Asia/Shanghai')),fetcher=fail)
    updated=store.get(plan['id']);assert '行情尚未更新' in updated['last_error']
    assert 'last_forecast_id' not in updated


def test_schedule_due_interval_daily_and_retry():
    from trading.auto_forecast import is_due
    zone=ZoneInfo('Asia/Shanghai')
    def at(hour,minute,day=6):return datetime(2025,1,day,hour,minute,tzinfo=zone)
    assert is_due({'enabled':True,'schedule_mode':'interval','interval_minutes':30},at(15,30))
    assert not is_due({'enabled':True,'interval_minutes':30,'last_checked':at(15,10).isoformat()},at(15,39))
    assert is_due({'enabled':True,'interval_minutes':30,'last_checked':at(15,10).isoformat()},at(15,40))
    daily={'enabled':True,'schedule_mode':'daily','fixed_time':'15:35'}
    assert not is_due(daily,at(15,34))
    assert is_due(daily,at(15,35))
    assert not is_due({**daily,'last_checked':at(15,36).isoformat(),'last_error':None},at(20,0))
    assert is_due({**daily,'last_checked':at(15,36).isoformat(),'last_error':'网络故障'},at(15,41))
    assert not is_due({**daily,'last_checked':at(15,36).isoformat(),'last_error':'网络故障'},at(15,40))
    assert is_due({**daily,'last_checked':at(15,36,5).isoformat(),'last_error':None},at(16,0))
    assert not is_due({'enabled':False},at(16,0))


def test_schedule_settings_validation_and_toggle_preserves_settings():
    dates=pd.bdate_range('2024-01-01',periods=100)
    dataset=data.save_dataset(pd.DataFrame({'timestamp':dates,'close':range(100,200)}),'周期测试','sh000300','1d','test')
    client=TestClient(app)
    strategy=client.post('/api/v1/strategies/macd',json={'dataset_id':dataset['id']}).json()
    url='/api/v1/forecast-schedules'
    sid=strategy['id']
    assert client.post(url,json={'strategy_id':sid,'interval_minutes':0}).status_code==422
    assert client.post(url,json={'strategy_id':sid,'fixed_time':'25:80'}).status_code==422
    response=client.post(url,json={'strategy_id':sid,'schedule_mode':'daily','fixed_time':'16:15','interval_minutes':30})
    assert response.status_code==200 and response.json()['fixed_time']=='16:15'
    response=client.post(url,json={'strategy_id':sid,'enabled':False})
    assert response.json()['schedule_mode']=='daily' and response.json()['fixed_time']=='16:15'
    response=client.post(url,json={'strategy_id':sid,'enabled':True})
    assert response.json()['interval_minutes']==30 and response.json()['fixed_time']=='16:15'


def test_global_schedule_and_individual_overrides():
    from trading.auto_forecast import effective_schedule, is_due, uses_global
    url='/api/v1/forecast-schedules';client=TestClient(app)
    dates=pd.bdate_range('2024-01-01',periods=100)
    dataset=data.save_dataset(pd.DataFrame({'timestamp':dates,'close':range(100,200)}),'统一设置测试','sh000300','1d','test')
    a=client.post('/api/v1/strategies/macd',json={'dataset_id':dataset['id'],'name':'统一'}).json()
    b=client.post('/api/v1/strategies/macd',json={'dataset_id':dataset['id'],'name':'独立'}).json()
    global_plan=client.post(url,json={'strategy_id':a['id']}).json()
    own_plan=client.post(url,json={'strategy_id':b['id'],'use_global':False,'schedule_mode':'interval','interval_minutes':15}).json()
    assert uses_global(global_plan) and not uses_global(own_plan)
    store.update(global_plan['id'],last_checked='2025-01-06T07:40:00+00:00')
    store.update(own_plan['id'],last_checked='2025-01-06T07:40:00+00:00')
    result=client.post('/api/v1/forecast-schedule-settings',json={'schedule_mode':'daily','fixed_time':'16:30','interval_minutes':5})
    assert result.status_code==200
    global_plan=store.get(global_plan['id']);own_plan=store.get(own_plan['id'])
    assert global_plan['last_checked'] is None and own_plan['last_checked'] is not None
    combined=effective_schedule(global_plan,result.json());own=effective_schedule(own_plan,result.json())
    assert combined['schedule_mode']=='daily' and combined['fixed_time']=='16:30'
    assert own['schedule_mode']=='interval' and own['interval_minutes']==15
    assert not is_due(combined,datetime(2025,1,6,16,29,tzinfo=ZoneInfo('Asia/Shanghai')))
    assert is_due(combined,datetime(2025,1,6,16,30,tzinfo=ZoneInfo('Asia/Shanghai')))
    # Toggling preserves saved override settings.
    client.post(url,json={'strategy_id':b['id'],'use_global':True})
    restored=client.post(url,json={'strategy_id':b['id'],'use_global':False}).json()
    assert restored['interval_minutes']==15 and not restored['use_global']
    assert not uses_global({'schedule_mode':'daily','fixed_time':'17:00'})
