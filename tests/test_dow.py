from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import pytest
from trading.dow import calculate, index_condition
from trading.forecast import make_forecast
from backend import data
from backend.api import create_dow, DowStrategy


def rows(values):
    return [dict(month=f'2026-{i+1:02}',date=f'2026-{i+1:02}-28T00:00:00',close=float(v)) for i,v in enumerate(values)]


def test_pivots_require_following_month_and_double_confirmation():
    seq=rows([10,13,8,12,7,9])
    assert not index_condition(seq,3)['down']
    assert index_condition(seq,4)['down']
    dates=pd.date_range('2026-01-28',periods=6,freq='MS')+pd.Timedelta(days=27)
    def frame(values):return pd.DataFrame({'timestamp':dates.strftime('%Y-%m-%dT00:00:00'),'close':values})
    f=frame([10,13,8,12,7,9]);other=frame([10,13,8,12,7,9])
    result=calculate(f,f,other,'2026-07-28',datetime(2026,8,2,tzinfo=ZoneInfo('Asia/Shanghai')))
    assert result['state']=='down' and result['confirmation_month']=='2026-06'
    # A different second index cannot establish a joint down confirmation.
    result=calculate(f,f,frame([10,11,12,13,14,15]),'2026-07-28',datetime(2026,8,2,tzinfo=ZoneInfo('Asia/Shanghai')))
    assert result['state'] is None


def test_dow_strategy_rejects_wrong_market():
    dates=pd.bdate_range('2023-01-01',periods=100)
    frame=pd.DataFrame({'timestamp':dates,'close':range(100,200)})
    a=data.save_dataset(frame,'wrong','WRONG','1d','test')
    with pytest.raises(ValueError,match='沪深300'):
        create_dow(DowStrategy(dataset_id=a['id'],sh_dataset_id=a['id'],sz_dataset_id=a['id']))
