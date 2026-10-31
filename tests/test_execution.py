import copy
import pandas as pd
import pytest
from trading.engine import create_state,step,fill_target,equity,summary


def config(mode='research'):
    return dict(mode=mode,control='manual',initial_cash=10000.,fee_rate=.001,close_today_rate=.002,
                slippage=0.,multiplier=1.,margin_rate=.1)


def prices():
    return pd.DataFrame(dict(timestamp=['2024-01-02T09:35:00','2024-01-02T09:40:00','2024-01-03T09:35:00','2024-01-03T09:40:00'],
                open=[100.,101.,103.,105.],close=[100.5,102.,104.,105.],settlement=[100.5,102.,104.,105.]))


def test_order_executes_next_bar_and_charges_each_side():
    c=config();frame=prices();s=create_state(c,frame,0);step(s,frame,c)
    s['pending']=dict(target=1,reason='test');assert not s['orders']
    step(s,frame,c);assert s['orders'][0]['price']==101
    assert s['orders'][0]['fee']==pytest.approx(101*s['units']*.001)
    assert s['orders'][0]['filled_at']==frame.timestamp.iloc[1]
    before=len(s['orders']);fill_target(s,1,102.,frame.timestamp.iloc[1],c,'同方向')
    assert len(s['orders'])==before
    s['pending']=dict(target=-1,reason='反向');step(s,frame,c)
    assert [x['action'] for x in s['orders']]==['OPEN','CLOSE','OPEN']
    assert s['position']==-1
    assert s['equity']==pytest.approx(equity(s,104.,c))


def test_daily_settlement_does_not_double_count():
    c=config('futures');frame=prices();s=create_state(c,frame,0)
    step(s,frame,c);s['pending']=dict(target=1,reason='test');step(s,frame,c)
    assert s['basis']==102.
    assert s['cash']==pytest.approx(10000-101*.001+1)
    step(s,frame,c);assert s['equity']==pytest.approx(10000-101*.001+3)
    step(s,frame,c)
    assert s['equity']==pytest.approx(10000-101*.001+4-105*.001)
    assert s['position']==0


def test_intraday_fee_and_margin_rejection():
    c=config('futures');frame=prices();s=create_state(c,frame,0)
    fill_target(s,1,100.,frame.timestamp.iloc[0],c,'开仓');fill_target(s,0,101.,frame.timestamp.iloc[1],c,'平今')
    assert s['fees']==pytest.approx(.1+.202)
    c['initial_cash']=1;s=create_state(c,frame,0);fill_target(s,1,100.,frame.timestamp.iloc[0],c,'不足')
    assert s['position']==0 and s['orders'][0]['status']=='REJECTED'


class FixedPredictor:
    strategy={'stop_loss':0}
    def predict(self,i):return dict(direction=1 if i<2 else -1,horizon=1,as_of=str(i),reason='fixed')


def test_replay_batching_and_future_mutation_do_not_change_history():
    c=config();c['control']='auto';frame=prices();a=create_state(c,frame,0);b=copy.deepcopy(a)
    for _ in range(len(frame)):step(a,frame,c,FixedPredictor())
    for _ in range(2):step(b,frame,c,FixedPredictor())
    changed=frame.copy();changed.loc[2:,'close']*=10
    history=copy.deepcopy(b)
    # Same known prefix has identical account values regardless of future prices.
    d=create_state(c,changed,0)
    for _ in range(2):step(d,changed,c,FixedPredictor())
    assert d['equity']==history['equity']
    for _ in range(2):step(b,frame,c,FixedPredictor())
    assert a['equity']==b['equity']
    assert [(o['price'],o['fee']) for o in a['orders']]==[(o['price'],o['fee']) for o in b['orders']]
