"""One deterministic execution engine for backtests and persisted paper sessions."""
import math
from backend import store
from research.metrics import performance


def create_state(config,frame,start):
    price=float(frame.get('open',frame.close).iloc[start])
    units=1 if config['mode']=='futures' else max(1,math.floor(config['initial_cash']/price))
    return dict(index=start-1,start_index=start,cash=config['initial_cash'],equity=config['initial_cash'],
                position=0,units=units,basis=0,entry_price=0,opened_at=None,margin=0,available=config['initial_cash'],
                fees=0,orders=[],ledger=[],signals=[],observations=[],curve=[],pending=None,valid_until=start,stopped=False,
                status='PAUSED',seen_keys=[],last_day=None)


def equity(state,price,c):
    mult=(c.get('multiplier') or 1) if c['mode']=='futures' else 1
    return state['cash']+state['position']*state['units']*mult*(price-state['basis'])


def fill_target(state,target,price,time,c,reason,signal_id=None,submitted_at=None):
    current=state['position'];mult=(c.get('multiplier') or 1) if c['mode']=='futures' else 1;qty=state['units'];day=time[:10]
    if current==target:return
    def record(side,action,fee,realized=0,status='FILLED'):
        id=store.uid('ord_')
        order=dict(id=id,side=side,action=action,quantity=qty,price=price,status=status,fee=fee,
                   filled_at=time,submitted_at=submitted_at or time,signal_id=signal_id,reason=reason,realized_pnl=realized)
        state['orders'].append(order)
        state['ledger'].append(dict(id=id,time=time,amount=realized-fee,cash=state['cash'],type=action))
    if current:
        execution=price-c['slippage']*current
        realized=current*qty*mult*(execution-state['basis'])
        rate=c['close_today_rate'] if c['mode']=='futures' and state['opened_at'][:10]==day else c['fee_rate']
        fee=execution*qty*mult*rate
        state['cash']+=realized-fee;state['fees']+=fee
        record('SELL' if current>0 else 'BUY','CLOSE',fee,realized)
        state['orders'][-1]['price']=execution;state['position']=0;state['margin']=0
    if target:
        execution=price+c['slippage']*target
        if c['mode']=='research':
            qty=max(0,math.floor(state['cash']/(execution*(1+c['fee_rate']))))
            state['units']=qty
            if qty==0:
                record('BUY' if target>0 else 'SELL','OPEN',0,status='REJECTED')
                state['orders'][-1]['reason']='资金不足一单位，拒绝开仓';return
        fee=execution*qty*mult*c['fee_rate']
        required=execution*qty*mult*(c.get('margin_rate') if c['mode']=='futures' else 1)
        if required+fee>state['cash']:
            record('BUY' if target>0 else 'SELL','OPEN',0,status='REJECTED')
            state['orders'][-1]['reason']='资金不足，拒绝开仓';return
        state['cash']-=fee;state['fees']+=fee;state['position']=target
        state['basis']=execution;state['entry_price']=execution;state['opened_at']=time;state['margin']=required if c['mode']=='futures' else 0
        record('BUY' if target>0 else 'SELL','OPEN',fee)
        state['orders'][-1]['price']=execution


def step(state,frame,c,predictor=None,manual=None):
    if state['status']=='COMPLETED':return state
    i=state['index']+1
    if i>=len(frame):state['status']='COMPLETED';return state
    row=frame.iloc[i];time=str(row.timestamp);close=float(row.close);openprice=float(row.get('open',close))
    if state['pending']:
        p=state['pending'];fill_target(state,p['target'],openprice,time,c,p['reason'],p.get('signal_id'),p.get('submitted_at'));state['pending']=None
    state['index']=i;day=time[:10];mult=(c.get('multiplier') or 1) if c['mode']=='futures' else 1
    mark=equity(state,close,c)
    if c['mode']=='futures' and state['position']:
        state['margin']=close*state['units']*mult*c['margin_rate']
        if mark<state['margin']:
            fill_target(state,0,close,time,c,'保证金不足，强制平仓');state['stopped']=True
        if i==len(frame)-1 or str(frame.timestamp.iloc[i+1])[:10]!=day:
            settle=float(row.settlement);pnl=state['position']*state['units']*mult*(settle-state['basis'])
            state['cash']+=pnl;state['basis']=settle
            state['ledger'].append(dict(id=store.uid('settle_'),time=time,amount=pnl,cash=state['cash'],type='DAILY_SETTLEMENT'))
    if manual is not None:
        state['pending']=dict(target=manual,reason='手动虚拟委托',signal_id=None,submitted_at=time)
    elif predictor and c['control']=='auto':
        observation=None
        if c.get('observe_each_bar',False):
            observation=predictor.predict(i)
            if observation:
                if observation.get('available_at',time)>time or observation.get('training_cutoff','')>=time:
                    raise ValueError('策略尝试使用当前时刻尚不可用的信息')
                if observation.get('direction') not in [-1,0,1] or not isinstance(observation.get('horizon'),int) or observation['horizon']<1:
                    raise ValueError('策略输出违反统一信号协议')
                observation['id']=store.uid('sig_')
                state.setdefault('observations',[]).append(observation)
        stop=predictor.strategy.get('stop_loss',0)
        if state['position'] and stop and state['position']*(close/state['entry_price']-1)<=-stop:
            state['pending']=dict(target=0,reason='收盘触发止损，下一行情执行',submitted_at=time);state['stopped']=True
        elif i>=state['valid_until']:
            signal=observation if c.get('observe_each_bar',False) else predictor.predict(i)
            if signal:
                if signal.get('direction') not in [-1,0,1] or not isinstance(signal.get('horizon'),int) or signal['horizon']<1:
                    raise ValueError('策略输出违反统一信号协议')
                if signal.get('available_at',time)>time or signal.get('training_cutoff','')>=time:
                    raise ValueError('策略尝试使用当前时刻尚不可用的信息')
                signal.setdefault('id',store.uid('sig_'));state['signals'].append(signal)
                state['valid_until']=i+signal['horizon'];state['stopped']=False
                state['pending']=dict(target=signal['direction'],reason=signal['reason'],signal_id=signal['id'],submitted_at=time)
    if i==len(frame)-1:
        fill_target(state,0,close,time,c,'预定回放区间结束平仓')
        if state['pending']:
            state['orders'].append(dict(id=store.uid('ord_'),status='CANCELLED',action='PENDING',side='—',price=None,
                 quantity=0,fee=0,filled_at=time,reason='回放结束，没有下一条可成交行情'))
        state['pending']=None;state['status']='COMPLETED'
    state['equity']=equity(state,close,c);state['available']=state['equity']-state['margin']
    state['last_day']=day
    state['curve'].append(dict(time=time,equity=state['equity'],cash=state['cash'],position=state['position'],price=close))
    return state


def summary(state,frame,c):
    curve=state['curve'];initial=c['initial_cash']
    benchmark=float(frame.close.iloc[state['index']]/frame.close.iloc[state['start_index']]-1) if curve else 0
    result=performance([p['equity'] for p in curve],initial,[p['time'] for p in curve],benchmark)
    filled=[o for o in state['orders'] if o['status']=='FILLED'];closed=[o for o in filled if o['action']=='CLOSE']
    result.update(fees=state['fees'],trades=len(filled),closed_trades=len(closed),
                  win_rate=sum(o['realized_pnl']>0 for o in closed)/len(closed) if closed else None,
                  benchmark_return=benchmark,execution='下一开盘' if 'open' in frame else '下一收盘',
                  mode=c['mode'],slippage=c['slippage'],gross_return=(state['equity']+state['fees'])/initial-1)
    return result
