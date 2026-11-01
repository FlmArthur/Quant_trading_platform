"""Monthly two-index Dow confirmation using only completed months."""
import pandas as pd


def monthly(frame, asof, now):
    rows=frame[pd.to_datetime(frame.timestamp)<=pd.Timestamp(asof)].copy()
    rows['date']=pd.to_datetime(rows.timestamp)
    # Wait until the calendar month has ended; do not guess the holiday calendar.
    rows=rows[rows.date.dt.to_period('M') < pd.Period(pd.Timestamp(now).date(),freq='M')]
    return {str(period):dict(month=str(period),date=str(g.timestamp.iloc[-1]),close=float(g.close.iloc[-1]))
            for period,g in rows.groupby(rows.date.dt.to_period('M'),sort=True)}


def index_condition(rows, i):
    highs=[];lows=[]
    for j in range(1,i):
        before,cur,after=(rows[k]['close'] for k in (j-1,j,j+1))
        if cur>before and cur>after:highs.append(j)
        if cur<before and cur<after:lows.append(j)
    down=up=False;down_ref=up_ref=None
    if len(highs)>=2:
        prior_lows=[j for j in lows if j<highs[-1]]
        if prior_lows:
            down_ref=rows[prior_lows[-1]]
            down=rows[highs[-1]]['close']<rows[highs[-2]]['close'] and rows[i]['close']<down_ref['close']
    if len(lows)>=2:
        prior_highs=[j for j in highs if j<lows[-1]]
        if prior_highs:
            up_ref=rows[prior_highs[-1]]
            up=rows[lows[-1]]['close']>rows[lows[-2]]['close'] and rows[i]['close']>up_ref['close']
    return dict(up=up,down=down,close=rows[i]['close'],last_high=rows[highs[-1]] if highs else None,
                previous_high=rows[highs[-2]] if len(highs)>=2 else None,
                last_low=rows[lows[-1]] if lows else None,previous_low=rows[lows[-2]] if len(lows)>=2 else None,
                down_reference=down_ref,up_reference=up_ref)


def calculate(target, sh, sz, asof, now):
    all_months=[monthly(f,asof,now) for f in (target,sh,sz)]
    common=sorted(set(all_months[0]) & set(all_months[1]) & set(all_months[2]))
    if len(common)<6:raise ValueError('共同的完整月线不足 6 个月，无法判断双指数结构')
    expected=str(min(pd.Timestamp(asof).to_period('M'),pd.Period(pd.Timestamp(now).date(),freq='M')-1))
    if common[-1]!=expected:raise ValueError(f'双指数数据未覆盖最近完整月份 {expected}，请更新行情')
    for m in common:
        if len({v[m]['date'][:10] for v in all_months})!=1:
            raise ValueError(f'{m} 三份数据月末交易日期不一致，可能缺失行情，请检查数据')
    for left,right in zip(common,common[1:]):
        if pd.Period(right,freq='M').ordinal-pd.Period(left,freq='M').ordinal != 1:
            raise ValueError(f'{left} 至 {right} 之间缺少月线数据，请补齐后再计算')
    shrows=[all_months[1][m] for m in common];szrows=[all_months[2][m] for m in common]
    state=None;events=[];latest=None
    for i,m in enumerate(common):
        a=index_condition(shrows,i);b=index_condition(szrows,i)
        signal='up' if a['up'] and b['up'] else 'down' if a['down'] and b['down'] else None
        if signal and signal!=state:
            state=signal;events.append(dict(month=m,direction=signal,sh_close=a['close'],sz_close=b['close']))
        latest=dict(month=m,sh=a,sz=b,state=state,signal=signal,confirmation_month=events[-1]['month'] if events else None)
    latest['history_start']=common[0]
    latest['events']=events[-20:]
    latest['month_end']=all_months[0][common[-1]]['date']
    return latest
