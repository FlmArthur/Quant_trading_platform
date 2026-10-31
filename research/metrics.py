import numpy as np
import pandas as pd


def classification(y, probabilities, threshold=.5):
    y=np.asarray(y,dtype=bool); p=np.asarray(probabilities)>=threshold
    tp=int((y&p).sum()); tn=int((~y&~p).sum()); fp=int((~y&p).sum()); fn=int((y&~p).sum())
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); beta=.4
    return dict(accuracy=float((y==p).mean()),precision=precision,recall=recall,
                fbeta=(1+beta**2)*precision*recall/max(beta**2*precision+recall,1e-12),
                threshold=float(threshold),confusion=[[tn,fp],[fn,tp]],samples=len(y))


def performance(equity, initial, dates, benchmark=None, rf=0):
    values=np.asarray(equity,dtype=float)
    if not len(values): return {}
    peak=np.maximum.accumulate(np.r_[initial,values])[1:]
    drawdown=1-values/peak
    daily=pd.Series(values,index=pd.to_datetime(dates)).groupby(lambda x:x.date()).last()
    daily_values=daily.to_numpy()
    previous=np.r_[initial,daily_values[:-1]]
    returns=(daily_values-previous)/previous
    sd=float(np.std(returns,ddof=1)) if len(returns)>1 else 0
    sharpe=float((returns.mean()-rf/252)/sd*np.sqrt(252)) if sd>1e-12 else None
    total=float(values[-1]/initial-1); mdd=float(drawdown.max())
    days=max(len(daily),1)
    annual=float(np.expm1(np.clip(np.log(max(values[-1]/initial,1e-12))*252/days,-50,50)))
    return dict(total_return=total,annual_return=annual,max_drawdown=mdd,sharpe=sharpe,
                refdown=total/mdd if mdd>1e-12 else None,
                excess_return=total-benchmark if benchmark is not None else None,
                trading_days=days,annual_days=252,risk_free=rf,drawdowns=drawdown.tolist())
