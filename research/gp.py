"""Typed, vectorized GP; per-generation shared RMO weights and frozen rolling trees."""
import copy
import pickle
import random
from pathlib import Path
import numpy as np
import pandas as pd
from deap import base, creator, gp, tools
from backend import store

if not hasattr(creator,'QuantFitness'): creator.create('QuantFitness',base.Fitness,weights=(1.0,))
if not hasattr(creator,'QuantTree'): creator.create('QuantTree',gp.PrimitiveTree,fitness=creator.QuantFitness)


def protected_div(a,b):
    a,b=np.asarray(a),np.asarray(b)
    with np.errstate(all='ignore'):
        return np.clip(np.divide(a,b,out=np.zeros(np.broadcast_shapes(a.shape,b.shape)),where=np.abs(b)>1e-9),-1e6,1e6)


def between(a,b,c): return np.logical_and(a>=np.minimum(b,c),a<=np.maximum(b,c))
def multiply(a,b): return np.clip(a*b,-1e6,1e6)


def factors(frame,c,market=None):
    if 'volume' not in frame: raise ValueError('论文二需要成交量 volume 列')
    if c.get('market_dataset_id'):
        if market is None: raise ValueError('市场环境数据不存在')
        env=frame[['timestamp']].merge(market[['timestamp','close']],on='timestamp',how='left')['close']
    elif 'market_close' in frame: env=frame.market_close
    elif not c['remove_market']: raise ValueError('30因子模式需要上证指数数据或market_close列；也可选择25因子消融模式')
    else: env=None
    base_rows=frame[frame.timestamp.str[:10]==c['base_date']]
    if base_rows.empty: raise ValueError(f'基准日 {c["base_date"]} 不在数据内，请显式选择一个已有交易日作为探索基准')
    base_price=float(base_rows.close.iloc[0]);base_volume=float(base_rows.volume.iloc[0])
    if base_volume<=0:raise ValueError('基准日成交量必须大于零')
    p=frame.close;v=frame.volume;out={}
    for i in range(1,6):out[f'P{i}']=p.shift(i)/base_price
    for k in [5,10,20,60,120]:out[f'MA{k}']=p.rolling(k).mean().shift(1)/base_price
    for i in range(1,6):out[f'V{i}']=v.shift(i)/base_volume
    for k in [5,10,20,60,120]:out[f'MV{k}']=v.rolling(k).mean().shift(1)/base_volume
    for k in [5,10,20,60,120]:out[f'SD{k}']=p.pct_change().rolling(k).std(ddof=1).shift(1)
    if not c['remove_market']:
        for i in range(1,6):out[f'dM{i}']=env.pct_change(fill_method=None).shift(i)
    return pd.DataFrame(out).replace([np.inf,-np.inf],np.nan)


def primitives(names):
    p=gp.PrimitiveSetTyped('Rule',[float]*len(names),bool)
    for f,types,ret,name in [(np.add,[float,float],float,'add'),(np.subtract,[float,float],float,'sub'),
            (multiply,[float,float],float,'mul'),(protected_div,[float,float],float,'div'),
            (np.maximum,[float,float],float,'max'),(np.minimum,[float,float],float,'min'),
            (np.greater,[float,float],bool,'gt'),(np.less,[float,float],bool,'lt'),
            (np.logical_and,[bool,bool],bool,'And'),(np.logical_or,[bool,bool],bool,'Or'),
            (np.logical_not,[bool],bool,'Not'),(np.where,[bool,float,float],float,'if_then_else'),
            (between,[float,float,float],bool,'isBetween')]:p.addPrimitive(f,types,ret,name=name)
    p.addTerminal(True,bool);p.addTerminal(False,bool)
    for n in [-1.,0.,1.]:p.addTerminal(n,float)
    p.renameArguments(**{f'ARG{i}':name for i,name in enumerate(names)})
    return p


def predict_tree(tree,pset,x):
    with np.errstate(all='ignore'):
        value=gp.compile(tree,pset)(*[x[:,i] for i in range(x.shape[1])])
    return np.broadcast_to(np.asarray(value,dtype=bool),(len(x),)).copy()


def raw_fitness(signal,opens,closes,fee):
    # A close-time decision executes at the following open. Labels stop before the training boundary.
    signal=np.asarray(signal,dtype=float)
    returns=(opens[2:]/opens[1:-1]-1)*signal[:-2]
    turnover=np.abs(np.diff(np.r_[0,signal[:-2]]))
    net=returns-fee*turnover
    equity=np.cumprod(1+np.clip(net,-.99,10));peak=np.maximum.accumulate(np.r_[1.,equity])[1:]
    total=float(equity[-1]-1) if len(equity) else 0
    mdd=float(np.max(1-equity/peak)) if len(equity) else 0
    sd=np.std(net,ddof=1) if len(net)>1 else 0
    sharpe=float(np.mean(net)/sd*np.sqrt(252)) if sd>1e-12 else -10.
    accu=float(np.mean(signal[:-2].astype(bool)==(opens[2:]>opens[1:-1]))) if len(returns) else 0
    ref=total/mdd if mdd>1e-8 else (-10. if total<=0 else min(total/1e-8,100))
    return np.array([np.clip(ref,-100,100),np.clip(sharpe,-100,100),accu])


def scores(raw,weights,kind):
    a,b,c=raw.T
    if kind=='Sharpe':return b
    if kind=='Accuracy':return c
    if kind=='RefDown':return a
    return np.stack([a,(b-1)/.5,(c-.5)/.015],axis=1)@np.asarray(weights)


def tree_json(tree,feature_values=None):
    cursor=0
    def node():
        nonlocal cursor
        index=cursor;el=tree[cursor];cursor+=1
        label=el.name if el.arity else str(el.value)
        result=dict(id=index,label=label,type='bool' if el.ret is bool else 'float',children=[node() for _ in range(el.arity)])
        if feature_values is not None and label in feature_values:result['value']=float(feature_values[label])
        return result
    return node()


def rolling_start(frame,c,valid):
    rows=np.flatnonzero(valid)
    if not len(rows):raise ValueError('没有完整因子行，请检查指标预热和市场环境数据')
    earliest=int(rows[0])+c['train_window']
    start=max(earliest,int(len(frame)*c['train_ratio']))
    if c.get('gp_start_date'):
        start=int(pd.to_datetime(frame.timestamp).searchsorted(pd.Timestamp(c['gp_start_date'])))
        if start<earliest:raise ValueError('样本外起点之前不足指标预热和完整训练窗口')
    if start>=len(frame)-5:raise ValueError('指标预热及训练窗后可预测数据不足，请增加日频数据')
    if c['base_date']>frame.timestamp.iloc[start-c['train_window']][:10]:
        raise ValueError('缩放基准日晚于最早训练样本，会使用未来信息；请选择更早且实际存在的基准日')
    return start


def fit(run,frame,meta,control,emit,market=None):
    c=run['config'];random.seed(c['seed']);np.random.seed(c['seed'])
    features=factors(frame,c,market);valid=features.notna().all(axis=1).to_numpy();names=list(features.columns)
    n=len(frame);start=rolling_start(frame,c,valid)
    pset=primitives(names);folder=store.DATA/'runs'/run['id'];folder.mkdir(parents=True,exist_ok=True)
    resume=folder/'gp-resume.pkl';state=pickle.loads(resume.read_bytes()) if resume.exists() else {}
    population=state.get('population');begin=state.get('window',start);generation=state.get('generation',0)
    windows=state.get('windows',[]);predictions=state.get('predictions',[])
    if state:random.setstate(state['random']);np.random.set_state(state['numpy'])
    expression=lambda:gp.genHalfAndHalf(pset,min_=2,max_=4)
    for window in range(begin,n,c['predict_window']):
        control();train_start=window-c['train_window'];ids=np.arange(train_start,window)
        if not valid[ids].all():raise ValueError('训练窗口的因子存在缺失，请检查成交量和上证指数日期对齐')
        x=features.iloc[ids].to_numpy(float);opens=frame.get('open',frame.close).iloc[ids].to_numpy(float);closes=frame.close.iloc[ids].to_numpy(float)
        if population is None:population=[creator.QuantTree(expression()) for _ in range(c['population'])]
        ckid=f'window-{window}-generation-{c["generations"]}'
        for gen in range(generation,c['generations']):
            control();weights=np.repeat(1/3,3)
            if c['fitness']=='RMO':weights=weights+np.random.uniform(0,c['random_weight_max'],3)
            elif c['fitness']=='PRMO':weights=np.random.uniform(0,c['random_weight_max'],3)
            offspring=[];operations=[]
            for _ in range(c['population']):
                r=random.random()
                if r<.7:
                    a,b=map(copy.deepcopy,random.sample(population,2));before=str(a);gp.cxOnePoint(a,b);child=a;kind='交叉'
                elif r<.9:
                    child=copy.deepcopy(random.choice(population));before=str(child)
                    gp.mutUniform(child,expr=lambda pset,type_:gp.genGrow(pset,0,1,type_),pset=pset);kind='变异'
                else:child=copy.deepcopy(random.choice(population));before=str(child);kind='复制'
                if child.height>c['max_depth']:child=copy.deepcopy(random.choice(population));kind='超深拒绝'
                offspring.append(child)
                if len(operations)<3:operations.append(dict(operation=kind,before=before,after=str(child)))
            candidates=population+offspring
            raw=np.array([raw_fitness(predict_tree(tree,pset,x),opens,closes,c['fee_rate']) for tree in candidates])
            fitness=scores(raw,weights,c['fitness'])
            for tree,score in zip(candidates,fitness):tree.fitness.values=(float(score),)
            bestindex=int(np.argmax(fitness));best=copy.deepcopy(candidates[bestindex])
            population=[best]+list(map(copy.deepcopy,tools.selTournament(candidates,c['population']-1,tournsize=3)))
            ckid=f'window-{window}-generation-{gen+1}'
            view=dict(checkpoint_id=ckid,window=window,generation=gen+1,weights=weights.tolist(),
                      raw_metrics=dict(zip(['refdown','sharpe','accuracy'],raw[bestindex].tolist())),fitness=float(fitness[bestindex]),
                      tree=tree_json(best),expression=str(best),operations=operations,
                      train_start=frame.timestamp.iloc[train_start],train_end=frame.timestamp.iloc[window-1],
                      test_start=frame.timestamp.iloc[window],feature_names=names,population=c['population'])
            ranked=np.argsort(fitness)[-10:][::-1]
            view['candidate_count']=len(candidates)
            view['population_sample']=[dict(expression=str(candidates[j]),height=candidates[j].height,
                 fitness=float(fitness[j]),metrics=raw[j].tolist()) for j in ranked]
            store.write_json(folder/f'{ckid}.json',view)
            state=dict(population=population,window=window,generation=gen+1,windows=windows,predictions=predictions,
                       random=random.getstate(),numpy=np.random.get_state())
            temp=resume.with_suffix('.tmp');temp.write_bytes(pickle.dumps(state));temp.replace(resume)
            emit('gp.generation_end',view)
            store.update(run['id'],progress=min(1,((window-start)+(gen+1)/c['generations']*c['predict_window'])/(n-start)),latest_checkpoint=ckid)
        stop=min(window+c['predict_window'],n)
        ids=np.arange(window,stop)
        if not valid[ids].all():raise ValueError('预测窗口因子存在缺失')
        signals=predict_tree(population[0],pset,features.iloc[ids].to_numpy(float))
        windows.append(dict(start=window,end=stop,training_cutoff=frame.timestamp.iloc[window-1],
                            start_time=frame.timestamp.iloc[window],end_time=frame.timestamp.iloc[stop-1],
                            expression=str(population[0]),tree=tree_json(population[0]),checkpoint_id=ckid))
        predictions.extend([dict(index=int(i),as_of=frame.timestamp.iloc[i],direction=int(p),window=window) for i,p in zip(ids,signals)])
        emit('gp.window_end',dict(window=window,end=stop,expression=str(population[0])))
        # New rolling window starts afresh. Its random state and boundary are durable.
        population=None;generation=0
        state=dict(population=None,window=window+c['predict_window'],generation=0,windows=windows,predictions=predictions,
                   random=random.getstate(),numpy=np.random.get_state())
        temp=resume.with_suffix('.tmp');temp.write_bytes(pickle.dumps(state));temp.replace(resume)
    available=[p for p in predictions if p['index']+2<len(frame)]
    opening=frame.get('open',frame.close)
    accuracy=float(np.mean([bool(p['direction'])==(opening.iloc[p['index']+2]>opening.iloc[p['index']+1]) for p in available])) if available else None
    result=dict(windows=windows,predictions=predictions,feature_names=names,training_cutoff=frame.timestamp.iloc[start-1],
                results=dict(windows=len(windows),predictions=len(predictions),accuracy=accuracy),artifacts=dict(gp=dict(path=str(resume))),
                deviations=['D05 随机项默认U(0,1/3)','D06 按式3.1-3.3变换','固定基准日；未含市场信息需显式25因子模式',
                            '每代父子代共享随机权重并全部重评分；无穷适应度使用有限约定'])
    store.write_json(folder/'gp-result.json',result);return result
