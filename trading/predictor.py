import numpy as np
from backend import store,data


class Predictor:
    def __init__(self,strategy,frame,meta,source_overrides=None):
        self.strategy=strategy;self.frame=frame;self.meta=meta
        if strategy.get('method')=='macd':
            from trading.macd import MACDPredictor
            self.custom=MACDPredictor(strategy,frame)
            return
        self.run=store.get(strategy['run_id'],'experiment');self.c=dict(self.run['config'])
        if self.c.get('method')=='stgp' and self.c.get('market_dataset_id') and source_overrides:
            original=store.get(self.c['market_dataset_id'],'dataset')
            replacement=source_overrides.get(original['symbol'].lower())
            if replacement:self.c['market_dataset_id']=replacement
        from research.registry import METHODS
        method=METHODS[self.c['method']]
        self.custom=method.predictor_factory(strategy,frame,meta,self.run) if method.predictor_factory else None
        if self.custom:return
        self.gp=self.c['method']=='stgp'
        if self.gp:
            from research.gp import factors,primitives
            market=data.load(self.c['market_dataset_id'])[0] if self.c.get('market_dataset_id') else None
            self.x=factors(frame,self.c,market);self.pset=primitives(list(self.x.columns));self.trees={}
        else:
            from research.neural import load_model
            self.direction=load_model(self.run['result']['artifacts']['direction'])
            durationrun=store.get(strategy['duration_run_id']) if strategy.get('duration_run_id') else self.run
            self.duration_config=durationrun['config'];self.duration=None
            if 'duration' in durationrun['result']['artifacts']:
                self.duration=load_model(durationrun['result']['artifacts']['duration'])

    def predict(self,index):
        if self.custom:return self.custom.predict(index)
        asof=self.frame.timestamp.iloc[index]
        if self.gp:
            from research.gp import predict_tree,tree_json
            from deap import gp
            eligible=[w for w in self.run['result']['windows'] if w['training_cutoff']<asof]
            if not eligible or self.x.iloc[index].isna().any():return None
            w=eligible[-1];expression=w['expression']
            if expression not in self.trees:self.trees[expression]=gp.PrimitiveTree.from_string(expression,self.pset)
            value=bool(predict_tree(self.trees[expression],self.pset,self.x.iloc[[index]].to_numpy(float))[0])
            return dict(direction=int(value),confidence=None,horizon=1,as_of=asof,available_at=asof,
                        training_cutoff=w['training_cutoff'],run_id=self.run['id'],
                        reason=f'规则判断为{value}；训练截止{w["training_cutoff"]}',
                        explanation=dict(expression=expression,tree=tree_json(self.trees[expression],self.x.iloc[index].to_dict())))
        from research.neural import infer
        out=infer(self.direction,self.frame,index,self.c,self.meta['frequency'])
        if out is None:return None
        value,view=out;prob=float(1/(1+np.exp(-np.clip(value,-50,50))))
        direction=1 if prob>=self.run['result']['artifacts']['direction']['threshold'] else -1
        horizon=self.strategy['fallback_horizon'];hours=None
        duration_view=None
        if self.duration:
            dur=infer(self.duration,self.frame,index,self.duration_config,self.meta['frequency'])
            if dur:
                hours=dur[0];duration_view=dur[1];minutes=240 if self.meta['frequency']=='1d' else int(self.meta['frequency'].rstrip('m'))
                horizon=int(np.clip(round(hours*60/minutes),1,200))
        threshold=self.run['result']['artifacts']['direction']['threshold']
        from research.trends import features_at
        minutes=240 if self.meta['frequency']=='1d' else int(self.meta['frequency'].rstrip('m'))
        sample=features_at(self.frame.close.to_numpy(float),index,self.c,minutes)
        last=sample['segments'][-1];previous_direction=1 if last['slope']>0 else -1
        candidate=direction!=previous_direction
        segments=[dict(start=self.frame.timestamp.iloc[v['start']],end=self.frame.timestamp.iloc[v['end']],
                       start_price=float(self.frame.close.iloc[v['start']]),end_price=float(self.frame.close.iloc[v['end']]),slope=v['slope'],bars=v['bars']) for v in sample['segments']]
        def overview(v):
            if not v:return None
            return dict(input=v['input'],trend=v['trend'],conv_channels=np.mean(v['conv2'],axis=1).tolist(),
                        gates=np.mean(v['gates'],axis=2).tolist(),fusion=v['fusion'][:16],active_branches=v['active_branches'])
        effective_hours=horizon*minutes/60
        return dict(direction=direction,confidence=prob,selected_direction_probability=prob if direction==1 else 1-prob,
                    horizon=horizon,raw_duration_hours=hours,effective_duration_hours=effective_hours,
                    origin_index=index,evaluation_horizon=self.c['label_horizon'],segment_error=self.c['segment_error'],min_segment=self.c['min_segment'],
                    as_of=asof,available_at=asof,training_cutoff=self.strategy['training_cutoff'],run_id=self.run['id'],
                    reason=f'上涨概率{prob:.3f}，阈值{threshold:.2f}，预计持有{horizon}条行情',
                    explanation=dict(method=self.c['method'],threshold=threshold,run_id=self.run['id'],
                        direction_model=self.run['id'],duration_model=self.strategy.get('duration_run_id') or (self.run['id'] if self.duration else None),
                        turning=dict(candidate=candidate,previous_direction=previous_direction,label=('疑似向上拐点' if direction==1 else '疑似向下拐点') if candidate else '延续当前趋势',
                            definition='当前已知历史末段方向与模型预测方向相反；候选拐点并非独立训练的拐点分类概率'),
                        segments=segments,direction_network=overview(view),duration_network=overview(duration_view),
                        duration_note='时长为交易小时，不含午休与休市；执行使用1至200条限幅，原始输出保留。负时长不具有实际意义。' if hours is not None else '未配置时长模型，使用固定持有期',
                        probability_note='概率为方向模型sigmoid输出，尚未校准，不等于当前预测必然正确的概率；'+('本方向模型同时使用CNN和LSTM融合。' if self.c['method']=='hybrid' else '本方向模型仅使用'+self.c['method'].upper()+'分支；时长来自单独配置的模型或固定持有期。')))
