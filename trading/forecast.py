"""Daily snapshots; strategy signals and a separate historical-return reference."""
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from backend import data, store
from trading.predictor import Predictor


def completed_prefix(frame, now=None):
    now = now or datetime.now(ZoneInfo('Asia/Shanghai'))
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo('Asia/Shanghai'))
    dates = pd.to_datetime(frame.timestamp)
    today = pd.Timestamp(now.date())
    mask = dates.dt.normalize() < today
    if (now.hour, now.minute) >= (15, 0):
        mask |= dates.dt.normalize() == today
    return frame.loc[mask].reset_index(drop=True)


def make_forecast(dataset_id, strategy_id, now=None, source_overrides=None, target_date=None):
    frame, meta = data.load(dataset_id)
    if meta.get('deleted_at') or meta['frequency'] != '1d':
        raise ValueError('请选择未删除的日线数据')
    strategy = store.get(strategy_id, 'strategy')
    if strategy.get('deleted_at'): raise ValueError('策略已删除')
    if strategy['symbol'] != meta['symbol'] or strategy['frequency'] != meta['frequency']:
        raise ValueError('策略与日线数据的标的或频率不匹配')
    frame = completed_prefix(frame, now)
    if len(frame) < 21: raise ValueError('已收盘的历史日线不足 21 条')
    asof = str(frame.timestamp.iloc[-1])
    if strategy.get('training_cutoff') and asof <= strategy['training_cutoff']:
        raise ValueError('行情截止不晚于模型训练及选参截止，无法生成样本外判断')
    if strategy.get('method') not in ('macd','dow'):
        run = store.get(strategy['run_id'], 'experiment')
        for key, artifact in run['result']['artifacts'].items():
            if store.checksum(artifact['path']) != strategy['artifact_hashes'][key]:
                raise ValueError('模型产物校验失败')
        if strategy.get('duration_run_id'):
            duration = store.get(strategy['duration_run_id'], 'experiment')
            for key, artifact in duration['result']['artifacts'].items():
                if store.checksum(artifact['path']) != strategy['duration_artifact_hashes'][key]:
                    raise ValueError('时长模型产物校验失败')
    if strategy.get('method')=='dow':
        from trading.dow import calculate
        sources=[]
        for key in ('sh_dataset_id','sz_dataset_id'):
            original=store.get(strategy[key],'dataset')
            source_id=(source_overrides or {}).get(original['symbol'].lower(),strategy[key])
            source,source_meta=data.load(source_id)
            if source_meta.get('deleted_at'):raise ValueError('策略关联的指数数据已删除，请恢复数据或创建新策略')
            source=completed_prefix(source,now)
            if source.empty or source.timestamp.iloc[-1]<asof:
                raise ValueError(f'「{source_meta["name"]}」未覆盖沪深300行情截止日，请更新指数数据并创建新版本')
            sources.append((source,source_meta))
        now_value=now or datetime.now(ZoneInfo('Asia/Shanghai'))
        result=calculate(frame,sources[0][0],sources[1][0],asof,now_value)
        state=result['state'];state_name={'up':'向上','down':'向下',None:'尚未确认'}[state]
        prediction=dict(direction=1 if state=='up' else 0 if state=='down' else None,confidence=None,
            available_at=asof,training_cutoff='',reason=f'双指数月线状态：{state_name}；最新完整月份 {result["month"]}，最近确认月份 {result["confirmation_month"] or "无"}',
            explanation=dict(method='dow',**result,sources=[dict(id=m['id'],name=m['name'],hash=m['hash'],end=m['end']) for _,m in sources]))
    else:
        predictor = Predictor(strategy, frame, meta,source_overrides=source_overrides)
        prediction = predictor.predict(len(frame)-1)
        if not prediction:
            if strategy.get('method') == 'stgp' and getattr(predictor, 'gp', False):
                missing = predictor.x.iloc[-1]
                missing_names = missing[missing.isna()].index.tolist()
                if any(name.startswith('dM') for name in missing_names):
                    market_id = predictor.c.get('market_dataset_id')
                    if market_id:
                        market = store.get(market_id, 'dataset')
                        raise ValueError(f'上证指数环境因子缺失：关联数据「{market["name"]}」截止 {market["end"][:10]}，无法覆盖当前判断需要的历史行情。请更新环境数据并重新训练、发布策略。')
                    raise ValueError('市场环境因子缺失：请检查行情中的 market_close 列是否覆盖所需历史日期。')
            raise ValueError('历史数据不足以构造策略输入或预热指标，请检查数据起点和完整性')
    if prediction.get('available_at', asof)>asof or prediction.get('training_cutoff','')>=asof:
        raise ValueError('策略使用了当时不可用的信息')
    returns = frame.close.astype(float).pct_change().dropna().tail(60).to_numpy()
    method = strategy['method']
    if method == 'dow':
        action = '长期趋势向上，可关注做多机会' if prediction['direction']==1 else '长期趋势向下，倾向观望' if prediction['direction']==0 else '尚无双指数确认信号，继续观察'
        interpretation = '上证与深证月线的市场状态，仅作为沪深300研究参考；不能预测下一交易日涨跌幅或概率'
    elif method in ('macd', 'stgp'):
        action = '倾向持有多仓' if prediction['direction']==1 else '倾向空仓观望'
        interpretation = '规则目标仓位；不等于下一交易日涨跌预测'
    else:
        action = '趋势偏上，可关注做多机会' if prediction['direction']==1 else '趋势偏下，宜谨慎观望'
        interpretation = '模型预测下一段趋势方向，不能解释为下一交易日涨跌概率'
    result = dict(id=store.uid('forecast_'),created_at=store.now(),dataset_id=dataset_id,dataset_hash=meta['hash'],
        dataset_name=meta['name'],symbol=meta['symbol'],frequency='1d',synthetic=meta['synthetic'],
        strategy_id=strategy_id,strategy_name=strategy['name'],method=method,as_of=asof,
        close=float(frame.close.iloc[-1]),training_cutoff=prediction.get('training_cutoff',''),
        direction=prediction['direction'],confidence=prediction.get('confidence'),
        reason=prediction['reason'],explanation=prediction.get('explanation',{}),advice=action,interpretation=interpretation,
        target_date=target_date, source_overrides=source_overrides or {},
        source_versions=[dict(symbol=store.get(id,'dataset')['symbol'],id=id,hash=store.get(id,'dataset')['hash']) for id in (source_overrides or {}).values()],
        reference=dict(mean=float(np.mean(returns)),low=float(np.quantile(returns,.1)),high=float(np.quantile(returns,.9)),
                       count=len(returns),start=str(frame.timestamp.iloc[-len(returns)-1]),end=asof))
    return result


def save_forecast(dataset_id, strategy_id, source_overrides=None, target_date=None, now=None):
    result = make_forecast(dataset_id, strategy_id, now=now,source_overrides=source_overrides,target_date=target_date)
    for old in store.listing('forecast'):
        if old['strategy_id']==strategy_id and old['dataset_hash']==result['dataset_hash'] and old['as_of']==result['as_of'] and old.get('explanation',{}).get('sources')==result.get('explanation',{}).get('sources') and old.get('source_overrides',{})==result.get('source_overrides',{}) and old.get('target_date')==result.get('target_date'):
            return old
    return store.put('forecast', result)


def with_outcome(record, datasets):
    result = dict(record); result['outcome'] = None
    # Compare only data versions which retain the forecast's exact reference close.
    for meta in datasets:
        if meta['symbol'] != record['symbol'] or meta['frequency'] != '1d': continue
        try: frame,_ = data.load(meta['id'])
        except (ValueError, OSError): continue
        frame = completed_prefix(frame)
        matches = frame.index[frame.timestamp == record['as_of']].tolist()
        if not matches: continue
        i=matches[0]
        if abs(float(frame.close.iloc[i])-record['close']) > max(1e-8,abs(record['close'])*1e-8): continue
        if i+1 >= len(frame): continue
        end=float(frame.close.iloc[i+1]); result['outcome']=dict(date=str(frame.timestamp.iloc[i+1]),close=end,
            daily_return=end/record['close']-1,source_dataset=meta['name'],
            note='对照数据中的下一条日线；若行情缺失交易日，需先核对数据完整性')
        break
    return result
