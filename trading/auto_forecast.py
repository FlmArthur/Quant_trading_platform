"""Fetch verified daily index bars and run enabled forecast schedules."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hashlib
import pandas as pd
from backend import data, store
from trading.forecast import save_forecast

# Automatic acquisition is intentionally limited to known daily Chinese indices.
SYMBOLS={'sh000300':'沪深300','sh000001':'上证指数','sz399001':'深证成指'}
ALIASES={'000300':'sh000300','000300.sh':'sh000300','000001.sh':'sh000001','399001.sz':'sz399001'}


def canonical(symbol):
    value=symbol.lower(); value=ALIASES.get(value,value)
    if value not in SYMBOLS:raise ValueError(f'自动更新暂不支持 {symbol}，请手动更新日线')
    return value


def calendar_dates(now, calendar):
    if now.tzinfo is None:now=now.replace(tzinfo=ZoneInfo('Asia/Shanghai'))
    now=now.astimezone(ZoneInfo('Asia/Shanghai'))
    days=sorted({pd.Timestamp(d).date() for d in calendar})
    today=now.date()
    completed=[d for d in days if d<today or (d==today and (now.hour,now.minute)>=(15,30))]
    if not completed:raise ValueError('交易日历未覆盖最近已收盘日期')
    latest=completed[-1]
    if (today-latest).days>15:raise ValueError('交易日历可能过期，请检查日历数据')
    future=[d for d in days if d>latest]
    if not future:raise ValueError('交易日历未覆盖下一交易日')
    return latest.isoformat(),future[0].isoformat()


def fetch_calendar():
    import akshare as ak
    return ak.tool_trade_date_hist_sina().iloc[:,0].tolist()


def fetch_index(symbol, expected):
    """Fetch each scheduled cycle, reusing an immutable version when data is unchanged."""
    symbol=canonical(symbol)
    configs=store.listing('auto_feed')
    import akshare as ak
    raw=ak.stock_zh_index_daily(symbol=symbol)
    frame,_=data.normalize(raw)
    # Do not save or use an unfinished bar from the current session.
    frame=frame[frame.timestamp.str[:10]<=expected].reset_index(drop=True)
    if len(frame)<20 or frame.timestamp.iloc[-1][:10]!=expected:
        observed=frame.timestamp.iloc[-1][:10] if len(frame) else '无数据'
        raise ValueError(f'{SYMBOLS[symbol]}行情尚未更新至 {expected}，目前截止 {observed}；稍后自动重试')
    digest=hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()
    existing=next((d for d in store.listing('dataset') if d['symbol'].lower()==symbol and d.get('hash')==digest),None)
    if existing:
        data.load(existing['id']);meta=existing
    else:
        meta=data.save_dataset(frame,f'{SYMBOLS[symbol]} · 自动更新',symbol,'1d','AKShare / Sina 自动更新')
    record=next((v for v in configs if v['symbol']==symbol),None)
    value=dict(symbol=symbol,as_of=expected,dataset_id=meta['id'],updated_at=store.now())
    if record:store.update(record['id'],**value)
    else:store.put('auto_feed',dict(id=store.uid('feed_'),**value))
    return meta


def required_symbols(strategy):
    symbols={canonical(strategy['symbol'])}
    if strategy['method']=='dow':symbols.update(['sh000001','sz399001'])
    elif strategy['method']=='stgp':
        c=store.get(strategy['run_id'],'experiment')['config']
        if c.get('market_dataset_id'):
            symbols.add(canonical(store.get(c['market_dataset_id'],'dataset')['symbol']))
        elif not c.get('remove_market'):
            raise ValueError('模型依赖行情文件内的市场列；无法自动更新，请配置独立上证指数数据的模型')
    return symbols


DEFAULT_SCHEDULE=dict(schedule_mode='interval',interval_minutes=5,fixed_time='15:35')


def uses_global(config):
    # Older records with explicitly saved timing keep their individual settings.
    return config.get('use_global', 'schedule_mode' not in config)


def effective_schedule(config, global_settings=None):
    if not uses_global(config):return config
    settings={**DEFAULT_SCHEDULE,**(global_settings or {})}
    return {**config,**{key:settings[key] for key in DEFAULT_SCHEDULE}}


def is_due(config, now=None):
    """Persisted interval or local daily clock; unsuccessful daily runs retry after five minutes."""
    if not config.get('enabled'):return False
    zone=ZoneInfo('Asia/Shanghai')
    now=now or datetime.now(zone)
    if now.tzinfo is None:now=now.replace(tzinfo=zone)
    now=now.astimezone(zone)
    last_text=config.get('last_checked')
    last=None
    if last_text:
        try:
            last=datetime.fromisoformat(last_text.replace('Z','+00:00'))
            if last.tzinfo is None:last=last.replace(tzinfo=zone)
            last=last.astimezone(zone)
        except ValueError:pass
    if config.get('schedule_mode','interval')=='interval':
        return not last or now-last>=timedelta(minutes=config.get('interval_minutes',5))
    hour,minute=map(int,config.get('fixed_time','15:35').split(':'))
    scheduled=now.replace(hour=hour,minute=minute,second=0,microsecond=0)
    if now<scheduled:return False
    if not last or last<scheduled:return True
    return bool(config.get('last_error')) and now-last>=timedelta(minutes=5)


def run_once(configs, calendar=None, now=None, fetcher=fetch_index):
    now=now or datetime.now(ZoneInfo('Asia/Shanghai'))
    active=[c for c in configs if c.get('enabled')]
    if not active:return
    try:expected,next_day=calendar_dates(now, calendar if calendar is not None else fetch_calendar())
    except Exception as error:
        for c in active:store.update(c['id'],last_error=f'无法确认交易日历：{error}',last_checked=store.now())
        return
    feeds={};failures={}
    for config in active:
        try:
            strategy=store.get(config['strategy_id'],'strategy')
            if strategy.get('deleted_at'):raise ValueError('策略已删除，请恢复或关闭自动预测')
            symbols=required_symbols(strategy)
            for symbol in sorted(symbols):
                if symbol not in feeds and symbol not in failures:
                    try:feeds[symbol]=fetcher(symbol,expected)
                    except Exception as error:failures[symbol]=str(error)
            missing=[f'{SYMBOLS[s]}：{failures[s]}' for s in sorted(symbols) if s in failures]
            if missing:raise ValueError('；'.join(missing))
            target=feeds[canonical(strategy['symbol'])]
            # Preserve the strategy's exact symbol (aliases are not interchangeable in matching).
            if target['symbol']!=strategy['symbol']:raise ValueError('策略标的代码与自动数据源代码不同，请使用匹配代码的数据版本')
            overrides={symbol:meta['id'] for symbol,meta in feeds.items() if symbol in symbols}
            snapshot=save_forecast(target['id'],strategy['id'],source_overrides=overrides,target_date=next_day,now=now)
            store.update(config['id'],last_error=None,last_checked=store.now(),last_as_of=expected,
                         next_trading_day=next_day,last_forecast_id=snapshot['id'])
        except Exception as error:store.update(config['id'],last_error=str(error),last_checked=store.now())
