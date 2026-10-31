import io
import re
from pathlib import Path
import numpy as np
import pandas as pd
from . import store

ALIASES = {'日期': 'timestamp', '时间': 'timestamp', 'date': 'timestamp', 'datetime': 'timestamp',
           '收盘': 'close', '收盘价': 'close', '开盘': 'open', '开盘价': 'open',
           '最高': 'high', '最低': 'low', '成交量': 'volume', '结算价': 'settlement',
           '合约': 'contract_id', '市场收盘': 'market_close'}


def normalize(frame):
    frame = frame.copy()
    frame.columns = [ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError('字段映射后出现重名列，请只保留一列日期和收盘价')
    if not {'timestamp', 'close'}.issubset(frame.columns):
        raise ValueError('文件至少需要 timestamp/date/日期 和 close/收盘 字段')
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='coerce', format='mixed')
    if frame['timestamp'].isna().any():
        raise ValueError('存在无法解析的日期，请使用 YYYY-MM-DD HH:MM:SS')
    if frame['timestamp'].dt.tz is not None:
        frame['timestamp'] = frame['timestamp'].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    for c in ['close', 'open', 'high', 'low', 'volume', 'market_close', 'settlement']:
        if c in frame:
            frame[c] = pd.to_numeric(frame[c].astype(str).str.replace(',', ''), errors='coerce')
            if not np.isfinite(frame[c]).all() or ((frame[c] < 0) if c == 'volume' else (frame[c] <= 0)).any():
                raise ValueError(f'{c} 包含空值、非有限数或无效数值')
    if frame['timestamp'].duplicated().any():
        raise ValueError('存在重复时间戳；请清理后重新导入，平台不会静默删除')
    if len(frame) < 20 or len(frame) > 200000:
        raise ValueError('数据应包含20至200000条记录')
    if 'high' in frame and 'low' in frame and (frame['high'] < frame['low']).any():
        raise ValueError('最高价不能低于最低价')
    was_sorted = frame['timestamp'].is_monotonic_increasing
    frame = frame.sort_values('timestamp').reset_index(drop=True)
    frame['timestamp'] = frame['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    warnings = [] if was_sorted else ['已按时间排序，原始文件哈希单独保存']
    if 'open' not in frame:
        warnings.append('缺少开盘价，订单将在下一条收盘价成交')
    if 'volume' not in frame:
        warnings.append('缺少成交量，无法训练论文二的完整因子集')
    return frame, warnings


def save_dataset(frame, name, symbol, frequency, source, raw_hash=None, synthetic=False):
    frame, warnings = normalize(frame)
    times = pd.to_datetime(frame.timestamp)
    if frequency != '1d':
        delta = int(frequency.rstrip('m'))
        diffs = times.diff().dt.total_seconds().div(60)
        within = (times.dt.date == times.shift().dt.date) & (diffs < 90)
        gaps = int(((diffs > delta) & within).sum())
        if gaps: warnings.append(f'发现 {gaps} 个盘中间隔超过 {delta} 分钟，请核对交易日历')
    if re.fullmatch(r'[A-Z]{1,3}0',symbol.upper()) or symbol.upper() in ['IF.CFE', 'IF9999']:
        warnings.append('连续合约仅用于研究回放；期货账户需真实合约及换月映射')
    id = store.uid('ds_'); path = store.DATA / 'datasets' / f'{id}.csv'
    path.parent.mkdir(exist_ok=True); frame.to_csv(path, index=False)
    meta = dict(id=id, name=name, symbol=symbol, frequency=frequency, source=source, rows=len(frame),
                start=frame.timestamp.iloc[0], end=frame.timestamp.iloc[-1], columns=list(frame.columns),
                hash=store.checksum(path), raw_hash=raw_hash, synthetic=synthetic, warnings=warnings,
                path=str(path), created_at=store.now(), timezone='Asia/Shanghai', version=1)
    return store.put('dataset', meta)


def load(id):
    meta = store.get(id, 'dataset')
    if store.checksum(meta['path']) != meta['hash']:
        raise ValueError('数据文件校验失败，文件已发生变化')
    return pd.read_csv(meta['path']), meta


def upload(content, filename, name, symbol, frequency):
    import hashlib
    if filename.lower().endswith('.xlsx'):
        frame = pd.read_excel(io.BytesIO(content))
    elif filename.lower().endswith(('.csv', '.tsv')):
        for enc in ['utf-8-sig', 'gb18030']:
            try:
                frame = pd.read_csv(io.BytesIO(content), encoding=enc, sep='\t' if filename.endswith('.tsv') else ',')
                break
            except UnicodeDecodeError:
                continue
        else: raise ValueError('文件编码无法识别，请保存为UTF-8 CSV')
    else:
        raise ValueError('支持 CSV、TSV 和 XLSX 文件')
    return save_dataset(frame, name, symbol, frequency, '手动上传', hashlib.sha256(content).hexdigest())


def acquire(config, progress=None):
    import akshare as ak
    kind = config['kind']; symbol = config['symbol']
    provider=config.get('provider','akshare')
    if provider=='tushare':
        from .history_provider import fetch
        frame=fetch(config,progress)
    elif kind == 'index_daily':
        frame = ak.stock_zh_index_daily(symbol=symbol)
    elif kind == 'futures_daily':
        frame = ak.futures_zh_daily_sina(symbol=symbol)
    else:
        frame = ak.futures_zh_minute_sina(symbol=symbol, period=config['period'])
    frame, _ = normalize(frame)
    available = [frame.timestamp.iloc[0], frame.timestamp.iloc[-1]]
    requested_start=pd.Timestamp(config['start']) if config.get('start') else None
    requested_end=pd.Timestamp(config['end']) if config.get('end') else None
    partial_start=requested_start is not None and (pd.Timestamp(available[0])-requested_start).days>10
    partial_end=requested_end is not None and (requested_end-pd.Timestamp(available[1])).days>10
    partial=partial_start or partial_end
    if partial and not config.get('allow_partial',False):
        raise ValueError(f'请求区间未覆盖：请求 {config.get("start") or "最早"} 至 {config.get("end") or "最新"}，源仅返回 {len(frame)} 条，实际 {available[0]} 至 {available[1]}。AKShare 新浪分钟接口不支持按日期回溯多年；请改用有权限的长期历史源或导入文件，也可明确选择仅保存可用部分。')
    if config.get('start'): frame = frame[frame.timestamp >= pd.Timestamp(config['start']).isoformat()]
    if config.get('end'): frame = frame[frame.timestamp < (pd.Timestamp(config['end']) + pd.Timedelta(days=1)).isoformat()]
    if len(frame) < 20:
        raise ValueError(f'所选区间不足20条；数据源实际可用范围 {available[0]} 至 {available[1]}。历史分钟数据可手动上传。')
    frequency = '1d' if kind != 'futures_minute' else config['period'] + 'm'
    meta = save_dataset(frame, config['name'], symbol, frequency, 'Tushare / 历史分钟' if provider=='tushare' else 'AKShare / Sina')
    meta.update(provider_available_range=available,requested_range=[config.get('start'),config.get('end')],coverage_status='PARTIAL' if partial else 'BOUNDARIES_COVERED')
    if partial:meta['warnings'].append('仅保存实际可用部分，未覆盖完整请求区间')
    if provider=='tushare' and symbol.upper().endswith('0'):meta['warnings'].append('按每日主力映射拼接，未复权；换月跳变不能视为可成交利润')
    return store.put('dataset', meta)


def bootstrap():
    if store.listing('dataset',include_deleted=True): return
    rng = np.random.default_rng(2026)
    for freq, n, symbol in [('5m', 2400, 'DEMO-IF'), ('1d', 800, 'DEMO-CSI300')]:
        if freq == '1d': times = pd.bdate_range('2021-01-04', periods=n)
        else:
            times = []
            for day in pd.bdate_range('2024-01-02', periods=50):
                times.extend(pd.date_range(day + pd.Timedelta(hours=9,minutes=35), periods=24, freq='5min'))
                times.extend(pd.date_range(day + pd.Timedelta(hours=13,minutes=5), periods=24, freq='5min'))
        t = np.arange(n)
        returns = .0001 + .002*np.sin(t/21) + rng.normal(0,.0025,n)
        close = 3500 * np.exp(np.cumsum(returns)); opens = np.r_[close[0], close[:-1]] * (1+rng.normal(0,.0003,n))
        frame = pd.DataFrame(dict(timestamp=times, close=close, open=opens,
                high=np.maximum(close, opens)*(1+rng.uniform(0,.001,n)),
                low=np.minimum(close, opens)*(1-rng.uniform(0,.001,n)),
                volume=rng.integers(10000,100000,n), market_close=3000*np.exp(np.cumsum(returns*.6+rng.normal(0,.001,n)))))
        save_dataset(frame, '教学合成 · '+('五分钟趋势' if freq=='5m' else '日频进化'), symbol, freq,
                     '固定种子合成数据 / 非真实行情', synthetic=True)


def audit_legacy_acquisitions():
    """Annotate old partial successes without changing bars or invalidating references."""
    for run in store.listing('experiment',include_deleted=True):
        if run.get('kind')!='acquisition' or not run.get('result',{}).get('id'):continue
        c=run['config']
        try:meta=store.get(run['result']['id'],'dataset')
        except ValueError:continue
        if meta.get('coverage_status'):continue
        partial=(bool(c.get('start')) and (pd.Timestamp(meta['start'])-pd.Timestamp(c['start'])).days>10) or (bool(c.get('end')) and (pd.Timestamp(c['end'])-pd.Timestamp(meta['end'])).days>10)
        if partial:
            message='旧任务仅保存了实际可用部分，未覆盖请求历史区间；请使用长期历史源或文件导入补齐'
            meta.update(coverage_status='PARTIAL',requested_range=[c.get('start'),c.get('end')])
            if message not in meta['warnings']:meta['warnings'].append(message)
            store.put('dataset',meta);store.update(run['id'],coverage_warning=message)
