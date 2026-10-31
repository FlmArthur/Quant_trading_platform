"""Optional authenticated historical feed. Tokens never enter task configs/events."""
import hashlib
import os
import re
import time
import pandas as pd
import requests
from . import store


def token():
    path=store.DATA/'secrets'/'tushare.token'
    return os.environ.get('TUSHARE_TOKEN','').strip() or (path.read_text().strip() if path.exists() else '')


def set_token(value):
    folder=store.DATA/'secrets';folder.mkdir(mode=0o700,exist_ok=True)
    path=folder/'tushare.token'
    # Replace credentials atomically, with owner-only permissions from creation.
    temp=folder/'tushare.token.tmp'
    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as f:f.write(value.strip())
    temp.replace(path)


def query(name,params):
    credential=token()
    if not credential:raise ValueError('长期历史源未配置 Tushare Token；请在数据中心配置有期货历史分钟权限的账号，或上传历史文件')
    cache=store.DATA/'history_cache';cache.mkdir(exist_ok=True)
    key=hashlib.sha256(store.encode([name,params]).encode()).hexdigest()
    path=cache/(key+'.json')
    # Current data can revise; cache only finished historical dates.
    historical=str(params.get('end_date',''))[:10].replace('-','')<pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y%m%d')
    if historical and path.exists():return pd.DataFrame(store.read_json(path))
    try:
        response=requests.post('https://api.tushare.pro',json=dict(api_name=name,token=credential,params=params,fields=''),timeout=35)
        response.raise_for_status();body=response.json()
    except (requests.RequestException,ValueError):raise ValueError('长期历史源连接失败；请检查网络后重试，已完成分块会复用缓存') from None
    if body.get('code')!=0:
        message=str(body.get('msg','接口拒绝访问')).replace(credential,'[已隐藏]')
        raise ValueError('Tushare：'+message[:400])
    payload=body.get('data') or {};frame=pd.DataFrame(payload.get('items',[]),columns=payload.get('fields',[]))
    if historical and len(frame):store.write_json(path,frame.to_dict('records'))
    return frame


def fetch(config,progress=None):
    symbol=config['symbol'].upper();start=pd.Timestamp(config['start']).normalize()
    end=pd.Timestamp(config.get('end') or pd.Timestamp.now(tz='Asia/Shanghai').date()).normalize()
    if end<start:raise ValueError('开始日期不能晚于结束日期')
    # Main-contract stitching is explicit, unadjusted and restricted to research mode.
    main=re.fullmatch(r'(IF|IH|IC|IM)0',symbol)
    if main:
        code=main[1]+'.CFX';mapping=query('fut_mapping',dict(ts_code=code,start_date=start.strftime('%Y%m%d'),end_date=end.strftime('%Y%m%d')))
        if mapping.empty:raise ValueError('主力合约映射为空；请确认映射权限与所选日期')
        mapping=mapping.drop_duplicates('trade_date').sort_values('trade_date')
        mapped_dates=pd.to_datetime(mapping.trade_date,format='%Y%m%d')
        if (mapped_dates.iloc[0]-start).days>10 or (end-mapped_dates.iloc[-1]).days>10:
            raise ValueError('主力映射未覆盖请求边界，请检查数据源权限和区间')
        expected_dates=set(mapping.trade_date.astype(str));chunks=[]
        for contract,group in mapping.groupby('mapping_ts_code'):
            dates=set(group.trade_date.astype(str));a=pd.to_datetime(min(dates));b=pd.to_datetime(max(dates))
            chunks.extend((contract,x,min(x+pd.Timedelta(days=6),b),dates) for x in pd.date_range(a,b,freq='7D'))
    else:
        if not re.fullmatch(r'[A-Z]+\d{4}\.[A-Z]+',symbol):
            raise ValueError('长期源代码示例：IF0（主力拼接）或 IF2306.CFX（具体合约）')
        expected_dates=None;chunks=[(symbol,x,min(x+pd.Timedelta(days=6),end),None) for x in pd.date_range(start,end,freq='7D')]
    collected=[]
    for j,(contract,a,b,dates) in enumerate(chunks):
        frame=query('ft_mins',dict(ts_code=contract,freq=config['period']+'min',start_date=a.strftime('%Y-%m-%d 00:00:00'),end_date=b.strftime('%Y-%m-%d 23:59:59')))
        if len(frame)>=8000:raise ValueError('数据源分块达到8000条上限，拒绝保存可能截断的行情')
        if len(frame):
            frame=frame.rename(columns={'trade_time':'timestamp','vol':'volume'})
            if dates is not None:frame=frame[pd.to_datetime(frame.timestamp).dt.strftime('%Y%m%d').isin(dates)]
            frame=frame.copy();frame['contract_id']=contract;collected.append(frame)
        if progress:progress((j+1)/len(chunks),f'已获取 {j+1}/{len(chunks)} 个历史分块')
        time.sleep(.15)
    if not collected:raise ValueError('所选历史区间没有分钟数据，请核对合约上市日期与数据权限')
    result=pd.concat(collected,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp')
    result=result[(pd.to_datetime(result.timestamp)>=start)&(pd.to_datetime(result.timestamp)<end+pd.Timedelta(days=1))]
    if expected_dates:
        actual=set(pd.to_datetime(result.timestamp).dt.strftime('%Y%m%d'))
        missing=sorted(expected_dates-actual)
        if missing:raise ValueError(f'有 {len(missing)} 个映射交易日缺少分钟数据，例如 {missing[:5]}；拒绝标记完整成功，分块已缓存可重试')
    return result
