import asyncio
import csv
import hashlib
import html
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, File, Form, UploadFile, Request, Header
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr
from . import store,data
from .schemas import ExperimentConfig,Acquisition,Publish,ExecutionConfig,MACDStrategy

LOCK=threading.RLock();PROCESSES={};PREDICTORS={};STOP=threading.Event()


def scheduler():
    while not STOP.wait(.3):
        queued=[r for r in store.listing('experiment') if r['status']=='QUEUED']
        if not queued:continue
        run=queued[-1];id=run['id'];logs=store.DATA/'logs';logs.mkdir(exist_ok=True)
        with (logs/f'{id}.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-m','backend.worker',id],cwd=store.ROOT,stdout=log,stderr=log)
            PROCESSES[id]=proc
            try:proc.wait(timeout=(3600 if run['config'].get('provider')=='tushare' else 90) if run['kind']=='acquisition' else None)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait();store.update(id,status='FAILED',error='数据源获取超时；可重试复用分块缓存或上传历史文件')
            current=store.get(id)
            if current['status'] in ['RUNNING','QUEUED']:
                store.update(id,status='FAILED',error='任务进程异常退出，检查运行日志后可从检查点恢复')
        PROCESSES.pop(id,None)


def forecast_scheduler():
    from trading.auto_forecast import effective_schedule,is_due,run_once
    while not STOP.wait(10):
        settings=global_forecast_schedule()
        run_once([config for config in store.listing('forecast_schedule') if is_due(effective_schedule(config,settings))])
        if STOP.wait(20):break


@asynccontextmanager
async def lifespan(app):
    data.bootstrap()
    data.audit_legacy_acquisitions()
    for r in store.listing('experiment'):
        if r['status'] in ['RUNNING','PAUSED']:
            store.update(r['id'],status='INTERRUPTED',command=None,error='服务重启，可从检查点继续')
    STOP.clear();thread=threading.Thread(target=scheduler,daemon=True);thread.start()
    threading.Thread(target=forecast_scheduler,daemon=True).start()
    yield
    STOP.set()
    for id,p in list(PROCESSES.items()):
        p.terminate();store.update(id,status='INTERRUPTED',command=None)


app=FastAPI(title='趋势研究所 API',version='1.0.0',lifespan=lifespan)


@app.exception_handler(ValueError)
async def value_error(request,error):
    return JSONResponse(status_code=400,content=dict(code='VALIDATION_ERROR',message=str(error)))


@app.middleware('http')
async def local_boundary(request,call_next):
    origin=request.headers.get('origin')
    if origin and origin not in ['http://127.0.0.1:8765','http://localhost:8765','http://127.0.0.1:5173','http://localhost:5173']:
        return JSONResponse(status_code=403,content={'message':'本地服务拒绝来自其他站点的请求'})
    return await call_next(request)


@app.get('/api/v1/health')
def health():return dict(status='ok',version='1.0.0',deployment='local-single-user')


@app.get('/api/v1/methods')
def methods():
    from research.registry import info
    return info()


@app.get('/api/v1/overview')
def overview():
    ds=store.listing('dataset');runs=store.listing('experiment');strategies=store.listing('strategy')
    return dict(datasets=len(ds),experiments=len([r for r in runs if r['kind']=='training']),
                completed=len([r for r in runs if r['status']=='COMPLETED']),strategies=len(strategies),
                backtests=len(store.listing('backtest')),recent=runs[:6])


@app.get('/api/v1/data-providers')
def providers():
    from .history_provider import token
    return dict(tushare_configured=bool(token()),akshare_minutes='近期窗口，不能回溯至2023',history_requires='Tushare 期货历史分钟权限；IF0还需主力映射权限')


class ProviderToken(BaseModel):
    token:SecretStr


@app.post('/api/v1/data-providers/tushare')
def configure_provider(body:ProviderToken):
    from .history_provider import set_token
    value=body.token.get_secret_value().strip()
    if not 10<=len(value)<=200:raise ValueError('Token长度不正确')
    set_token(value)
    return dict(configured=True)


@app.get('/api/v1/trash')
def trash():
    return [dict(id=r['id'],kind=kind,name=r.get('name') or r.get('config',{}).get('name') or ('模拟会话 · ' if kind=='session' else '回测 · ')+r.get('created_at',r['id']),deleted_at=r['deleted_at'])
        for kind in ['dataset','experiment','strategy','backtest','session','forecast'] for r in store.listing(kind,include_deleted=True) if r.get('deleted_at')]


@app.post('/api/v1/trash/{id}/restore')
def restore_record(id:str):
    with LOCK:
        r=store.get(id)
        if not r.get('deleted_at'):raise ValueError('记录未被删除')
        return store.update(id,deleted_at=None)


def soft_delete(id,kind):
    with LOCK:
        r=store.get(id,kind)
        if kind=='experiment' and r.get('status') in ['RUNNING','QUEUED','PAUSED']:
            raise ValueError('请先取消任务，待任务停止后再删除模型')
        if kind=='dataset' and any(x['config'].get('dataset_id')==id and x['status'] in ['QUEUED','RUNNING','PAUSED'] for x in store.listing('experiment')):
            raise ValueError('该数据正在训练，请先停止相关任务')
        store.update(id,deleted_at=store.now())
        if kind=='session':PREDICTORS.pop(id,None)
    return dict(id=id,deleted=True,recoverable=True,message='已移入回收站；已有策略、会话和报告引用仍保留，可恢复')


@app.delete('/api/v1/datasets/{id}')
def delete_dataset(id:str):return soft_delete(id,'dataset')


@app.delete('/api/v1/experiments/{id}')
def delete_experiment(id:str):return soft_delete(id,'experiment')


@app.delete('/api/v1/strategies/{id}')
def delete_strategy(id:str):return soft_delete(id,'strategy')


@app.delete('/api/v1/backtests/{id}')
def delete_backtest(id:str):return soft_delete(id,'backtest')


@app.delete('/api/v1/sessions/{id}')
def delete_session(id:str):return soft_delete(id,'session')


@app.get('/api/v1/datasets')
def datasets():return [{k:v for k,v in d.items() if k!='path'} for d in store.listing('dataset')]


@app.get('/api/v1/datasets/template')
def template():
    text='timestamp,open,high,low,close,volume,market_close\n2024-01-02T09:35:00,3500,3504,3498,3502,12000,3001\n'
    return StreamingResponse(iter([text]),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=upload-template.csv'})


@app.get('/api/v1/datasets/{id}')
def dataset(id:str,offset:int=0,limit:int=2000):
    frame,meta=data.load(id);offset=max(0,offset);limit=min(max(limit,1),10000)
    # Downsample display only; source file and algorithm always use full-resolution bars.
    if offset==0 and len(frame)>limit:
        import numpy as np
        indices=np.unique(np.linspace(0,len(frame)-1,limit).astype(int));preview=frame.iloc[indices]
    else:preview=frame.iloc[offset:offset+limit]
    return dict(meta={k:v for k,v in meta.items() if k!='path'},bars=preview.to_dict('records'))


@app.post('/api/v1/datasets/import')
async def import_data(file:UploadFile=File(),name:str=Form(),symbol:str=Form(),frequency:str=Form()):
    if frequency not in ['1d','1m','5m','15m','30m','60m']:raise ValueError('不支持的频率')
    content=await file.read(50*1024*1024+1)
    if len(content)>50*1024*1024:raise ValueError('文件大小上限50MB')
    return data.upload(content,file.filename or '',name[:100],symbol[:30],frequency)


def submit(kind,config,key=None):
    fingerprint=hashlib.sha256(store.encode(dict(kind=kind,config=config)).encode()).hexdigest()
    with LOCK:
        if key:
            with store.connect() as db:
                row=db.execute('SELECT fingerprint,result FROM requests WHERE key=?',(key,)).fetchone()
                if row:
                    if row[0]!=fingerprint:raise ValueError('幂等键已用于其他配置')
                    return store.get(row[1])
        id=store.uid('run_');codehash=hashlib.sha256()
        for folder in ['backend','research','trading']:
            for path in sorted((store.ROOT/folder).rglob('*.py')):codehash.update(path.read_bytes())
        run=dict(id=id,kind=kind,config=config,status='QUEUED',progress=0,created_at=store.now(),command=None,
                 code_version=codehash.hexdigest(),dependency_lock=store.checksum(store.ROOT/'requirements.lock') if (store.ROOT/'requirements.lock').exists() else None)
        folder=store.DATA/'runs'/id;folder.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(folder/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for directory in ['backend','research','trading']:
                for path in (store.ROOT/directory).rglob('*.py'):archive.write(path,path.relative_to(store.ROOT))
            for name in ['requirements.txt','requirements.lock']:
                path=store.ROOT/name
                if path.exists():archive.write(path,name)
        run['source_archive_hash']=store.checksum(folder/'source.zip')
        store.put('experiment',run);store.event(id,'task.state_changed',dict(status='QUEUED'))
        if key:
            with store.connect() as db:db.execute('INSERT INTO requests VALUES(?,?,?)',(key,fingerprint,id))
    return run


class Slice(BaseModel):
    name:str=Field(max_length=100)
    start:str
    end:str


@app.post('/api/v1/datasets/{id}/slice')
def slice_dataset(id:str,body:Slice):
    import pandas as pd
    frame,meta=data.load(id)
    sliced=frame[(frame.timestamp>=pd.Timestamp(body.start).isoformat()) & (frame.timestamp<(pd.Timestamp(body.end)+pd.Timedelta(days=1)).isoformat())]
    saved=data.save_dataset(sliced,body.name,meta['symbol'],meta['frequency'],meta['source'],synthetic=meta['synthetic'])
    saved['parent_id']=id;saved['parent_hash']=meta['hash'];return store.put('dataset',saved)


@app.post('/api/v1/datasets/akshare')
def fetch(c:Acquisition,idempotency_key:str|None=Header(None)):
    return submit('acquisition',c.model_dump(),idempotency_key)


def validate(c):
    frame,meta=data.load(c['dataset_id'])
    from research.registry import METHODS
    method=METHODS[c['method']]
    if method.frequency!='any' and method.frequency!=meta['frequency']:raise ValueError('该方法与数据频率不匹配')
    if method.validator:return method.validator(frame,meta,c)
    if c['method']=='stgp':
        if meta['frequency']!='1d':raise ValueError('论文二使用日频数据，请选择日线数据版本')
        from research.gp import factors,rolling_start
        market=data.load(c['market_dataset_id'])[0] if c.get('market_dataset_id') else None
        f=factors(frame,c,market)
        start=rolling_start(frame,c,f.notna().all(axis=1).to_numpy())
        return dict(samples=int(f.notna().all(axis=1).sum()),features=len(f.columns),warnings=meta['warnings'],first_prediction=frame.timestamp.iloc[start])
    from research.trends import prepare
    samples,protocol=prepare(frame,c,meta['frequency'])
    return dict(samples=len(samples),counts={k:sum(s['partition']==k for s in samples) for k in ['train','validation','test']},protocol=protocol,warnings=meta['warnings'])


@app.post('/api/v1/experiments/validate')
def validate_experiment(c:ExperimentConfig):return validate(c.model_dump())


@app.post('/api/v1/experiments')
def experiment(c:ExperimentConfig,idempotency_key:str|None=Header(None)):
    validate(c.model_dump());return submit('training',c.model_dump(),idempotency_key)


@app.get('/api/v1/experiments')
def experiments():
    return [{k:v for k,v in r.items() if k!='result'} for r in store.listing('experiment')]


@app.get('/api/v1/experiments/{id}')
def get_experiment(id:str):return store.get(id,'experiment')


class Control(BaseModel):
    action:str


@app.post('/api/v1/experiments/{id}/controls')
def control(id:str,body:Control):
    run=store.get(id,'experiment');action=body.action
    with LOCK:
        if action=='cancel' and run['status']=='QUEUED':return store.update(id,status='CANCELLED',command='cancel')
        if action=='pause' and run['status']=='RUNNING':return store.update(id,command='pause')
        if action=='resume' and run['status']=='PAUSED':return store.update(id,command=None)
        if action=='resume' and run['status'] in ['FAILED','INTERRUPTED']:
            return store.update(id,status='QUEUED',command=None,error=None)
        if action=='cancel' and run['status'] in ['RUNNING','PAUSED']:return store.update(id,command='cancel')
    raise ValueError('当前状态不允许此操作')


@app.get('/api/v1/experiments/{id}/events')
async def event_stream(id:str,request:Request,after:int=0,last_event_id:str|None=Header(None)):
    store.get(id,'experiment');cursor=max(after,int(last_event_id or 0))
    async def stream():
        nonlocal cursor
        while not await request.is_disconnected():
            batch=store.events(id,cursor)
            for event in batch:
                cursor=event['event_id'];yield f'id: {cursor}\ndata: {store.encode(event)}\n\n'
            if not batch:
                if store.get(id)['status'] in ['COMPLETED','FAILED','CANCELLED','INTERRUPTED']:break
                yield ': heartbeat\n\n'
            await asyncio.sleep(.4)
    return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache'})


@app.get('/api/v1/experiments/{id}/history')
def history(id:str,after:int=0):return store.events(id,after)


@app.get('/api/v1/experiments/{id}/snapshots')
def snapshots(id:str,checkpoint:str|None=None):
    run=store.get(id,'experiment');folder=store.DATA/'runs'/id
    paths=sorted(folder.glob('*.json'),key=lambda p:p.stat().st_mtime)
    ids=[p.stem for p in paths if p.stem not in ['result','gp-result']]
    if not ids:return dict(checkpoints=[],snapshot=None)
    selected=checkpoint or run.get('latest_checkpoint') or ids[-1]
    if selected not in ids:raise ValueError('检查点不存在')
    if run['config']['method'] in ['hybrid','cnn','lstm']:
        from research.neural import inspection_snapshot
        view=inspection_snapshot(folder,selected)
    else:view=store.read_json(folder/f'{selected}.json')
    return dict(checkpoints=ids,snapshot=view)


@app.get('/api/v1/experiments/{id}/samples')
def samples(id:str,index:int=0):
    run=store.get(id,'experiment');frame,meta=data.load(run['config']['dataset_id'])
    if run['config']['method']=='stgp':raise ValueError('GP方法请查看表达式树检查点')
    from research.trends import prepare
    values,protocol=prepare(frame,run['config'],meta['frequency']);s=values[max(0,min(index,len(values)-1))]
    return dict(count=len(values),sample={k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in s.items()},protocol=protocol)


@app.get('/api/v1/experiments/{id}/factors')
def factor_sample(id:str,index:int|None=None):
    run=store.get(id,'experiment');c=run['config']
    if c['method']!='stgp':raise ValueError('该实验不是遗传规划方法')
    from research.gp import factors
    frame,meta=data.load(c['dataset_id'])
    market=data.load(c['market_dataset_id'])[0] if c.get('market_dataset_id') else None
    features=factors(frame,c,market)
    i=min(max(121,index if index is not None else int(len(frame)*c['train_ratio'])),len(frame)-1)
    values=features.iloc[i].dropna().to_dict()
    return dict(as_of=frame.timestamp.iloc[i],index=i,base_date=c['base_date'],feature_count=len(features.columns),values=values,
                price=float(frame.close.iloc[i]),volume=float(frame.volume.iloc[i]),source='全部因子至少滞后一个交易日；市场数据按日期精确对齐')


class Explain(BaseModel):
    checkpoint:str
    mask:int=Field(ge=0,le=63)


@app.post('/api/v1/experiments/{id}/explain')
def explain(id:str,body:Explain):
    run=store.get(id,'experiment');folder=store.DATA/'runs'/id
    allowed={p.stem for p in folder.glob('*.pt') if p.stem!='resume'}
    if body.checkpoint not in allowed:raise ValueError('请选择已有神经网络检查点')
    if body.mask>=run['config']['channels']:raise ValueError('通道编号超出当前模型范围')
    from research.neural import load_model,tensor_samples
    from research.trends import prepare
    import torch
    frame,meta=data.load(run['config']['dataset_id']);values,_=prepare(frame,run['config'],meta['frequency'])
    sample=next(s for s in values if s['partition']=='train');a,b=tensor_samples([sample])
    model=load_model({'path':str(folder/f'{body.checkpoint}.pt')})
    with torch.no_grad():pred,view=model(a,b,True,body.mask)
    objective=body.checkpoint.split('-')[0]
    view.update(checkpoint_id=body.checkpoint,sample_id=sample['index'],as_of=sample['as_of'],objective=objective,
                raw=sample['raw'].tolist(),segments=sample['segments'],masked_channel=body.mask,run_id=id,
                prediction=float(pred.sigmoid()[0]) if objective=='direction' else float(pred[0]))
    return view


class ForecastRequest(BaseModel):
    dataset_id:str
    strategy_ids:list[str]=Field(min_length=1,max_length=20)


@app.post('/api/v1/forecasts')
def create_forecasts(body:ForecastRequest):
    from trading.forecast import save_forecast
    results=[];errors=[]
    for strategy_id in dict.fromkeys(body.strategy_ids):
        try:
            with LOCK: results.append(save_forecast(body.dataset_id,strategy_id))
        except (ValueError,OSError) as e: errors.append(dict(strategy_id=strategy_id,message=str(e)))
    return dict(results=results,errors=errors)


@app.get('/api/v1/forecasts')
def forecasts():
    from trading.forecast import with_outcome
    datasets=sorted(store.listing('dataset'),key=lambda d:(d['end'],d['created_at']),reverse=True)
    return [with_outcome(r,datasets) for r in store.listing('forecast')[:200]]


class ForecastSchedule(BaseModel):
    strategy_id:str
    enabled:bool=True
    schedule_mode:Literal['interval','daily']|None=None
    interval_minutes:int|None=Field(None,ge=1,le=1440)
    fixed_time:str|None=Field(None,pattern=r'^(?:[01][0-9]|2[0-3]):[0-5][0-9]$')
    use_global:bool|None=None


@app.delete('/api/v1/forecasts/{id}')
def delete_forecast(id:str):return soft_delete(id,'forecast')


class GlobalForecastSchedule(BaseModel):
    schedule_mode:Literal['interval','daily']='interval'
    interval_minutes:int=Field(5,ge=1,le=1440)
    fixed_time:str=Field('15:35',pattern=r'^(?:[01][0-9]|2[0-3]):[0-5][0-9]$')


@app.get('/api/v1/forecast-schedule-settings')
def global_forecast_schedule():
    from trading.auto_forecast import DEFAULT_SCHEDULE
    saved=store.listing('forecast_settings')
    return {**DEFAULT_SCHEDULE,**(saved[0] if saved else {})}


@app.post('/api/v1/forecast-schedule-settings')
def save_global_forecast_schedule(body:GlobalForecastSchedule):
    from trading.auto_forecast import uses_global
    values=body.model_dump();existing=store.listing('forecast_settings')
    if existing:
        old=existing[0]
        changed=any(old.get(key)!=value for key,value in values.items())
        result=store.update(old['id'],**values)
    else:
        changed=True
        result=store.put('forecast_settings',dict(id=store.uid('forecast_settings_'),**values,created_at=store.now()))
    if changed:
        for plan in store.listing('forecast_schedule'):
            if uses_global(plan):store.update(plan['id'],last_checked=None,last_error=None)
    return result


@app.get('/api/v1/forecast-schedules')
def forecast_schedules():return store.listing('forecast_schedule')


@app.post('/api/v1/forecast-schedules')
def forecast_schedule(body:ForecastSchedule):
    strategy=store.get(body.strategy_id,'strategy')
    if strategy.get('deleted_at') or strategy['frequency']!='1d':raise ValueError('请选择未删除的日线策略')
    existing=next((s for s in store.listing('forecast_schedule') if s['strategy_id']==body.strategy_id),None)
    from trading.auto_forecast import uses_global
    prior_global=uses_global(existing) if existing else True
    use_global=body.use_global if body.use_global is not None else (False if body.schedule_mode is not None else prior_global)
    settings=dict(schedule_mode=body.schedule_mode or (existing or {}).get('schedule_mode','interval'),
        interval_minutes=body.interval_minutes or (existing or {}).get('interval_minutes',5),
        fixed_time=body.fixed_time or (existing or {}).get('fixed_time','15:35'))
    if existing:
        changed=any(existing.get(k,default)!=v for (k,v),default in zip(settings.items(),('interval',5,'15:35')))
        changes=dict(enabled=body.enabled,use_global=use_global,**settings)
        if changed or prior_global!=use_global or (body.enabled and not existing.get('enabled')):changes.update(last_checked=None,last_error=None)
        return store.update(existing['id'],**changes)
    return store.put('forecast_schedule',dict(id=store.uid('fs_'),strategy_id=body.strategy_id,
        symbol=strategy['symbol'],enabled=body.enabled,use_global=use_global,last_error=None,created_at=store.now(),**settings))


@app.get('/api/v1/strategies')
def strategies():return store.listing('strategy')


class DowStrategy(BaseModel):
    name:str=Field('道氏理论 · 双指数月线',min_length=1,max_length=100)
    dataset_id:str
    sh_dataset_id:str
    sz_dataset_id:str


@app.post('/api/v1/strategies/dow')
def create_dow(p:DowStrategy):
    refs=[]
    for id in [p.dataset_id,p.sh_dataset_id,p.sz_dataset_id]:
        frame,meta=data.load(id)
        if meta.get('deleted_at') or meta['frequency']!='1d':raise ValueError('请选择未删除的日线数据')
        refs.append(meta)
    target,sh,sz=refs
    if target['symbol'].lower() not in ('sh000300','000300','000300.sh'):
        raise ValueError('道氏双指数参考策略目前仅支持沪深300日线')
    if sh['symbol'].lower() not in ('sh000001','000001.sh') or sz['symbol'].lower() not in ('sz399001','399001.sz'):
        raise ValueError('请选择上证指数 sh000001 和深证成指 sz399001 日线')
    return store.put('strategy',dict(id=store.uid('strategy_'),name=p.name,method='dow',run_id=None,
        dataset_id=target['id'],symbol=target['symbol'],frequency='1d',training_cutoff='',
        sh_dataset_id=sh['id'],sz_dataset_id=sz['id'],forecast_only=True,created_at=store.now(),
        synthetic=any(d['synthetic'] for d in refs)))


@app.post('/api/v1/strategies/macd')
def create_macd(p:MACDStrategy):
    frame,meta=data.load(p.dataset_id)
    if meta.get('deleted_at'):raise ValueError('数据已删除，请先恢复')
    if len(frame)<p.slow+p.signal:raise ValueError('行情数据不足以完成 MACD 预热并执行交易')
    return store.put('strategy',dict(id=store.uid('strategy_'),name=p.name,method='macd',run_id=None,
        dataset_id=meta['id'],symbol=meta['symbol'],frequency=meta['frequency'],training_cutoff='',
        parameters=dict(fast=p.fast,slow=p.slow,signal=p.signal),stop_loss=p.stop_loss,
        created_at=store.now(),synthetic=meta['synthetic']))


@app.post('/api/v1/strategies')
def publish(p:Publish):
    run=store.get(p.run_id,'experiment')
    if run.get('deleted_at'):raise ValueError('模型已删除，请先在回收站恢复')
    if run['status']!='COMPLETED' or run['kind']!='training':raise ValueError('只能发布已完成的训练实验')
    if run.get('result',{}).get('protocol',{}).get('research_only'):
        raise ValueError('论文离线分段实验可能包含未来端点信息，仅用于研究比较；请用历史前缀协议训练后发布交易策略')
    if run['config']['method'] in ['hybrid','cnn','lstm'] and 'direction' not in run['result']['artifacts']:raise ValueError('交易策略必须包含方向模型')
    frame,meta=data.load(run['config']['dataset_id']);cutoff=run['result']['training_cutoff']
    if p.duration_run_id:
        duration=store.get(p.duration_run_id,'experiment')
        if duration['status']!='COMPLETED' or 'duration' not in duration.get('result',{}).get('artifacts',{}):raise ValueError('时长实验未完成')
        if duration.get('deleted_at') or duration.get('result',{}).get('protocol',{}).get('research_only'):raise ValueError('时长模型已删除或含未来端点，不能发布')
        if run['config']['method'] not in ['hybrid','cnn','lstm']:raise ValueError('该策略方法不支持附加神经时长模型')
        if duration['config']['dataset_id']!=run['config']['dataset_id']:raise ValueError('时长模型与方向模型须使用同一数据版本')
        if any(duration['config'].get(k)!=run['config'].get(k) for k in ['label_horizon','segment_error','min_segment','sampling_protocol']):raise ValueError('方向与时长模型的标签窗口、分段误差和样本协议须一致，才能共同核验')
        cutoff=max(cutoff,duration['result']['training_cutoff'])
    result=run['result'];hashes={}
    duration_hashes={key:store.checksum(a['path']) for key,a in duration['result']['artifacts'].items()} if p.duration_run_id else {}
    for key,artifact in result['artifacts'].items():hashes[key]=store.checksum(artifact['path'])
    return store.put('strategy',dict(id=store.uid('strategy_'),**p.model_dump(),symbol=meta['symbol'],frequency=meta['frequency'],
        dataset_id=meta['id'],training_cutoff=cutoff,artifact_hashes=hashes,duration_artifact_hashes=duration_hashes,method=run['config']['method'],created_at=store.now(),synthetic=meta['synthetic']))


def session_inputs(c):
    frame,meta=data.load(c['dataset_id']);strategy=store.get(c['strategy_id'],'strategy') if c.get('strategy_id') else None
    if c['mode']=='futures':
        if 'settlement' not in frame:raise ValueError('期货模式需上传settlement结算价列，或改用名义本金研究模式')
        if 'contract_id' in frame and frame.contract_id.nunique()>1:
            raise ValueError('当前保证金账户支持单一合约；多合约数据请先拆分，连续序列只用于研究回放')
        if meta['symbol'].upper().endswith('0') or meta['symbol'].startswith('DEMO'):
            raise ValueError('连续合约或教学数据仅支持研究模式；期货账户请上传真实单合约行情')
    if strategy:
        if strategy.get('deleted_at'):raise ValueError('策略已删除，请先恢复或选择其他策略')
        if strategy.get('forecast_only'):raise ValueError('道氏双指数策略目前仅用于每日预测，不支持回测或模拟交易')
        if strategy['symbol']!=meta['symbol'] or strategy['frequency']!=meta['frequency']:raise ValueError('策略与数据标的或频率不匹配')
        if strategy.get('method')!='macd':
            run=store.get(strategy['run_id'])
            for key,artifact in run['result']['artifacts'].items():
                if store.checksum(artifact['path'])!=strategy['artifact_hashes'][key]:raise ValueError('模型产物校验失败')
        if strategy.get('duration_run_id'):
            duration=store.get(strategy['duration_run_id'],'experiment')
            for key,artifact in duration['result']['artifacts'].items():
                if store.checksum(artifact['path'])!=strategy.get('duration_artifact_hashes',{}).get(key):raise ValueError('时长模型产物校验失败')
        eligible=frame.index[frame.timestamp>strategy['training_cutoff']].tolist()
        if not eligible:raise ValueError('没有晚于训练及阈值选择截止时刻的样本外数据')
        start=c.get('start_index') if c.get('start_index') is not None else eligible[0]
        if start not in eligible:raise ValueError('回放时间必须晚于模型训练及阈值选择截止，避免未来信息')
    else:start=c.get('start_index') or 0
    if c.get('start_date'):
        import pandas as pd
        requested=pd.Timestamp(c['start_date'])
        if requested.normalize()<pd.Timestamp(frame.timestamp.iloc[0]).normalize():raise ValueError('开始日期早于此数据版本范围，请先导入对应历史行情')
        matches=frame.index[pd.to_datetime(frame.timestamp)>=requested].tolist()
        if not matches:raise ValueError('开始日期晚于数据截止日期')
        start=matches[0]
        if strategy and frame.timestamp.iloc[start]<=strategy['training_cutoff']:
            raise ValueError('所选历史日期早于或等于模型训练及选参截止；请训练更早截止的模型，避免未来信息')
    if start>=len(frame)-1:raise ValueError('回放起点之后不足两条行情')
    return frame,meta,strategy,start


@app.post('/api/v1/sessions')
def create_session(body:ExecutionConfig):
    from trading.engine import create_state
    c=body.model_dump();frame,meta,strategy,start=session_inputs(c)
    session=dict(id=store.uid('session_'),config=c,state=create_state(c,frame,start),created_at=store.now())
    session['engine_version']=store.checksum(store.ROOT/'trading'/'engine.py')
    return store.put('session',session)


@app.get('/api/v1/sessions')
def sessions():return [dict(id=s['id'],config=s['config'],status=s['state']['status'],created_at=s['created_at']) for s in store.listing('session')]


def public_session(s):
    from trading.engine import summary
    frame,meta=data.load(s['config']['dataset_id']);index=s['state']['index']
    from trading.assessment import assess
    evaluation=assess(s['state'],frame,meta['frequency'])
    latest=(s['state'].get('observations') or s['state']['signals'] or [None])[-1]
    return dict(**s,replay_start=str(frame.timestamp.iloc[s['state']['start_index']]),evaluation=evaluation,latest_prediction=latest,bars=frame.iloc[max(0,index-500):index+1].to_dict('records'),meta={k:v for k,v in meta.items() if k!='path'},
                total_bars=len(frame),metrics=summary(s['state'],frame,s['config']))


@app.get('/api/v1/sessions/{id}')
def get_session(id:str):return public_session(store.get(id,'session'))


class Step(BaseModel):
    bars:int=Field(1,ge=1,le=100)


@app.post('/api/v1/sessions/{id}/step')
def step_session(id:str,body:Step):
    from trading.engine import step
    from trading.predictor import Predictor
    with LOCK:
        s=store.get(id,'session')
        if s.get('deleted_at'):raise ValueError('会话已删除，请先从回收站恢复后再继续')
        frame,meta=data.load(s['config']['dataset_id'])
        if s.get('engine_version') and s['engine_version']!=store.checksum(store.ROOT/'trading'/'engine.py'):
            raise ValueError('该会话绑定的撮合引擎版本已变化，请新建会话；旧记录仍可查看')
        predictor=None
        if s['config']['control']=='auto':
            if id not in PREDICTORS:PREDICTORS[id]=Predictor(store.get(s['config']['strategy_id']),frame,meta)
            predictor=PREDICTORS[id]
        for _ in range(body.bars):step(s['state'],frame,s['config'],predictor)
        store.put('session',s)
    return public_session(s)


class Order(BaseModel):
    target:int=Field(ge=-1,le=1)
    key:str=Field(min_length=8,max_length=100)


@app.post('/api/v1/sessions/{id}/orders')
def order(id:str,body:Order):
    with LOCK:
        s=store.get(id,'session');state=s['state']
        if s.get('deleted_at'):raise ValueError('会话已删除，请先从回收站恢复后再继续')
        if s['config']['control']!='manual':raise ValueError('请在独立手动账户下单，自动策略账户不接受手动委托')
        if state['status']=='COMPLETED':raise ValueError('会话已结束')
        if state['index']<state['start_index']:raise ValueError('请先推进一条行情后提交委托')
        if body.key in state['seen_keys']:return public_session(s)
        if state['pending']:raise ValueError('已有待执行委托，请推进或先处理该委托')
        state['seen_keys'].append(body.key)
        frame,_=data.load(s['config']['dataset_id'])
        state['pending']=dict(target=body.target,reason='手动模拟委托',submitted_at=str(frame.timestamp.iloc[state['index']]))
        store.put('session',s)
    return public_session(s)


@app.post('/api/v1/backtests')
def backtest(body:ExecutionConfig):
    from trading.engine import create_state,step,summary
    from trading.predictor import Predictor
    c=body.model_dump();c['observe_each_bar']=False;frame,meta,strategy,start=session_inputs(c)
    if c['control']!='auto':raise ValueError('回测请选择自动策略')
    predictor=Predictor(strategy,frame,meta);state=create_state(c,frame,start)
    while state['status']!='COMPLETED':step(state,frame,c,predictor)
    result=dict(id=store.uid('bt_'),config=c,created_at=store.now(),state=state,metrics=summary(state,frame,c),
                dataset_hash=meta['hash'],strategy_version=strategy['id'],source=meta['source'],synthetic=meta['synthetic'])
    result['engine_version']=store.checksum(store.ROOT/'trading'/'engine.py')
    return store.put('backtest',result)


@app.get('/api/v1/backtests')
def backtests():return [{k:v for k,v in b.items() if k!='state'} for b in store.listing('backtest')]


@app.get('/api/v1/backtests/{id}')
def get_backtest(id:str):return store.get(id,'backtest')


@app.get('/api/v1/backtests/{id}/trades.csv')
def trades(id:str):
    b=store.get(id,'backtest');out=io.StringIO();rows=b['state']['orders']
    fields=['id','signal_id','submitted_at','filled_at','status','side','action','quantity','price','fee','realized_pnl','reason']
    writer=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return StreamingResponse(iter(['\ufeff'+out.getvalue()]),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename={id}-trades.csv'})


class Report(BaseModel):
    run_id:str
    backtest_id:str|None=None


@app.post('/api/v1/reports')
def report(body:Report):
    run=store.get(body.run_id,'experiment');bt=store.get(body.backtest_id,'backtest') if body.backtest_id else None
    if bt and store.get(bt['config']['strategy_id'])['run_id']!=run['id']:raise ValueError('回测与实验不对应')
    r=dict(id=store.uid('report_'),run_id=run['id'],backtest_id=body.backtest_id,name=run['config']['name']+' · 复现报告',
           created_at=store.now(),run=run,backtest=bt)
    return store.put('report',r)


@app.get('/api/v1/reports')
def reports():return [{k:v for k,v in r.items() if k not in ['run','backtest']} for r in store.listing('report')]


@app.get('/api/v1/reports/{id}/download')
def download_report(id:str):
    r=store.get(id,'report');run=r['run'];meta=store.get(run['config']['dataset_id']);bt=r['backtest']
    from .reporting import render
    text=render(r,meta)
    return HTMLResponse(text,headers={'Content-Disposition':f'attachment; filename={id}.html'})


dist=store.ROOT/'frontend'/'dist'
if dist.exists():app.mount('/',StaticFiles(directory=dist,html=True),name='frontend')
