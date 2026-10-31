"""Top-down piecewise linear segmentation and prefix-only feature construction."""
import numpy as np
import pandas as pd


def segment(prices, error=5., minimum=3):
    prices = np.asarray(prices, dtype=float)
    result = []; stack = [(0,len(prices)-1)]
    while stack:
        a,b = stack.pop()
        if b <= a: continue
        fit = np.linspace(prices[a],prices[b],b-a+1)
        distance = np.abs(prices[a:b+1]-fit)
        # Shared endpoints, each child must retain minimum observations.
        if b-a+1 >= 2*minimum-1:
            k = a+minimum-1 + int(np.argmax(distance[minimum-1:-(minimum-1)]))
            if distance[k-a] > error:
                stack.extend([(k,b),(a,k)]); continue
        result.append(dict(start=a,end=b,slope=float((prices[b]-prices[a])/(b-a+1)),bars=b-a+1))
    return sorted(result,key=lambda s:s['start'])


def features_at(prices, index, config, minutes=5):
    start = max(0,index-config['lookback']+1)
    local = np.asarray(prices[start:index+1])
    segments = segment(local,config['segment_error'],config['min_segment'])
    valid = [s for s in segments if s['end'] >= 4]
    if len(valid) < 4: return None
    selected = valid[-4:]
    raw = np.array([local[s['end']-4:s['end']+1] for s in selected],dtype=np.float32)
    trend = np.array([[s['slope'],s['bars']*minutes/60] for s in selected],dtype=np.float32)
    return dict(raw=raw, cnn=raw-raw[0,0], trend=trend,
                segments=[dict(s,start=s['start']+start,end=s['end']+start) for s in selected], index=index)


def prepare(frame, config, frequency):
    prices = frame.close.to_numpy(float)
    minutes = 240 if frequency=='1d' else int(frequency.rstrip('m'))
    horizon = config['label_horizon']; n=len(prices)
    split1=int(n*config['train_ratio']); split2=int(n*(config['train_ratio']+config['val_ratio']))
    if config.get('train_end_date') and config.get('validation_end_date'):
        times=pd.to_datetime(frame.timestamp)
        split1=int(times.searchsorted(pd.Timestamp(config['train_end_date'])+pd.Timedelta(days=1)))
        split2=int(times.searchsorted(pd.Timestamp(config['validation_end_date'])+pd.Timedelta(days=1)))
        if not 40<split1<split2<n-horizon:
            raise ValueError('日期切分需在数据范围内，训练、验证、测试各保留足够样本')
    stride=max(config['stride'],max(1,(n-40)//config['max_samples']))
    samples=[]
    offline=config.get('sampling_protocol')=='paper_offline'
    if offline:
        segments=segment(prices,config['segment_error'],config['min_segment'])
        for k in range(4,len(segments)):
            history=segments[k-4:k];target=segments[k];index=history[-1]['end'];matured=target['end']
            if any(s['end']<4 for s in history):continue
            raw=np.array([prices[s['end']-4:s['end']+1] for s in history],dtype=np.float32)
            partition='train' if matured<split1 else ('validation' if index>=split1 and matured<split2 else ('test' if index>=split2 else 'purged'))
            if partition=='purged':continue
            samples.append(dict(raw=raw,cnn=raw-raw[0,0],trend=np.array([[s['slope'],s['bars']*minutes/60] for s in history],dtype=np.float32),
                segments=history,index=index,label_end=matured,partition=partition,as_of=frame.timestamp.iloc[index],
                direction=float(target['slope']>0),duration=target['bars']*minutes/60))
        if len(samples)>config['max_samples']:
            samples=samples[::int(np.ceil(len(samples)/config['max_samples']))]
    for index in ([] if offline else range(40,n-horizon-1,stride)):
        feat=features_at(prices,index,config,minutes)
        if feat is None: continue
        future=prices[index+1:index+1+horizon]
        target=segment(future,config['segment_error'],config['min_segment'])[0]
        matured=index+horizon
        partition='train' if matured<split1 else ('validation' if index>=split1 and matured<split2 else ('test' if index>=split2 else 'purged'))
        if partition=='purged': continue
        feat.update(direction=float(target['slope']>0),duration=target['bars']*minutes/60,
                    label_end=matured,partition=partition,as_of=frame.timestamp.iloc[index])
        samples.append(feat)
    for part in ['train','validation','test']:
        if sum(s['partition']==part for s in samples)<5:
            raise ValueError(f'{part} 有效趋势样本不足5条，请增加数据、降低分段误差或缩短标签窗口')
    return samples,dict(train_end=split1-1,validation_end=split2-1,minutes=minutes,
                        protocol='论文式离线全样本分段；历史4段预测下一完整段；仅研究不可交易' if offline else '固定时刻历史前缀分段；未来有界窗口只生成训练标签；跨切分标签剔除',
                        research_only=offline,
                        purged_boundaries=True,stride=stride)
