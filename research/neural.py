from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from backend import store
from .trends import prepare, features_at
from .metrics import classification

torch.set_num_threads(2)


class TrendNetwork(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c; h=c['hidden']; channels=c['channels']; k=c['kernel']
        self.conv1=nn.Conv1d(5,channels,k,padding=k//2)
        self.conv2=nn.Conv1d(channels,channels,k,padding=k//2)
        self.cnn_projection=nn.Linear(channels*4,h)
        self.cell=nn.LSTMCell(2,h)
        self.lstm_projection=nn.Linear(h,h)
        self.dropout=nn.Dropout(c['dropout']); self.out=nn.Linear(h,1)

    def forward(self,raw,trend,inspect=False,mask=None):
        conv1=F.relu(self.conv1(raw.transpose(1,2))); conv2=F.relu(self.conv2(conv1))
        if mask is not None:
            conv2=conv2.clone(); conv2[:,mask,:]=0
        cp=self.cnn_projection(conv2.flatten(1))
        h=torch.zeros(trend.shape[0],self.c['hidden'],device=trend.device); state=torch.zeros_like(h)
        hs=[];cs=[];gates=[]
        for t in range(trend.shape[1]):
            z=F.linear(trend[:,t],self.cell.weight_ih,self.cell.bias_ih)+F.linear(h,self.cell.weight_hh,self.cell.bias_hh)
            i,f,g,o=z.chunk(4,1); i=i.sigmoid(); f=f.sigmoid(); g=g.tanh(); o=o.sigmoid()
            state=f*state+i*g; h=o*state.tanh()
            hs.append(h);cs.append(state);gates.append(torch.stack([i,f,g,o],1))
        lp=self.lstm_projection(h)
        merged=cp+lp if self.c['method']=='hybrid' else (cp if self.c['method']=='cnn' else lp)
        prediction=self.out(self.dropout(torch.tanh(merged))).squeeze(-1)
        if not inspect: return prediction
        def arr(x): return x.detach().cpu().numpy().tolist()
        return prediction,dict(input=arr(raw[0]),trend=arr(trend[0]),conv1=arr(conv1[0]),conv2=arr(conv2[0]),
                hidden=arr(torch.stack(hs,1)[0]),cell=arr(torch.stack(cs,1)[0]),gates=arr(torch.stack(gates,1)[0]),
                cnn_projection=arr(cp[0]),lstm_projection=arr(lp[0]),fusion=arr(merged[0]),
                conv_kernel=arr(self.conv1.weight),conv_bias=arr(self.conv1.bias),
                conv2_kernel=arr(self.conv2.weight),conv2_bias=arr(self.conv2.bias),
                output_weight=arr(self.out.weight[0]),output_bias=float(self.out.bias[0].detach()),
                head_input=arr(torch.tanh(merged)[0]),
                output=float(prediction[0].detach()),axis_mapping='batch × time(4) × price channels(5)',
                fusion_type='add_projected',active_branches=self.c['method'])


def tensor_samples(samples):
    return (torch.tensor(np.stack([s['cnn'] for s in samples]),dtype=torch.float32),
            torch.tensor(np.stack([s['trend'] for s in samples]),dtype=torch.float32))


def snapshot(model,sample,objective,checkpoint_id):
    a,b=tensor_samples([sample]);model.eval()
    with torch.no_grad(): p,view=model(a,b,True)
    view.update(checkpoint_id=checkpoint_id,sample_id=sample['index'],as_of=sample['as_of'],objective=objective,
                raw=sample['raw'].tolist(),segments=sample['segments'],source='真实检查点的固定样本前向计算')
    view['prediction']=float(p.sigmoid()[0]) if objective=='direction' else float(p[0])
    view['target']=sample[objective]
    return view


def atomic_torch(path,obj):
    path=Path(path); temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)


def fit(run,frame,meta,control,emit):
    c=run['config']; seed=c['seed']; random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    samples,protocol=prepare(frame,c,meta['frequency'])
    groups={key:[s for s in samples if s['partition']==key] for key in ['train','validation','test']}
    emit('data.prepared',dict(counts={k:len(v) for k,v in groups.items()},protocol=protocol))
    folder=store.DATA/'runs'/run['id'];folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'resume.pt'
    resumed=torch.load(checkpoint,weights_only=False) if checkpoint.exists() else None
    objectives=['direction','duration'] if c['objective']=='both' else [c['objective']]
    results={}; artifacts={}
    for objective in objectives:
        if resumed and resumed['objective'] in objectives and objectives.index(objective)<objectives.index(resumed['objective']):
            results.update(resumed.get('results',{}));artifacts.update(resumed.get('artifacts',{}));continue
        model=TrendNetwork(c);opt=torch.optim.Adagrad(model.parameters(),lr=c['learning_rate'])
        start=0
        if resumed and resumed['objective']==objective:
            model.load_state_dict(resumed['model']);opt.load_state_dict(resumed['optimizer']);start=resumed['epoch']
            torch.set_rng_state(resumed['rng']);np.random.set_state(resumed['np_rng']);random.setstate(resumed['py_rng'])
            results.update(resumed.get('results',{}));artifacts.update(resumed.get('artifacts',{}))
        train=groups['train'];validation=groups['validation']; a,b=tensor_samples(train);va,vb=tensor_samples(validation)
        target=torch.tensor([s[objective] for s in train],dtype=torch.float32)
        vt=torch.tensor([s[objective] for s in validation],dtype=torch.float32)
        def loss(pred,y):
            return F.binary_cross_entropy_with_logits(pred,y) if objective=='direction' else torch.sqrt(F.mse_loss(pred,y)+1e-8)
        for epoch in range(start,c['epochs']):
            control()
            model.train();indices=torch.randperm(len(train)) if c['shuffle'] else torch.arange(len(train))
            total=0.;regtotal=0.
            for offset in range(0,len(train),c['batch_size']):
                ids=indices[offset:offset+c['batch_size']];opt.zero_grad()
                main=loss(model(a[ids],b[ids]),target[ids])
                regular=c['l2']*sum(p.square().sum() for p in model.parameters())
                value=main+regular
                if not torch.isfinite(value): raise ValueError('损失出现非有限值，请检查数据或降低学习率')
                value.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
                total+=float(main.detach())*len(ids);regtotal+=float(regular.detach())*len(ids)
            model.eval()
            with torch.no_grad():
                vp=model(va,vb);vl=float(loss(vp,vt));p=vp.sigmoid().numpy() if objective=='direction' else vp.numpy()
            metric=classification(vt.numpy(),p) if objective=='direction' else dict(rmse=vl,mae=float(np.abs(p-vt.numpy()).mean()))
            event=dict(objective=objective,epoch=epoch+1,epochs=c['epochs'],train_loss=total/len(train),
                       regularization=regtotal/len(train),validation_loss=vl,metrics=metric)
            ckid=f'{objective}-{epoch+1}'
            state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,objective=objective,config=c,
                       rng=torch.get_rng_state(),np_rng=np.random.get_state(),py_rng=random.getstate(),results=results,artifacts=artifacts)
            atomic_torch(checkpoint,state)
            atomic_torch(folder/f'{ckid}.pt',dict(model=model.state_dict(),config=c,objective=objective))
            view=snapshot(model,train[0],objective,ckid);view['run_id']=run['id']
            store.write_json(folder/f'{ckid}.json',view)
            emit('train.epoch_end',event);emit('model.snapshot_ready',dict(checkpoint_id=ckid,objective=objective))
            store.update(run['id'],progress=(objectives.index(objective)+(epoch+1)/c['epochs'])/len(objectives),latest_checkpoint=ckid)
        with torch.no_grad():
            prediction=model(va,vb).sigmoid().numpy() if objective=='direction' else model(va,vb).numpy()
        threshold=.5
        if objective=='direction':
            threshold=max(np.arange(0,.901,.05),key=lambda x:classification(vt.numpy(),prediction,x)['fbeta'])
        test=groups['test'];ta,tb=tensor_samples(test)
        with torch.no_grad(): out=model(ta,tb);values=out.sigmoid().numpy() if objective=='direction' else out.numpy()
        y=np.array([s[objective] for s in test])
        metrics=classification(y,values,threshold) if objective=='direction' else dict(rmse=float(np.sqrt(np.mean((values-y)**2))),mae=float(np.abs(values-y).mean()),samples=len(y))
        results[objective]=dict(metrics=metrics,fixed_threshold=classification(y,values,.5) if objective=='direction' else None,
                                predictions=[dict(as_of=s['as_of'],actual=float(t),prediction=float(p)) for s,t,p in zip(test,y,values)])
        artifacts[objective]=dict(path=str(folder/f'{objective}-{c["epochs"]}.pt'),threshold=float(threshold))
    return dict(results=results,artifacts=artifacts,protocol=protocol,
                counts={k:len(v) for k,v in groups.items()},training_cutoff=frame.timestamp.iloc[protocol['validation_end']],
                deviations=['D01 上涨1下跌0','D03 样本边界实际生成','D04 ReLU与恒等池化及正则系数为实现约定',
                            protocol['protocol'],'Python/PyTorch替代原Keras'])


def load_model(artifact):
    value=torch.load(artifact['path'],map_location='cpu',weights_only=False)
    model=TrendNetwork(value['config']);model.load_state_dict(value['model']);model.eval();return model


def infer(model,frame,index,c,frequency,mask=None):
    minutes=240 if frequency=='1d' else int(frequency.rstrip('m'))
    sample=features_at(frame.close.to_numpy(float),index,c,minutes)
    if sample is None:return None
    a,b=tensor_samples([sample])
    with torch.no_grad(): p,explain=model(a,b,True,mask)
    return float(p[0]),explain


def inspection_snapshot(folder, checkpoint):
    """Enrich existing immutable views from their original checkpoint, without retraining."""
    path=Path(folder)/f'{checkpoint}.json'
    weights=Path(folder)/f'{checkpoint}.pt'
    return _inspection_snapshot(str(path),path.stat().st_mtime_ns,str(weights),weights.stat().st_mtime_ns if weights.exists() else 0)


from functools import lru_cache


@lru_cache(maxsize=32)
def _inspection_snapshot(path, json_mtime, weights, weights_mtime):
    view=store.read_json(path)
    if 'input' not in view or 'output_weight' in view or not weights_mtime:return view
    model=load_model({'path':weights})
    raw=torch.tensor([view['input']],dtype=torch.float32)
    trend=torch.tensor([view['trend']],dtype=torch.float32)
    with torch.no_grad():_,details=model(raw,trend,True)
    # Preserve the saved fixed sample, activations and metadata. Only add the omitted parameters.
    return dict(view,**{key:details[key] for key in ['conv2_kernel','conv2_bias','output_weight','output_bias','head_input']})
