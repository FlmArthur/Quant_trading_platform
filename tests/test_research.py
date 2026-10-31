import copy
import numpy as np
import pytest
import torch
from backend import store,data
from backend.schemas import ExperimentConfig
from research.trends import segment,features_at,prepare
from research.neural import TrendNetwork,fit
from research.gp import primitives,predict_tree,scores,raw_fitness,fit as gp_fit
from deap import gp


@pytest.fixture(scope='module')
def datasets():
    data.bootstrap()
    return {d['frequency']:d for d in store.listing('dataset')}


def config(datasets,method='hybrid',**changes):
    d=datasets['1d' if method=='stgp' else '5m']
    return ExperimentConfig(dataset_id=d['id'],method=method,epochs=3,hidden=8,channels=4,
             population=10,generations=2,predict_window=60,base_date=d['start'][:10],**changes).model_dump()


def test_segment_hand_computed():
    s=segment([1,2,3,2,1],.1,2)
    assert [(x['start'],x['end']) for x in s]==[(0,2),(2,4)]
    assert s[0]['slope']==pytest.approx(2/3)


def test_future_prices_cannot_change_prefix(datasets):
    c=config(datasets);frame,_=data.load(c['dataset_id']);p=frame.close.to_numpy(copy=True)
    a=features_at(p,600,c);p[601:]*=15;b=features_at(p,600,c)
    np.testing.assert_array_equal(a['cnn'],b['cnn']);np.testing.assert_array_equal(a['trend'],b['trend'])


def test_labels_are_purged_at_boundaries(datasets):
    c=config(datasets);frame,meta=data.load(c['dataset_id']);samples,protocol=prepare(frame,c,meta['frequency'])
    assert all(s['label_end']<=protocol['train_end'] for s in samples if s['partition']=='train')
    assert all(s['index']>protocol['train_end'] and s['label_end']<=protocol['validation_end'] for s in samples if s['partition']=='validation')
    assert all(s['index']>protocol['validation_end'] for s in samples if s['partition']=='test')
    for s in samples:assert max(x['end'] for x in s['segments'])<=s['index']


def test_parallel_fusion_and_real_lstm_gates(datasets):
    torch.manual_seed(3);c=config(datasets);m=TrendNetwork(c);m.eval()
    raw=torch.randn(2,4,5);trend=torch.randn(2,4,2)
    pred,view=m(raw,trend,True)
    np.testing.assert_allclose(view['fusion'],np.array(view['cnn_projection'])+view['lstm_projection'],rtol=1e-5,atol=1e-6)
    h=torch.zeros(2,c['hidden']);state=torch.zeros_like(h)
    for i in range(4):
        h,state=m.cell(trend[:,i],(h,state))
        np.testing.assert_allclose(view['hidden'][i],h[0].detach(),atol=1e-6)
        np.testing.assert_allclose(view['cell'][i],state[0].detach(),atol=1e-6)
    pred.sum().backward()
    assert m.conv1.weight.grad.abs().sum()>0
    assert m.cell.weight_ih.grad.abs().sum()>0
    assert m.cnn_projection.weight.grad.abs().sum()>0


def test_gp_types_and_shared_weight_scoring():
    p=primitives(['P1','MA5']);tree=gp.PrimitiveTree.from_string('gt(P1, MA5)',p)
    out=predict_tree(tree,p,np.array([[2.,1.],[0.,1.]]));assert out.tolist()==[True,False]
    with pytest.raises(TypeError):gp.PrimitiveTree.from_string('gt(True, MA5)',p)
    raw=np.array([[1,2,.55],[2,1.6,.53]])
    a=scores(raw,[.3,.5,.6],'RMO');b=scores(raw,[.6,.3,.4],'RMO')
    assert not np.array_equal(a,b)
    np.testing.assert_allclose(a,np.stack([raw[:,0],(raw[:,1]-1)/.5,(raw[:,2]-.5)/.015],1)@np.array([.3,.5,.6]))
    assert np.isfinite(raw_fitness(np.zeros(5),np.ones(5),np.ones(5),.001)).all()


def make_run(c):
    r=dict(id=store.uid('test_'),config=c,kind='training',status='RUNNING',created_at=store.now())
    return store.put('experiment',r)


def test_neural_checkpoint_restart_is_exact(datasets):
    c=config(datasets);frame,meta=data.load(c['dataset_id']);baseline=make_run(c)
    expected=fit(baseline,frame,meta,lambda:None,lambda *args:None)
    interrupted=make_run(c);calls=0
    def stop():
        nonlocal calls
        calls+=1
        if calls==3:raise RuntimeError('模拟进程中断')
    with pytest.raises(RuntimeError):fit(interrupted,frame,meta,stop,lambda *args:None)
    resumed=fit(interrupted,frame,meta,lambda:None,lambda *args:None)
    assert expected['results']==resumed['results']
    for obj in ['direction','duration']:
        first=torch.load(expected['artifacts'][obj]['path'],weights_only=False)['model']
        second=torch.load(resumed['artifacts'][obj]['path'],weights_only=False)['model']
        for key in first:assert torch.equal(first[key],second[key])


def test_gp_window_restart_is_exact(datasets):
    c=config(datasets,'stgp');frame,meta=data.load(c['dataset_id']);frame=frame.iloc[:360].copy();c['train_ratio']=.6
    first=make_run(c);expected=gp_fit(first,frame,meta,lambda:None,lambda *args:None)
    second=make_run(c);calls=0
    def stop():
        nonlocal calls
        calls+=1
        if calls==3:raise RuntimeError('中断')
    with pytest.raises(RuntimeError):gp_fit(second,frame,meta,stop,lambda *args:None)
    resumed=gp_fit(second,frame,meta,lambda:None,lambda *args:None)
    assert expected['windows']==resumed['windows']
    assert expected['predictions']==resumed['predictions']
