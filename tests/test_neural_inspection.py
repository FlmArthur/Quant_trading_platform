import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from backend import store
from backend.api import app
from backend.schemas import ExperimentConfig
from research.neural import TrendNetwork,snapshot,atomic_torch


@pytest.mark.parametrize('kernel',[1,3])
def test_legacy_checkpoint_enrichment_and_stage_calculation_consistency(kernel):
    c=ExperimentConfig(dataset_id='inspection-only',hidden=8,channels=4,kernel=kernel).model_dump()
    torch.manual_seed(24);model=TrendNetwork(c).eval()
    sample=dict(cnn=np.arange(20,dtype=np.float32).reshape(4,5)-4,raw=np.arange(20).reshape(4,5),
        trend=np.array([[.2,.3],[-.4,.4],[.6,.5],[-.8,.6]],np.float32),index=50,as_of='2024-01-02T14:00:00',direction=1,segments=[])
    view=snapshot(model,sample,'direction','direction-1')
    id=store.uid('run_');folder=store.DATA/'runs'/id;folder.mkdir(parents=True)
    for key in ['conv2_kernel','conv2_bias','output_weight','output_bias','head_input']:view.pop(key)
    path=folder/'direction-1.json';store.write_json(path,view);original_hash=store.checksum(path)
    atomic_torch(folder/'direction-1.pt',dict(model=model.state_dict(),config=c,objective='direction'))
    store.put('experiment',dict(id=id,kind='training',status='COMPLETED',config=c,latest_checkpoint='direction-1'))
    response=TestClient(app).get(f'/api/v1/experiments/{id}/snapshots')
    assert response.status_code==200,response.text
    s=response.json()['snapshot']
    assert store.checksum(path)==original_hash
    assert s['sample_id']==50 and s['checkpoint_id']=='direction-1'
    weights=np.array(s['conv2_kernel']);inputs=np.pad(np.array(s['conv1']),((0,0),(kernel//2,kernel//2)))
    for channel in range(4):
        for position in range(4):
            actual=max(0,float((weights[channel]*inputs[:,position:position+kernel]).sum()+s['conv2_bias'][channel]))
            assert actual==pytest.approx(s['conv2'][channel][position],abs=2e-5)
    for t in range(4):
        i,f,g,o=np.array(s['gates'][t]);previous=np.array(s['cell'][t-1]) if t else np.zeros(8)
        cell=f*previous+i*g
        np.testing.assert_allclose(cell,s['cell'][t],atol=1e-7)
        np.testing.assert_allclose(o*np.tanh(cell),s['hidden'][t],atol=1e-7)
    np.testing.assert_allclose(np.array(s['cnn_projection'])+s['lstm_projection'],s['fusion'],atol=1e-6)
    output=np.dot(s['output_weight'],np.tanh(s['fusion']))+s['output_bias']
    assert output==pytest.approx(s['output'],abs=1e-6)
    assert 1/(1+np.exp(-output))==pytest.approx(s['prediction'],abs=1e-6)
