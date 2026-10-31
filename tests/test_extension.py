from fastapi.testclient import TestClient
from backend import store
from backend.api import app
from backend.worker import main as worker
from research.registry import Method,register


def test_new_registered_method_uses_existing_account_without_engine_changes():
    def validate(frame,meta,c):return {'samples':len(frame)}
    def train(run,frame,meta,control,emit):
        path=store.DATA/'runs'/run['id']/'rule.json';store.write_json(path,{'threshold':0})
        return dict(results={'sample_count':len(frame)},artifacts={'rule':{'path':str(path)}},
                    training_cutoff=frame.timestamp.iloc[100],deviations=['测试用可信扩展'])
    class Predictor:
        def __init__(self,strategy,frame,meta,run):self.frame=frame;self.cutoff=run['result']['training_cutoff']
        def predict(self,index):
            t=self.frame.timestamp.iloc[index]
            return dict(direction=int(self.frame.close.iloc[index]>self.frame.close.iloc[index-1]),
                        horizon=1,as_of=t,available_at=t,training_cutoff=self.cutoff,reason='扩展规则')
    register(Method('extension_test','扩展测试方法','测试论文','第1页','验证统一接口',['准备','计算','发布'],
                    trainer=train,validator=validate,predictor_factory=Predictor))
    client=TestClient(app);d=next(d for d in store.listing('dataset') if d['frequency']=='5m' and d['rows']>2000)
    response=client.post('/api/v1/experiments',json=dict(method='extension_test',dataset_id=d['id']))
    assert response.status_code==200,response.text
    id=response.json()['id'];worker(id)
    assert store.get(id)['status']=='COMPLETED'
    strategy=client.post('/api/v1/strategies',json=dict(run_id=id,name='可信扩展')).json()
    session=client.post('/api/v1/sessions',json=dict(strategy_id=strategy['id'],dataset_id=d['id'])).json()
    step=client.post(f'/api/v1/sessions/{session["id"]}/step',json=dict(bars=3))
    assert step.status_code==200,step.text
    assert step.json()['state']['signals'] and step.json()['state']['orders']
    report=client.post('/api/v1/reports',json={'run_id':id}).json()
    response=client.get(f'/api/v1/reports/{report["id"]}/download')
    assert response.status_code==200 and 'sample_count' in response.text
