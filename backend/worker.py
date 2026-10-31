import os
import sys
import time
import traceback
from . import store,data


class Cancelled(Exception):pass


def main(id):
    run=store.get(id);store.update(id,status='RUNNING',pid=os.getpid(),error=None)
    store.event(id,'task.state_changed',dict(status='RUNNING'))
    def emit(type,payload):return store.event(id,type,payload)
    def control():
        current=store.get(id)
        if current.get('command')=='cancel':raise Cancelled()
        if current.get('command')=='pause':
            store.update(id,status='PAUSED');emit('task.state_changed',dict(status='PAUSED'))
            while store.get(id).get('command')=='pause':time.sleep(.2)
            if store.get(id).get('command')=='cancel':raise Cancelled()
            store.update(id,status='RUNNING');emit('task.state_changed',dict(status='RUNNING'))
    try:
        if run['kind']=='acquisition':
            def progress(fraction,message):
                control();store.update(id,progress=fraction);emit('data.fetch_progress',dict(progress=fraction,message=message))
            result=data.acquire(run['config'],progress);control();store.update(id,result=result)
        else:
            frame,meta=data.load(run['config']['dataset_id'])
            from research.registry import METHODS
            method=METHODS[run['config']['method']]
            if method.trainer:
                result=method.trainer(run,frame,meta,control,emit)
            elif run['config']['method']=='stgp':
                from research.gp import fit
                market=data.load(run['config']['market_dataset_id'])[0] if run['config'].get('market_dataset_id') else None
                result=fit(run,frame,meta,control,emit,market)
            else:
                from research.neural import fit
                result=fit(run,frame,meta,control,emit)
            store.write_json(store.DATA/'runs'/id/'result.json',result)
            store.update(id,result=result)
        store.update(id,status='COMPLETED',progress=1,completed_at=store.now())
        emit('task.state_changed',dict(status='COMPLETED'))
    except Cancelled:
        store.update(id,status='CANCELLED');emit('task.state_changed',dict(status='CANCELLED'))
    except Exception as error:
        store.update(id,status='FAILED',error=str(error));emit('task.error',dict(message=str(error)))
        print(traceback.format_exc(),file=sys.stderr)


if __name__=='__main__':main(sys.argv[1])
