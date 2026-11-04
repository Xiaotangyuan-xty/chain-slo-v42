"""One explicitly authorized long A/C block on the assigned GPU."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

def dump(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');os.replace(q,p)
def main():
    root,stage=map(Path,sys.argv[1:3]);p=json.loads((stage/'PROTOCOL.json').read_text());python=json.loads((root/'PREPARE.json').read_text())['python']
    assert p['study_mode']=='long_execution_bound'and sorted(p['planned_arms'])==['A','C']and p['controller_wall_seconds']>=7200
    approval=json.loads((stage/'LONG_EXECUTION_APPROVAL.json').read_text())
    assert approval['authorized'] is True and approval['long_experiment_authorized'] is True
    assert json.loads((stage/'receipts/CPU_QUALIFICATION.json').read_text())['gate']=='PASS'
    lock=open('/tmp/xuetianyou_chain_slo_gpu'+str(p['arm_devices']['A']['physical_gpu_index'])+'.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(json.dumps({'pid':os.getpid(),'stage':str(stage),'mode':'long_explicitly_authorized'}));lock.flush()
    status={'state':'RUNNING','utc':datetime.now(timezone.utc).isoformat(),'GPU':p['arm_devices']['A']['physical_gpu_index'],'completed_arms':[],'active_arm':None}
    def state(**x):status.update(x);status['updated_utc']=datetime.now(timezone.utc).isoformat();dump(stage/'CALIBRATION_STATUS.json',status);print(json.dumps(x),flush=True)
    try:
        for arm in p['planned_arms']:
            state(active_arm=arm,phase='CALIBRATION_CONTROLLER')
            name=p['attempts'][arm]
            with(stage/f'{arm}_CONTROLLER.log').open('x')as log:
                r=subprocess.run([python,str(stage/'programs/controller.py'),str(root),str(stage),arm,name,'full'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError('Controller failed; no retries: '+arm)
            state(phase='VALIDATING_ARM')
            with(stage/f'{arm}_GATE.log').open('x')as log:
                r=subprocess.run([python,str(stage/'programs/offline_compare.py'),str(root),str(stage),name],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError('Calibration validity failed; preserved: '+arm)
            status['completed_arms'].append(arm);state(completed_arms=status['completed_arms'])
        gates={a:json.loads((stage/'receipts'/('RUN_GATE_'+p['attempts'][a]+'.json')).read_text())for a in ['A','C']}
        assert gates['A']['request_signatures']==gates['C']['request_signatures'],'A/C frozen request contract differs'
        state(state='COMPLETED',phase='DONE',active_arm=None,completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as e:state(state='FAILED_PRESERVED',phase='STOPPED',error=str(e),traceback=traceback.format_exc());raise
    finally:lock.close()

if __name__=='__main__':main()
