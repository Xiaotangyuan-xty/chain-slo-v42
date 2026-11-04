"""One exclusive GPU case, original A, semantic A*, and v46; stop on failure."""
from datetime import datetime,timezone
from pathlib import Path
import fcntl,hashlib,json,os,subprocess,sys,traceback

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def study_root(stage):return stage.parent.parent if stage.parent.name=='cases'else stage.parent
def main(root,stage):
 p=read(stage/'PROTOCOL.json');gpu=p['arm_devices']['A']['physical_gpu_index'];os.sched_setaffinity(0,set(p['cpu_affinity']['A']))
 q=read(stage/'receipts/CPU_QUALIFICATION.json');assert q['gate']=='PASS'
 for base,files in [(root,q['base_files']),(stage,q['stage_files'])]:assert all(sha(base/n)==h for n,h in files.items())
 lock=open(f'/tmp/xuetianyou_chain_slo_gpu{gpu}.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 lock.seek(0);lock.truncate();lock.write(json.dumps({'pid':os.getpid(),'stage':str(stage),'study':'v46'}));lock.flush()
 prep=read(root/'PREPARE.json');state={'state':'RUNNING','utc':datetime.now(timezone.utc).isoformat(),'GPU':gpu,'completed_arms':[],'active_arm':None,'phase':'STARTING'}
 def save(**v):state.update(v,updated_utc=datetime.now(timezone.utc).isoformat());dump(stage/'CALIBRATION_STATUS.json',state)
 save()
 try:
  for label in p['planned_arms']:
   name=p['attempts'][label];arm='C'if label=='C'else'A';save(active_arm=label,phase='CONTROLLER')
   with(stage/f'{label}_CONTROLLER.log').open('x')as log:
    subprocess.run([prep['python'],str(stage/'programs/controller.py'),str(root),str(stage),arm,name,'full'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
   save(phase='VALIDATING')
   with(stage/f'{label}_GATE.log').open('x')as log:
    subprocess.run([prep['python'],str(stage/'programs/offline_compare.py'),str(root),str(stage),name],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
   state['completed_arms'].append(label);save(completed_arms=state['completed_arms'])
  gates={a:read(stage/'receipts'/('RUN_GATE_'+p['attempts'][a]+'.json'))for a in p['planned_arms']}
  assert all(g['gate']=='PASS'for g in gates.values());signatures=[g['request_signatures']for g in gates.values()];assert all(s==signatures[0]for s in signatures)
  study=study_root(stage)
  subprocess.run([sys.executable,str(study/'common/programs/window_analysis.py'),str(stage),str(stage/'WINDOW_ANALYSIS.json')],check=True)
  dump(stage/'COMPARISON.json',{'gate':'PASS','GPU':gpu,'kind':p['workload_kind'],'metrics':{a:g['metrics']for a,g in gates.items()},'all_request_signatures_equal':True,
   'A_semantics':'Original native scheduling and original online stage allocator.','Astar_semantics':'Same native scheduling with explicitly corrected cumulative-through-current stage budget.',
   'C_semantics':'v46 joint admission plus the same stage-budget adapter as Astar.','arrival_gate_seconds':.5,'JCT_constraint':False})
  save(state='COMPLETED',phase='DONE',active_arm=None,completed_utc=datetime.now(timezone.utc).isoformat())
 except BaseException as e:save(state='FAILED_PRESERVED',phase='STOPPED',error=str(e),traceback=traceback.format_exc());raise
 finally:lock.close()
if __name__=='__main__':main(*map(Path,sys.argv[1:3]))
