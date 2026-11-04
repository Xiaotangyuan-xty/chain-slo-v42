"""Launch only a qualified, unused v46 study; compatibility promotion is structural."""
from datetime import datetime,timezone
from pathlib import Path
import fcntl,hashlib,json,os,socket,subprocess,sys

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_new(p,d):
 with p.open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def main(root,study,mode,commit):
 assert mode in ['pilot','matrix'];case=study/'pilot_gpu2'if mode=='pilot'else study
 assert not(case/'LONG_RUNNER_LAUNCH.json').exists()
 if mode=='matrix':
  pilot=study/'pilot_gpu2';assert read(pilot/'CALIBRATION_STATUS.json')['state']=='COMPLETED'
  audit=read(pilot/'attempts/PILOT_V46_GPU2_01/V46_DECISION_AUDIT.json');assert audit['joint_selection_observed']and not audit['bad']
  stages=[Path(c['stage'])for c in read(study/'MATRIX.json')['cases']]
 else:stages=[case]
 launch_lock=open('/tmp/xuetianyou_v46_'+mode+'.lock','a+');fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 states={};bases={}
 for stage in stages:
  q=read(stage/'receipts/CPU_QUALIFICATION.json');assert q['gate']=='PASS';bases.update(q['base_files'])
  assert not list((stage/'attempts').iterdir())
  for n,h in q['stage_files'].items():assert sha(stage/n)==h,(stage,n)
  p=read(stage/'PROTOCOL.json');g=p['arm_devices']['A']['physical_gpu_index']
  if g in states:continue
  gpu=subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=index,uuid,memory.used','--format=csv,noheader'],text=True).strip()
  compute=subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
  assert gpu==f'{g}, '+p['arm_devices']['A']['gpu_uuid']+', 0 MiB'and not compute,(gpu,compute)
  with open(f'/tmp/xuetianyou_chain_slo_gpu{g}.lock','a+')as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for port in [p['http_ports']['A'],p['prediction_ports']['A']]:
   with socket.socket()as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))
  states[g]={'GPU':gpu,'compute':compute,'ports_free':True,'lock_free':True}
 assert all(sha(root/n)==h for n,h in bases.items())
 common=study/'common';qualification=read(common/'receipts/CPU_V46_INTEGRATION.json');assert qualification['gate']=='PASS'
 assert all(sha(common/n)==h for n,h in qualification['source_files'].items())
 now=datetime.now(timezone.utc).isoformat();write_new(case/'PRE_GPU_FREEZE.json',{'utc':now,'git_commit':commit,'mode':mode,'GPU_states':states,
  'base_paths_checked':len(bases),'common_CPU_qualification_sha256':sha(common/'receipts/CPU_V46_INTEGRATION.json'),
  'promotion_is_independent_of_chain_wins_and_JCT':True})
 command=[sys.executable,str(common/'programs'/('case_runner.py'if mode=='pilot'else'dispatch_matrix.py')),str(root),str(case)]
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 with(case/'LONG_RUNNER.stdout.log').open('x')as log:
  p=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 tail=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()
 record={'pid':p.pid,'pgid':os.getpgid(p.pid),'proc_start_ticks':int(tail[19]),'utc':now,'git_commit':commit,'command':command,'GPUs':list(states),'mode':mode}
 write_new(case/'LONG_RUNNER_LAUNCH.json',record);print(json.dumps(record),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],sys.argv[4])
