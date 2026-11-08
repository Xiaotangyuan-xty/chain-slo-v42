"""Start the already qualified three-case serial queue exactly once."""
from pathlib import Path
import fcntl,json,os,subprocess,sys
R=Path('/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1');S=Path(sys.argv[1]).resolve();GPU=int(sys.argv[2]);assert GPU==0
assert not(S.parent/'CPU_PREVIEW_ONLY.json').exists(),'CPU preview must never launch GPU'
sys.path.insert(0,str(R/'stage29_llama_Astar_v46_normalKV_20261106_v1/common/programs'));import study_runner as L
os.sched_setaffinity(0,{28,30,84,86})
with(S/'LAUNCH.lock').open('a+')as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(S/'LAUNCH_STARTED.json').exists()and not(S/'RUNNER_LAUNCH.json').exists(),'Existing launch: inspect only'
 q=L.read(S/'CPU_QUALIFICATION.json');assert q['gate']=='PASS'and L.sha(S/'runner.py')==q['runner_sha256']and L.sha(S/'start.py')==q['start_sha256']and L.sha(S/'MATRIX.json')==q['matrix_sha256']
 matrix=L.read(S/'MATRIX.json');assert matrix['GPU']==GPU
 for c in matrix['cases']:
  C=Path(c['stage']);assert not list((C/'attempts').iterdir());assert L.sha(C/'receipts/CPU_QUALIFICATION.json')==c['qualification_sha256'];assert L.sha(C/'WORKLOAD.json')==c['workload_sha256']
 assert matrix['methods']==['v47pro'] and 0<matrix['horizon']<=600
 assert matrix['v47pro_commit']=='UNPUBLISHED_DEVELOPMENT_SNAPSHOT'
 assert all(c['gpu_blocks']==matrix['models']['llama8']['gpu_blocks'] for c in matrix['cases'])
 assert matrix['GPU']==GPU and [c['version']for c in matrix['cases']]==matrix['methods']
 with open(f'/tmp/xuetianyou_chain_slo_gpu{GPU}.lock','a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  gpu=subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip();apps=subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip();f=[s.strip()for s in gpu.split(',')];assert f[1]==L.UUIDS[GPU]and int(f[2].split()[0])<=16 and int(f[3].split()[0])==0 and not apps,(gpu,apps)
  L.dump(S/'PRE_LAUNCH.json',{'utc':L.utc(),'assigned_gpu':gpu,'assigned_compute':apps,'exclusive_lock_available':True})
 cmd=['taskset','-c',','.join(map(str,q['CPU_affinity'])),L.PYTHON,str(S/'runner.py'),str(S),str(GPU)]
 L.dump(S/'LAUNCH_STARTED.json',{'utc':L.utc(),'command':cmd,'do_not_retry':True})
 with(S/'RUNNER.log').open('x')as log:p=subprocess.Popen(cmd,cwd=R,env=L.env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 st=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split();launch={'utc':L.utc(),'pid':p.pid,'pgid':os.getpgid(p.pid),'proc_start_ticks':int(st[19]),'command':cmd,'GPU_allowed':[GPU],'formal_cases':len(matrix['cases']),'CPU_qualification_sha256':L.sha(S/'CPU_QUALIFICATION.json'),'matrix_sha256':L.sha(S/'MATRIX.json'),'study':str(S)};L.dump(S/'RUNNER_LAUNCH.json',launch);print(json.dumps(launch))
