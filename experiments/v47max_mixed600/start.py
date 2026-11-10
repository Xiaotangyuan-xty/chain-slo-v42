"""Seal verification and one unique detached launch; no per-case gate dependency."""
from pathlib import Path
import fcntl,json,os,subprocess,sys
R=Path('/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1')
sys.path.insert(0,str(R/'stage29_llama_Astar_v46_normalKV_20261106_v1/common/programs'))
import study_runner as L
S=Path(sys.argv[1]).resolve();gpu=int(sys.argv[2]);assert gpu==0 and S.parent==R
os.sched_setaffinity(0,{28,30,84,86})
with (S/'LAUNCH.lock').open('a+') as guard:
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not any((S/n).exists()for n in ['RUNNER_LAUNCH.json','LAUNCH_STARTED.json'])
    q=L.read(S/'CPU_QUALIFICATION.json');m=L.read(S/'MATRIX.json')
    assert q['gate']=='PASS' and m['GPU']==gpu and m['continue_after_case_failure'] is True
    assert L.sha(S/'runner.py')==q['runner_sha256'] and L.sha(S/'start.py')==q['start_sha256'] and L.sha(S/'MATRIX.json')==q['matrix_sha256']
    for spec in m['cases']:
        C=Path(spec['stage']);assert C.parent==S/'cases' and spec['GPU']==gpu
        assert not list((C/'attempts').iterdir())
        assert L.sha(C/'receipts/CPU_QUALIFICATION.json')==spec['qualification_sha256']
        cq=L.read(C/'receipts/CPU_QUALIFICATION.json');assert cq['gate']=='PASS'
        assert all(L.sha(C/n)==h for n,h in cq['stage_files'].items())
        assert all(L.sha(R/n)==h for n,h in cq['base_files'].items())
    with open(f'/tmp/xuetianyou_chain_slo_gpu{gpu}.lock','a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        actual=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        assert actual==L.UUIDS[gpu]
    cmd=['taskset','-c',','.join(map(str,q['CPU_affinity'])),L.PYTHON,str(S/'runner.py'),str(S),str(gpu)]
    L.dump(S/'LAUNCH_STARTED.json',{'utc':L.utc(),'command':cmd,'never_repeat_start':True})
    with(S/'RUNNER.log').open('x')as log:
        proc=subprocess.Popen(cmd,cwd=R,env=L.env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    tail=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()
    receipt={'utc':L.utc(),'pid':proc.pid,'pgid':os.getpgid(proc.pid),'proc_start_ticks':int(tail[19]),'command':cmd,'study':str(S),'GPU_allowed':[gpu],'formal_cases':len(m['cases']),'CPU_qualification_sha256':L.sha(S/'CPU_QUALIFICATION.json'),'matrix_sha256':L.sha(S/'MATRIX.json')}
    L.dump(S/'RUNNER_LAUNCH.json',receipt);print(json.dumps(receipt))
