"""Independent finite cases: preserve a failed case and continue the matrix."""
from pathlib import Path
import fcntl, json, os, subprocess, sys, time, traceback

R = Path('/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1')
sys.path.insert(0, str(R/'stage29_llama_Astar_v46_normalKV_20261106_v1/common/programs'))
import study_runner as L


def still_alive(reg):
    try:
        tail = Path(f"/proc/{reg['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return int(tail[19]) == reg['proc_start_ticks'] and tail[0] != 'Z'
    except FileNotFoundError:
        return False


def independent_cases(specs, execute, record):
    """A failed controller or final gate must not suppress the next case."""
    rows = []
    for spec in specs:
        try:
            result = execute(spec)
            assert result['gate'] == 'PASS'
        except Exception as error:
            result = {'gate': 'FAIL', 'error': str(error), 'traceback': traceback.format_exc()}
        row = {'case': spec, **result}
        rows.append(row)
        record(spec, row, rows)
    return rows


def main():
    S = Path(sys.argv[1]).resolve(); gpu = int(sys.argv[2])
    assert gpu == 0 and S.parent == R
    matrix = L.read(S/'MATRIX.json'); q = L.read(S/'CPU_QUALIFICATION.json')
    assert matrix['GPU'] == gpu and q['gate'] == 'PASS'
    assert L.sha(S/'runner.py') == q['runner_sha256']
    assert L.sha(S/'MATRIX.json') == q['matrix_sha256']
    specs = matrix['cases']; state = {'state': 'RUNNING', 'phase': 'LOCK_GPU',
        'active_case': None, 'completed': [], 'failed': [], 'pending': [s['name'] for s in specs],
        'failure_policy': 'preserve each failed attempt, no automatic retry, continue independent cases'}
    def status(**fields):
        state.update(fields, updated_utc=L.utc()); L.dump(S/'STUDY_STATUS.json', state)
    def execute(spec):
        C = Path(spec['stage']); A = C/'attempts'/spec['attempt']; p = L.read(C/'PROTOCOL.json')
        assert not list((C/'attempts').iterdir()), 'Existing attempt: never repeat'
        assert L.sha(C/'receipts/CPU_QUALIFICATION.json') == spec['qualification_sha256']
        until = time.monotonic()+21600
        while True:
            apps = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
            free = int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free >= p['minimum_free_memory_MiB_before_load'] and (p['external_coexistence_allowed'] or not apps):
                break
            status(phase='WAIT_RESOURCES_NO_EXTERNAL_SIGNALS',active_case=spec['name'],external_processes=apps,memory_free_MiB=free)
            if time.monotonic() >= until:
                raise TimeoutError('Resource wait exceeded six hours; case preserved, no external signals')
            time.sleep(30)
        status(phase='CONTROLLER',active_case=spec['name'],external_processes=apps)
        L.dump(C/'CASE_STATUS.json',{'state':'RUNNING','utc':L.utc(),'case':spec})
        with (C/'CONTROLLER.log').open('x') as log:
            subprocess.run([L.PYTHON,str(C/'programs/controller.py'),str(R),str(C),spec['arm'],spec['attempt'],'full'],
                cwd=R,env=L.env(),stdout=log,stderr=subprocess.STDOUT,check=True)
        status(phase='FINAL_GATE')
        with (C/'GATE.log').open('x') as log:
            subprocess.run([L.PYTHON,str(C/'programs/offline_compare.py'),str(R),str(C),spec['attempt']],
                cwd=R,env=L.env(),stdout=log,stderr=subprocess.STDOUT,check=True)
        gate=L.read(C/'receipts'/('RUN_GATE_'+spec['attempt']+'.json'))
        assert gate['gate']=='PASS' and all(gate['checks'].values())
        common=L.read(A/'common_audit_summary.json')
        assert common['resources']['gpu_blocks']==spec['gpu_blocks']
        signature=L.sha(A/'common_audit_summary.json')
        return {'gate':'PASS','metrics':gate['metrics'],'actual_gpu_blocks':spec['gpu_blocks'],
            'request_signatures_sha256':__import__('hashlib').sha256(json.dumps(gate['request_signatures'],sort_keys=True).encode()).hexdigest(),
            'common_audit_sha256':signature,'external_at_start':apps}
    def record(spec,row,rows):
        C=Path(spec['stage']);A=C/'attempts'/spec['attempt']
        if row['gate']=='PASS':
            L.dump(C/'CASE_STATUS.json',{'state':'COMPLETED','utc':L.utc(),'metrics':row['metrics'],'actual_gpu_blocks':spec['gpu_blocks']})
        else:
            L.dump(C/'CASE_STATUS.json',{'state':'FAILED_PRESERVED','phase':'CONTINUE_NEXT_NO_RETRY','utc':L.utc(),'error':row['error'],'traceback':row.get('traceback')})
            L.dump(C/'CASE_FAILURE.json',row)
        status(completed=[r['case']['name'] for r in rows if r['gate']=='PASS'],
            failed=[r['case']['name'] for r in rows if r['gate']!='PASS'],
            pending=[s['name'] for s in specs[len(rows):]],results=rows)
        # Continuing is safe only after this attempt's registered processes exit.
        registry=L.read(A/'processes.json') if (A/'processes.json').exists() else []
        live=[r for r in registry if still_alive(r)]
        if live:
            raise RuntimeError('Owned processes remain after controller cleanup; refuse overlap: '+repr(live))
    try:
        with open(f'/tmp/xuetianyou_chain_slo_gpu{gpu}.lock','a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            lock.seek(0);lock.truncate();lock.write(json.dumps({'pid':os.getpid(),'GPU':gpu,'study':str(S)}));lock.flush()
            status()
            rows=independent_cases(specs,execute,record)
            failed=[r['case']['name'] for r in rows if r['gate']!='PASS']
            L.dump(S/'FINAL_RESULTS.json',{'gate':'PASS' if not failed else 'COMPLETED_WITH_FAILURES','utc':L.utc(),'cases':rows,'failed':failed,'no_automatic_retries':True})
            status(state='COMPLETED' if not failed else 'COMPLETED_WITH_FAILURES',phase='DONE',active_case=None,pending=[])
    except BaseException as error:
        status(state='NEEDS_ATTENTION',phase='INVARIANT_OR_CLEANUP_FAILURE',error=str(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
