"""Three authorized Llama8B native-ToT mixed replays per GPU; no retry."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time,traceback
R=Path('/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1');S=Path(sys.argv[1]).resolve();GPU=int(sys.argv[2]);assert GPU==0
assert not(S.parent/'CPU_PREVIEW_ONLY.json').exists(),'CPU preview must never launch GPU'
sys.path.insert(0,str(R/'stage29_llama_Astar_v46_normalKV_20261106_v1/common/programs'));import study_runner as L

def main():
 q=L.read(S/'CPU_QUALIFICATION.json');assert q['gate']=='PASS'and L.sha(S/'runner.py')==q['runner_sha256']and L.sha(S/'MATRIX.json')==q['matrix_sha256']
 matrix=L.read(S/'MATRIX.json');specs=matrix['cases'];expected=matrix['methods']
 assert len(specs)==len(expected)and [c['version']for c in specs]==expected and all(c['GPU']==GPU for c in specs)
 state={'state':'RUNNING','phase':'LOCK_ASSIGNED_GPU','utc':L.utc(),'active_case':None,'completed':[],'pending':[c['name']for c in specs]};results=[];active=None;signatures={}
 def status(**kw):state.update(kw,updated_utc=L.utc());L.dump(S/'STUDY_STATUS.json',state)
 try:
  with open(f'/tmp/xuetianyou_chain_slo_gpu{GPU}.lock','a+')as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);lock.truncate();lock.write(json.dumps({'pid':os.getpid(),'GPU':GPU,'study':str(S),'scope':'three_serial_nativeToT_mixed'}));lock.flush()
   for i,spec in enumerate(specs):
    active=spec;C=Path(spec['stage']);a=C/'attempts'/spec['attempt'];p=L.read(C/'PROTOCOL.json')
    assert L.sha(C/'receipts/CPU_QUALIFICATION.json')==spec['qualification_sha256']
    assert not list((C/'attempts').iterdir()),'No repeat execution allowed'
    # Do not kill foreign jobs; stop if a new conflict appears between cases.
    apps=subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip();assert not apps,('Assigned GPU conflict; stop without signals',apps)
    status(phase='CONTROLLER_START',active_case=spec['name'],case_index=i+1,pending=[c['name']for c in specs[i:]])
    L.dump(C/'CASE_STATUS.json',{'state':'RUNNING','utc':L.utc(),'case':spec})
    with(C/'CONTROLLER.log').open('x')as log:subprocess.run([L.PYTHON,str(C/'programs/controller.py'),str(R),str(C),spec['arm'],spec['attempt'],'full'],cwd=R,env=L.env(),stdout=log,stderr=subprocess.STDOUT,check=True)
    status(phase='FINAL_OFFLINE_GATE')
    with(C/'GATE.log').open('x')as log:subprocess.run([L.PYTHON,str(C/'programs/offline_compare.py'),str(R),str(C),spec['attempt']],cwd=R,env=L.env(),stdout=log,stderr=subprocess.STDOUT,check=True)
    g=L.read(C/'receipts'/('RUN_GATE_'+spec['attempt']+'.json'));assert g['gate']=='PASS'and all(g['checks'].values())
    model=spec['model_key']
    if model in signatures:assert signatures[model]==g['request_signatures'],'Same-model request signatures changed'
    else:signatures[model]=g['request_signatures']
    adapter=a/'BASELINE_BUDGET_ADAPTER.json'
    if spec['version']=='v47pro':
     runtime=L.read(a/'V47PRO_RUNTIME.json');assert runtime['version']==matrix['runtime_version']and not runtime['stage_budget_adapter']and not (a/'V47PRO_BUDGET_ADAPTER.json').exists()and not adapter.exists()and runtime['prediction_comparison_ready']
    elif spec['version']=='A':assert not adapter.exists()
    else:assert L.read(adapter)['variant']=='cumulative_through_current_declared_depth_v1'
    result={'case':spec,'gate':'PASS','metrics':g['metrics']};results.append(result);L.dump(C/'CASE_STATUS.json',{'state':'COMPLETED','utc':L.utc(),'metrics':g['metrics']})
    status(completed=[c['case']['name']for c in results],pending=[c['name']for c in specs[i+1:]],results=results)
    if(i+1)%3==0:
     group=results[-3:];assert len({x['case']['version']for x in group})==3
     L.dump(S/f"GROUP_{model}_ORDER{spec['order']}.json",{'gate':'PASS','model':model,'order':spec['order'],'sequence':[x['case']['version']for x in group],'same_request_signatures':True,'results':group})
   L.dump(S/'FINAL_RESULTS.json',{'gate':'PASS','cases':results,'development_only_not_full_hour_comparison':True,'utc':L.utc()})
   status(state='COMPLETED',phase='DONE',active_case=None,pending=[],completed_utc=L.utc())
 except BaseException as e:
  failure={'state':'FAILED_PRESERVED','phase':'STOPPED_NO_RETRY','utc':L.utc(),'error':str(e),'traceback':traceback.format_exc()}
  if active:L.dump(Path(active['stage'])/'CASE_STATUS.json',failure)
  status(**failure,failed_case=active['name']if active else None,subsequent_cases_not_run=[c['name']for c in specs[len(results)+(1 if active else 0):]])
  raise
if __name__=='__main__':main()
