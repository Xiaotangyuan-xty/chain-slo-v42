"""Read-only source/evidence audit; no native engine/model/GPU construction."""
from pathlib import Path
import hashlib,json,subprocess,sys

ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    f=read(ROOT/'FREEZE_V47PRO.json');bad=[]
    for name,expected in f['files'].items():
        path=ROOT/name
        if not path.is_file() or sha(path)!=expected:bad.append(name)
    stage=ROOT/f['stage'];evidence=ROOT/'evidence/v47pro'
    parent=subprocess.run([sys.executable,str(ROOT/'verify_v50pro_freeze.py')],text=True,capture_output=True)
    if parent.returncode:bad.append('prior_version_freezes')
    else:
        summary=json.loads(parent.stdout.strip().splitlines()[-1])
        if summary['gate']!='PASS':bad.append('prior_version_freezes')
    cpu=read(evidence/'CPU_QUALIFICATION.json')
    if not (cpu['gate']=='PASS' and cpu['tests']==48 and cpu['GPU_used'] is False):bad.append('CPU_qualification')
    for name,expected in cpu['source_files'].items():
        if sha(stage/name)!=expected:bad.append('CPU_source:'+name)
    gpu=read(evidence/'GPU_VALIDATION.json')
    if gpu['gate']!='PASS' or len(gpu['release_source_cases'])!=2:bad.append('GPU_release_cases')
    capacities=set()
    for case in gpu['release_source_cases']:
        audit=read(evidence/case['audit']);export=read(evidence/case['terminal_export'])
        capacities.add(case['KV_GiB'])
        if not (audit['gate']=='PASS' and all(audit['checks'].values()) and audit['horizon_seconds']==300 and audit['KV_GiB']==case['KV_GiB']):bad.append('GPU_audit:'+case['label'])
        if export['state']['state']!='COMPLETED' or export['cases'][0]['gate']['gate']!='PASS':bad.append('GPU_terminal:'+case['label'])
        for name,expected in audit['source_publication_hashes'].items():
            if sha(stage/name)!=expected:bad.append('GPU_source:'+name)
        if any(p['same_live_identity'] for p in audit['processes']):bad.append('GPU_processes:'+case['label'])
    if capacities!={1.,19.875}:bad.append('GPU_capacities')
    failed=read(evidence/gpu['preserved_failed_case']['terminal_export'])
    gate=failed['cases'][0]['gate']
    if failed['state']['state']!='FAILED_PRESERVED' or gate['gate']!='FAIL' or gate['checks'].get('no_outside_candidate_selection') is not False:bad.append('failed_GPU_snapshot_not_preserved')
    for case in gpu.get('superseded_validated_cases',[]):
        old_audit=read(evidence/case['audit'])
        if old_audit['gate']!='PASS' or not all(old_audit['checks'].values()):bad.append('superseded_validated_case')
    old=ROOT/'stage30_v47_token_goodput_20261106_v1'
    for name in ['assets/CHAIN_MODEL.json','programs/token_reward.py']:
        if (stage/name).read_bytes()!=(old/name).read_bytes():bad.append('v47_frozen_origin:'+name)
    common=ROOT/'stage46_v50pro_executable_commitments_20261108_v1/programs'
    for name in ['prediction_transport.py','native_tot_adapter.py','execution_observation.py','baseline_budget.py','protocol_paths.py']:
        if (stage/'programs'/name).read_bytes()!=(common/name).read_bytes():bad.append('shared_infrastructure:'+name)
    prior=(common/'shared_prediction.py').read_text().replace("'v50pro']","'v50pro','v47pro']")
    if (stage/'programs/shared_prediction.py').read_text()!=prior:bad.append('shared_prediction_label_only')
    final=read(evidence/'PRE_PUBLICATION_STATE.json')
    if not (final['GPU0_compute_empty'] and final['all_development_runners_exited'] and final['Stage47_state']=='STOPPED_BY_USER' and final['GPU1_untouched'] and final['GPU2_untouched']):bad.append('pre_publication_state')
    result={'gate':'PASS' if not bad else 'FAIL','version':'v47pro','CPU_tests':48,'completed_GPU_development_cases':2,'preserved_failed_GPU_case':1,'parent_versions_preserved':parent.returncode==0,'source_files':len(cpu['source_files']),'GPU_used_by_this_verifier':False,'model_execution':False,'bad':bad}
    print(json.dumps(result));return bool(bad)

if __name__=='__main__':raise SystemExit(main())
