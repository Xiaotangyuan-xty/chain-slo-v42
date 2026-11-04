import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    root,stage=map(Path,sys.argv[1:3]);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    os.environ.pop('STUDY_C_VERSION',None)
    parent=json.loads((stage/'PARENT_CPU_QUALIFICATION.json').read_text());tests={}
    commands={'frozen_v45_native_block_manager': [str(stage/'programs/controlled_native.py')],
              'frozen_admission_forecast_transport':[ '-m','unittest','controlled_stability.LeasePacking','controlled_stability.ForecastRefresh','controlled_stability.TransportIsolation'],
              'mixed_equivalence_transition_stream_scope':[str(stage/'programs/tests.py')]}
    commands['safe_statistics_numeric_final_tbt']=[str(stage/'programs/test_safe_statistics.py')]
    for name,args in commands.items():
        r=subprocess.run([sys.executable]+args,cwd=stage/'programs',env=os.environ.copy(),capture_output=True,text=True)
        log=stage/'receipts'/('CPU_'+name+'.log');log.write_text(r.stdout+r.stderr)
        tests[name]={'pass':r.returncode==0,'log_sha256':sha(log)}
        print(name,r.returncode,flush=True)
        if r.returncode:print((r.stdout+r.stderr)[-14000:],flush=True)
    bases=dict(parent['base_files']);work=json.loads((stage/'WORKLOAD.json').read_text());bases.update(work['source_files'])
    bad=[n for n,h in bases.items()if sha(root/n)!=h]
    tests['parent_and_source_inputs_preserved']={'pass':not bad,'bad':bad}
    parent_path=Path(json.loads((stage/'BUILD_RECEIPT.json').read_text())['parent'])
    normalized=(stage/'programs/a_server.py').read_text().replace('    from safe_statistics import install as install_safe_statistics\n    install_safe_statistics(root,stage,output_dir)\n','')
    tests['native_A_launcher_has_only_declared_common_observer']={'pass':hashlib.sha256(normalized.encode()).hexdigest()==sha(parent_path/'programs/a_server.py')}
    stage_files={str(p.relative_to(stage)):sha(p)for p in sorted((stage/'programs').glob('*.py'))}
    for n in ['PROTOCOL.json','DEVICE_AUTHORIZATION.json','WORKLOAD.json','TRACE_MANIFEST.json','FROZEN_V45_CORE.json','PARENT_V45.json','PARENT_CPU_QUALIFICATION.json','BUILD_RECEIPT.json','assets/CHAIN_MODEL.json']:
        stage_files[n]=sha(stage/n)
    result={'schema':'stage20-cpu-qualification/v1','gate':'PASS'if all(x['pass']for x in tests.values())else'FAIL','tests':tests,'gpu_used':False,
            'base_files':bases,'stage_files':stage_files,'created_utc':datetime.now(timezone.utc).isoformat(),'calibration_only':False,
            'scope':'Native fallback equivalent to native on mixed/foreground CPU fixtures; compound-only equivalent to frozen v45; transitions and real block-manager swap/recompute qualified. Long execution requires subsequent user GPU assignment. This is the declared conservative coexistence policy, not an optimized concurrent mixed-v45 policy.'}
    (stage/'receipts/CPU_QUALIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'gate':result['gate'],'tests':tests,'base_files':len(bases),'stage_files':len(stage_files)}),flush=True)
    return 0 if result['gate']=='PASS'else 1

if __name__=='__main__':raise SystemExit(main())
