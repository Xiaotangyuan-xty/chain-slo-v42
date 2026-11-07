"""Read-only v49 source/CPU evidence check plus preserved v48/v47/v46 checks."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read(R/'FREEZE_V49.json');stage=R/m['stage'];bad=[]
for rel,h in m['files'].items():
    if Path(rel).is_absolute()or'..'in Path(rel).parts:bad.append('unsafe path');continue
    p=R/rel
    if not p.is_file()or sha(p)!=h:bad.append(rel)
q=read(R/m['CPU_qualification'])
if q['gate']!='PASS'or q['GPU_used']or q['model_execution']or q['CUDA_VISIBLE_DEVICES']!='':bad.append('CPU-only qualification')
for rel,h in q['source_files'].items():
    if not(stage/rel).is_file()or sha(stage/rel)!=h:bad.append('CPU source:'+rel)
for name,test in q['tests'].items():
    p=R/'evidence/v49/cpu_attempts'/m['active_attempt']/(name+'.log')
    if not test['pass']or sha(p)!=test['log_sha256']:bad.append('CPU test:'+name)
    elif '"CUDA_context_initialized": false'not in p.read_text():bad.append('missing CPU context proof:'+name)
origin=read(stage/'V48_ORIGIN.json')
for rel,h in origin['unchanged_pure_files'].items():
    if sha(stage/rel)!=h or sha(R/origin['parent_stage']/rel)!=h:bad.append('pure core:'+rel)
for name in ['native_chain.py','chain_forecast.py','chain_admission.py','token_reward.py',
             'mixed_joint.py','mixed_control.py','credit_curve.py','step_cost.py','mixed_forecast.py']:
    tree=ast.parse((stage/'programs'/name).read_text())
    keys={n.attr for n in ast.walk(tree)if isinstance(n,ast.Attribute)}|{
        n.slice.value for n in ast.walk(tree)if isinstance(n,ast.Subscript)
        and isinstance(n.slice,ast.Constant)and isinstance(n.slice.value,str)}
    if keys&{'real_output_len','target_output_length','target_output_tokens'}:bad.append('future truth:'+name)
parent_shared=R/origin['parent_stage']/'programs/shared_prediction.py'
new_shared=stage/'programs/shared_prediction.py'
if sha(parent_shared)!=origin['frozen_shared_prediction_parent_sha256']:bad.append('shared parent')
if new_shared.read_text().replace("method not in ['A','Astar','v48','v49']","method not in ['A','Astar','v48']")!=parent_shared.read_text():bad.append('shared operational semantics changed')
bank=read(stage/'assets/TOT_CALIBRATION.json')
if bank['GPU_labels_used']or not all(x['evaluation_graphs_disjoint']and len(x['model']['records'])==14 for x in bank['models'].values()):bad.append('ToT reserved calibration')
if sha(stage/'assets/TOT_CALIBRATION.json')!=sha(R/origin['parent_stage']/'assets/TOT_CALIBRATION.json'):bad.append('ToT calibration changed')
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
parent=subprocess.run([sys.executable,str(R/'verify_v48_freeze.py')],env=env,capture_output=True,text=True)
if parent.returncode:bad.append('parent v48/v47/v46')
if sum(x['tests']for x in q['tests'].values())!=m['CPU_tests']:bad.append('CPU test count')
print(json.dumps({'gate':'PASS'if not bad else'FAIL','version':m['version'],'CPU_tests':m['CPU_tests'],
                  'files_checked':len(m['files']),'GPU_used':False,'model_execution':False,
                  'pure_v47_bytes_preserved':not any(x.startswith('pure core:')for x in bad),
                  'shared_prediction_semantics_preserved':'shared operational semantics changed'not in bad,
                  'parent_v48_v47_v46_preserved':parent.returncode==0,'bad':bad}))
raise SystemExit(bool(bad))
