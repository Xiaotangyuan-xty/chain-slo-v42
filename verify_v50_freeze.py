"""Read-only source/native-CPU evidence audit; never constructs an engine."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read(R/'FREEZE_V50.json');P=R/m['stage'];bad=[]
for rel,h in m['files'].items():
    if Path(rel).is_absolute()or'..'in Path(rel).parts:bad.append('unsafe path');continue
    p=R/rel
    if not p.is_file()or sha(p)!=h:bad.append('file:'+rel)
q=read(R/m['CPU_qualification'])
if q['gate']!='PASS'or q['GPU_used']or q['model_execution']or q['CUDA_VISIBLE_DEVICES']!='':bad.append('CPU contract')
for rel,h in q['source_files'].items():
    if not(P/rel).is_file()or sha(P/rel)!=h:bad.append('qualified source:'+rel)
for name,t in q['tests'].items():
    p=R/'evidence/v50/cpu_attempts'/m['active_attempt']/(name+'.log')
    if not t['pass']or sha(p)!=t['log_sha256']:bad.append('native test:'+name)
    elif '"CUDA_context_initialized": false'not in p.read_text():bad.append('CUDA proof:'+name)
if sum(t['tests']for t in q['tests'].values())!=m['CPU_tests']:bad.append('test count')
parent=R/'stage41_v49pro_stage_rollout_20261107_v1'
for rel,h in m['preserved_core_and_assets'].items():
    if sha(P/rel)!=h or sha(parent/rel)!=h:bad.append('preserved core:'+rel)
a=(P/'programs/shared_prediction.py').read_text().replace("method not in ['A','Astar','v48','v49','v49pro','v50']","method not in ['A','Astar','v48','v49','v49pro']")
if a!=(parent/'programs/shared_prediction.py').read_text():bad.append('shared operational semantics')
for name in ['native_chain.py','chain_forecast.py','chain_admission.py','token_reward.py',
             'mixed_joint.py','window_control.py','window_projection.py','candidate_budget.py','step_cost.py']:
    tree=ast.parse((P/'programs'/name).read_text())
    keys={n.attr for n in ast.walk(tree)if isinstance(n,ast.Attribute)}|{n.slice.value for n in ast.walk(tree)if isinstance(n,ast.Subscript)and isinstance(n.slice,ast.Constant)and isinstance(n.slice.value,str)}
    if keys&{'real_output_len','target_output_length','target_output_tokens'}:bad.append('future truth:'+name)
    if 'TRACE_PUBLIC_SNAPSHOTS' in (P/'programs'/name).read_text():bad.append('qualification fixture read by runtime:'+name)
preview=read(R/'evidence/v50/DEPLOYMENT_DRY_RUN.json')
if preview['gate']!='PASS'or preview['GPU_launches']or preview['formal_main_calls']or len(preview['cases'])!=6:bad.append('deployment preview')
replay=read(R/'evidence/v50/PUBLIC_TRACE_REPLAY.json')
if replay['GPU_goodput_prediction_claim']or replay['snapshots']['interventions']==0 or replay['source_membership_mismatches']:bad.append('public trace qualification')
for file in (P/'harness').glob('*.py'):
    ast.parse(file.read_text(),str(file))
    if 'stage.parents[2]'in file.read_text():bad.append('parent-depth root:'+file.name)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
prior=subprocess.run([sys.executable,str(R/'verify_v49pro_freeze.py')],env=env,capture_output=True,text=True)
if prior.returncode:bad.append('parent v49/v48/v47/v46')
print(json.dumps({'gate':'PASS'if not bad else'FAIL','version':m['version'],'CPU_tests':m['CPU_tests'],
                  'files_checked':len(m['files']),'GPU_used':False,'model_execution':False,
                  'pure_DR_core_preserved':not any(x.startswith('preserved core:')for x in bad),
                  'shared_prediction_operational_semantics_preserved':'shared operational semantics'not in bad,
                  'parent_versions_preserved':prior.returncode==0,'bad':bad}))
raise SystemExit(bool(bad))
