"""Read-only artifact/source audit; never constructs a model/engine/device."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys
R=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text());bad=[]
m=read(R/'FREEZE_V50PRO.json');P=R/m['stage'];q=read(R/m['CPU_qualification']);g=read(R/m['GPU_components']);parent=R/'stage43_v50_window_projection_20261108_v1'
for n,h in {**m['files'],**m['evidence_files']}.items():
 if not(R/n).is_file()or sha(R/n)!=h:bad.append('file:'+n)
for n,h in q['source_files'].items():
 if not(P/n).is_file()or sha(P/n)!=h:bad.append('CPU-source:'+n)
if q['gate']!='PASS'or q['GPU_used']or q['model_execution']or q['CUDA_VISIBLE_DEVICES']!=''or q['CPU_tests']!=119:bad.append('CPU qualification')
for n,h in m['preserved_core_assets'].items():
 if sha(P/n)!=h or sha(parent/n)!=h:bad.append('preserved:'+n)
s=(P/'programs/shared_prediction.py').read_text().replace("'v49pro','v50','v50pro']","'v49pro','v50']")
if s!=(parent/'programs/shared_prediction.py').read_text():bad.append('shared prediction numeric/operational semantics')
for n in ['native_chain.py','chain_forecast.py','chain_admission.py','token_reward.py','mixed_joint.py','window_control.py','window_projection.py','compute_admission.py','service_commitments.py','step_cost.py','stage_scenarios.py','credit_cues.py']:
 text=(P/'programs'/n).read_text();tree=ast.parse(text,n);keys={a.attr for a in ast.walk(tree)if isinstance(a,ast.Attribute)}|{a.slice.value for a in ast.walk(tree)if isinstance(a,ast.Subscript)and isinstance(a.slice,ast.Constant)and isinstance(a.slice.value,str)}
 if keys&{'real_output_len','target_output_length','target_output_tokens'}:bad.append('future truth:'+n)
 if any(k in text for k in ['TRACE_PUBLIC_SNAPSHOTS','WORKLOAD.json','MAIN_START.json']):bad.append('benchmark/runtime separation:'+n)
if g['completed_pass']!=8 or g['preserved_failed']!=1:bad.append('GPU component counts')
for item in g['records']:
 if item['component']=='component03_physical_dynamic':
  if item['state']['state']!='FAILED_PRESERVED'or not item['controller_cleanup']['gpu_released']:bad.append('failed component preservation')
 elif item['gate']!='PASS'or not all(item['original_gate_checks'].values()):bad.append('component gate:'+item['component'])
last=g['records'][-1]
if last.get('independent_audit',{}).get('gate')!='PASS':bad.append('latest independent audit')
if any(last['source_hashes'].get(n)!=h for n,h in q['source_files'].items()):bad.append('latest GPU source binding')
for p in(P/'harness').glob('*.py'):
 ast.parse(p.read_text(),str(p))
 if 'stage.parents[2]'in p.read_text():bad.append('implicit root:'+p.name)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run([sys.executable,str(R/'verify_v50_freeze.py')],env=env,capture_output=True,text=True)
if r.returncode:bad.append('parent version preservation')
print(json.dumps({'gate':'PASS'if not bad else'FAIL','version':'v50pro','CPU_tests':119,'completed_GPU_components':8,'preserved_failed_component':1,'source_files':len(m['files']),'GPU_used_by_this_verifier':False,'model_execution':False,'parent_versions_preserved':r.returncode==0,'bad':bad}))
raise SystemExit(bool(bad))
