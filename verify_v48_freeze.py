"""Read-only v48 source/CPU evidence and parent-v47 verification; no GPU."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read(R/'FREEZE_V48.json');stage=R/m['stage'];bad=[]
for rel,digest in m['files'].items():
 if Path(rel).is_absolute()or'..'in Path(rel).parts:bad.append('unsafe manifest path');continue
 p=R/rel
 if not p.is_file()or sha(p)!=digest:bad.append(rel)
q=read(R/m['CPU_qualification'])
if q['gate']!='PASS'or q['GPU_used']or q['model_execution']or q['CUDA_VISIBLE_DEVICES']!='':bad.append('CPU-only qualification')
for rel,digest in q['source_files'].items():
 if not(stage/rel).is_file()or sha(stage/rel)!=digest:bad.append('CPU source:'+rel)
for name,test in q['tests'].items():
 p=R/'evidence/v48/cpu_attempts'/m['active_attempt']/(name+'.log')
 if not test['pass']or sha(p)!=test['log_sha256']:bad.append('CPU test:'+name)
 elif '"CUDA_context_initialized": false'not in p.read_text():bad.append('missing CPU context check:'+name)
origin=read(stage/'V47_ORIGIN.json')
for rel,digest in origin['unchanged_pure_files'].items():
 if sha(stage/rel)!=digest or sha(R/origin['parent_stage']/rel)!=digest:bad.append('pure core:'+rel)
for name in ['native_chain.py','chain_admission.py','chain_forecast.py','token_reward.py','mixed_joint.py','mixed_control.py','mixed_forecast.py']:
 tree=ast.parse((stage/'programs'/name).read_text())
 attrs={n.attr for n in ast.walk(tree)if isinstance(n,ast.Attribute)}
 keys={n.slice.value for n in ast.walk(tree)if isinstance(n,ast.Subscript)and isinstance(n.slice,ast.Constant)and isinstance(n.slice.value,str)}
 if (attrs|keys)&{'real_output_len','target_output_length','target_output_tokens'}:bad.append('future truth:'+name)
bank=read(stage/'assets/TOT_CALIBRATION.json')
if bank['GPU_labels_used']or not all(v['evaluation_graphs_disjoint']and len(v['model']['records'])==14 and not v['performance_or_coverage_validated']for v in bank['models'].values()):bad.append('ToT calibration scope')
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
parent=subprocess.run([sys.executable,str(R/'verify_v47_freeze.py')],env=env,capture_output=True,text=True)
if parent.returncode:bad.append('parent v47/v46 freeze')
total=sum(t['tests']for t in q['tests'].values())
if total!=m['CPU_tests']:bad.append('test count')
print(json.dumps({'gate':'PASS'if not bad else'FAIL','version':m['version'],'files_checked':len(m['files']),'CPU_tests':total,'GPU_used':False,'model_execution':False,'pure_v47_bytes_preserved':not any(x.startswith('pure core:')for x in bad),'parent_v47_v46_preserved':parent.returncode==0,'bad':bad}))
raise SystemExit(bool(bad))
