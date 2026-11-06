"""Read-only verification of v47, active CPU evidence and unchanged v46."""
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import sys

R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read(R/'FREEZE_V47.json');stage=R/m['stage'];bad=[]
for rel,digest in m['files'].items():
    assert not Path(rel).is_absolute() and '..'not in Path(rel).parts
    p=R/rel
    if not p.is_file()or sha(p)!=digest:bad.append(rel)
q=read(R/m['CPU_qualification'])
if q['gate']!='PASS' or q['GPU_used'] or q['model_execution']:bad.append('CPU contract')
for rel,digest in q['source_files'].items():
    if sha(stage/rel)!=digest:bad.append('CPU source:'+rel)
for name,test in q['tests'].items():
    if not test['pass'] or sha(R/'evidence/v47/cpu_attempts'/m['active_attempt']/(name+'.log'))!=test['log_sha256']:
        bad.append('CPU test:'+name)
origin=read(stage/'V46_ORIGIN.json')
for rel,digest in origin['parent_files'].items():
    if sha(R/origin['parent_stage']/rel)!=digest:bad.append('changed parent:'+rel)
if sha(stage/'assets/CHAIN_MODEL.json')!=origin['parent_files']['assets/CHAIN_MODEL.json']:
    bad.append('FIT model changed')
asset=read(stage/'assets/CHAIN_MODEL.json')
if len(asset['records'])!=295 or set(asset['fit_ids'])&set(asset['cal_ids']):bad.append('FIT split')
for name in ['native_chain.py','chain_forecast.py','chain_admission.py','token_reward.py']:
    tree=ast.parse((stage/'programs'/name).read_text())
    if {n.attr for n in ast.walk(tree)if isinstance(n,ast.Attribute)}&{'target_output_length','real_output_len'}:
        bad.append('future truth:'+name)
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_VISIBLE_DEVICES']=''
old=subprocess.run([sys.executable,str(R/'verify_v46_freeze.py')],env=env,capture_output=True,text=True)
if old.returncode:bad.append('v46 freeze verifier')
print(json.dumps({'gate':'PASS'if not bad else'FAIL','version':m['version'],
                  'files_checked':len(m['files']),'CPU_tests':sum(v['tests']for v in q['tests'].values()),
                  'GPU_used':False,'model_execution':False,'v46_preserved':old.returncode==0,'bad':bad}))
raise SystemExit(bool(bad))
