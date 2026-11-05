"""Verify current v46 scope, source/CPU evidence and preserved history. No CUDA."""
from pathlib import Path
import hashlib, importlib.util, json

R=Path(__file__).resolve().parent
manifest=json.loads((R/'FREEZE_V46.json').read_text())
bad=[]


def check(path, expected, label):
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        bad.append(label)


for name,expected in manifest['files'].items():
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    check(R/name,expected,name)
stage=R/manifest['stage']
spec=importlib.util.spec_from_file_location('v46_scope',stage/'tools/verify_scope.py')
scope=importlib.util.module_from_spec(spec);spec.loader.exec_module(scope)
if scope.verify()['gate']!='PASS':bad.append('two-change scope')
qpath=R/manifest['active_CPU_qualification']
q=json.loads(qpath.read_text())
if q['gate']!='PASS' or q['GPU_used'] or q['model_execution']:
    bad.append('CPU qualification contract')
for name,digest in q['files'].items():
    check(stage/name,digest,'qualified source:'+name)
for name,test in q['tests'].items():
    if not test['pass']:bad.append('CPU test:'+name)
    check(qpath.parent/(name+'.log'),test['log_sha256'],'CPU log:'+name)
history=R/'history/v46_superseded_20261105'
old=json.loads((history/'FREEZE_V46_original.json').read_text())
archived={'README.md':'ROOT_README_original.md',
          'verify_v46_freeze.py':'verify_v46_freeze_original.py'}
for name,digest in old['files'].items():
    path=history/archived[name] if name in archived else R/name
    check(path,digest,'preserved superseded file:'+name)
print(json.dumps({'gate':'PASS'if not bad else'FAIL','version':manifest['candidate_policy'],
                  'files_checked':len(manifest['files']),'preserved_old_files':len(old['files']),
                  'CPU_tests':manifest['CPU_tests'],'bad':bad,'GPU_used':False}))
raise SystemExit(1 if bad else 0)
