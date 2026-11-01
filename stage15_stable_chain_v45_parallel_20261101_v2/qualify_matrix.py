"""CPU-only qualification of every predeclared round using the pinned runtime."""
import json,os,subprocess,sys
from pathlib import Path
root,matrix=map(Path,sys.argv[1:3])
f=json.loads((root/'FREEZE.json').read_text());c=f['runtime'];library=json.loads(Path(c['library_contract']).read_text())
env=os.environ.copy()
for k in c['reuse_only']:env[k]=library[k]
env.update(c['new_environment_variables']);env['CUDA_VISIBLE_DEVICES']=''
paths=[root/'stage7_dynamic/programs',root/'stage7_dynamic/backend_src',root/'stage7_dynamic/v2_src',root/'stage3_gpu0/programs']
paths += [Path(p) if os.path.isabs(p) else root/p for p in c['pythonpath_entries']]
env['PYTHONPATH']=os.pathsep.join(map(str,paths))
for row in json.loads((matrix/'MATRIX.json').read_text())['rounds']:
    stage=matrix/row['directory'];(stage/'receipts').mkdir(exist_ok=False)
    with (stage/'CPU_QUALIFICATION_LAUNCH.log').open('x') as log:
        code=subprocess.run([c['python'],str(stage/'programs/qualify_cpu.py'),str(root),str(stage)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    q=json.loads((stage/'receipts/CPU_QUALIFICATION.json').read_text()) if (stage/'receipts/CPU_QUALIFICATION.json').exists() else {}
    print(json.dumps({'round':row['round'],'returncode':code,'gate':q.get('gate'),'tests':q.get('tests')}),flush=True)
    if code:raise SystemExit(code)
