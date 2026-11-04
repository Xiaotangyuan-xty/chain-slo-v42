"""Repeatable CPU qualification, preserving every previous source/test attempt."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,subprocess,sys

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def qualify(root,common):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    old=read(common/'receipts/CPU_V46_INTEGRATION.json');prep=read(root/'PREPARE.json');base=read(root/'FREEZE.json');libs=read(Path(base['runtime']['library_contract']));env=os.environ.copy()
    for k in base['runtime']['reuse_only']:env[k]=libs[k]
    env.update(base['runtime']['new_environment_variables']);paths=[common/'programs',root/'stage7_dynamic/programs',root/'stage7_dynamic/backend_src',root/'stage7_dynamic/v2_src',root/'stage3_gpu0/programs']+[Path(x)if os.path.isabs(x)else root/x for x in base['runtime']['pythonpath_entries']]
    env.update(PYTHONPATH=os.pathsep.join(map(str,paths)),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    parent=common/'cpu_attempts';directory=parent/f'attempt{len(list(parent.iterdir()))+1:02}';directory.mkdir();tests={}
    for label,filename in [('v46','test_v46.py'),('native_v45_reference','controlled_native.py'),('export_statistics','test_safe_statistics.py'),('lifetime_observer','test_lifetime_observation.py'),('fresh_transport_contract','test_transport_v46.py')]:
        r=subprocess.run([prep['python'],str(common/'programs'/filename)],cwd=common/'programs',env=env,capture_output=True,text=True,timeout=180)
        log=directory/(label+'.log');log.write_text(r.stdout+r.stderr);tests[label]={'pass':r.returncode==0,'log_sha256':sha(log),'log_path':str(log.relative_to(common))}
        print(label,r.returncode,flush=True)
        if r.returncode:print((r.stdout+r.stderr)[-18000:],flush=True)
    tests['original_native_base_unchanged']={'pass':all(sha(root/n)==h for n,h in old['base_files'].items())}
    q={'gate':'PASS'if all(x['pass']for x in tests.values())else'FAIL','tests':tests,'base_files':old['base_files'],'GPU_used':False,
      'source_files':{str(f.relative_to(common)):sha(f)for f in(common/'programs').glob('*.py')},'utc':datetime.now(timezone.utc).isoformat()}
    for p in [directory/'CPU_V46_INTEGRATION.json',common/'receipts/CPU_V46_INTEGRATION.json']:p.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
    return q
if __name__=='__main__':
    r=qualify(*map(Path,sys.argv[1:3]));print(json.dumps({'gate':r['gate']}));raise SystemExit(0 if r['gate']=='PASS'else 1)
