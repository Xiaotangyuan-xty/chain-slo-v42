"""Read-only per-study model configuration; native policy and model loaders untouched."""
from pathlib import Path
import hashlib,json
ALLOWED_MODELS={'qwen14':'/mnt/sda1/Qwen2.5-14B-instruct','llama8':'/mnt/sda1/Meta-Llama-3.1-8B-Instruct'}
def read(p):return json.loads(Path(p).read_text())
def profile(stage):
 stage=Path(stage);p=read(stage/'PROTOCOL.json');file=stage/'MODEL_PROFILE.json'
 assert hashlib.sha256(file.read_bytes()).hexdigest()==p['model_profile_sha256'],'Model profile changed'
 m=read(file);assert m['model_path']==ALLOWED_MODELS[m['model_key']],'Unexpected generation model'
 assert p['model_key']==m['model_key'] and p['model_path']==m['model_path']
 assert p['expected_resource_caps']==m['expected_resource_caps']
 return m

def prepare(root,stage):
 base=read(Path(root)/'PREPARE.json');m=profile(stage)
 base.update(model=m['model_path'],qrf_model=m['qrf_model'],qrf_vectorizer=m['qrf_vectorizer'])
 return base

def resources(base,stage):
 m=profile(stage);result=dict(base);result.update(m['resources']);return result

def verify_external(stage):
 m=profile(stage)
 p=read(Path(stage)/'PROTOCOL.json')
 if p.get('asset_receipt'):
  receipt_path=Path(p['asset_receipt']);assert hashlib.sha256(receipt_path.read_bytes()).hexdigest()==p['asset_receipt_sha256']
  receipt=read(receipt_path);assert receipt['gate']=='PASS' and receipt['model_path']==m['model_path']
  for name,spec in m['external_files'].items():
   f=Path(name);st=f.stat();r=receipt['files'][name]
   assert r['sha256']==spec['sha256'] and r['bytes']==spec['bytes']
   assert {k:getattr(st,k)for k in ['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns']}==r['stat'],('Asset changed after study SHA verification',name)
  return {'gate':'PASS','model_key':m['model_key'],'model_path':m['model_path'],'external_files':len(m['external_files']),'verification':'Full study SHA before GPU allocation, identity/size/mtime/ctime rechecked each attempt','receipt_sha256':p['asset_receipt_sha256']}
 for name,spec in m['external_files'].items():
  path=Path(name);assert path.stat().st_size==spec['bytes'],('Asset size',name)
  with path.open('rb')as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
  assert h==spec['sha256'],('Asset SHA',name)
 return {'gate':'PASS','model_key':m['model_key'],'model_path':m['model_path'],'external_files':len(m['external_files'])}

def resource_checks(protocol,actual):
 return {k:actual.get(k)==v for k,v in protocol['expected_resource_caps'].items()}
