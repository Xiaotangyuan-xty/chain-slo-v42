import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
s=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'SNAPSHOT_V44_MANIFEST.json').read_text())
bad=[n for n,v in m['files'].items() if not (r/n).is_file() or s(r/n)!=v['sha256']]
f=json.loads((r/'FREEZE_V44.json').read_text());stage=r/f['stage']
q=json.loads((stage/'CPU_QUALIFICATION.json').read_text());assert q['gate']=='PASS'
for n,h in q['stage_files'].items():
    if not (stage/n).is_file() or s(stage/n)!=h:bad.append(str(stage/n))
for n,h in q['base_files'].items():
    if n in f['external_native_binary_sha256']:
        assert n.endswith('.so') and f['external_native_binary_sha256'][n]==h
    elif not (r/n).is_file() or s(r/n)!=h:bad.append(n)
assert not bad,bad
print(json.dumps({'gate':'PASS','files':len(m['files']),'stage_files':len(q['stage_files']),'base_present':f['qualified_base_files_present'],'external_binaries':len(f['external_native_binary_sha256']),'gpu_used':False}))
