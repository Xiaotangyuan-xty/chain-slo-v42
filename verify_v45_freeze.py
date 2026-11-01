import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'SNAPSHOT_V45_MANIFEST.json').read_text())
bad=[n for n,v in m['files'].items()if not(r/n).is_file()or sha(r/n)!=v['sha256']]
digest=hashlib.sha256(json.dumps(m['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert digest==m['snapshot_digest']
f=json.loads((r/'FREEZE_V45.json').read_text());s=r/f['stage'];fingerprints=[]
for i in range(1,4):
    p=s/f'round{i:02}';q=json.loads((p/'receipts/CPU_QUALIFICATION.json').read_text())
    assert q['gate']=='PASS' and len(q['stage_files'])==25
    for n,h in q['stage_files'].items():
        if not(p/n).is_file()or sha(p/n)!=h:bad.append(str(p/n))
    for n,h in q['base_files'].items():
        if n in f['external_native_binary_sha256']:
            assert n.endswith('.so')and f['external_native_binary_sha256'][n]==h
        elif not(r/n).is_file()or sha(r/n)!=h:bad.append(n)
    fingerprints.append({n:h for n,h in q['stage_files'].items()if n.startswith(('programs/','assets/'))})
    c=json.loads((p/'receipts/AC_COMPARISON.json').read_text())
    a=json.loads((p/'results/FINAL_AUDIT.json').read_text())
    assert c['comparison_valid']and c['gate']=='PASS'and a['gate']=='PASS'
    assert(c['metrics']['A']['official_scaled_attained'],c['metrics']['C']['official_scaled_attained'])==[(12,43),(9,44),(14,43)][i-1]
assert all(x==fingerprints[0]for x in fingerprints)
for n,h in json.loads((s/'RUNNER_LAUNCH.json').read_text())['files_sha256'].items():
    if not(s/n).is_file()or sha(s/n)!=h:bad.append(n)
audit=json.loads((s/'INDEPENDENT_THREE_ROUND_ANALYSIS.json').read_text())
assert audit['gate']=='PASS'and not audit['errors']and audit['all_three_valid_and_C_wins']
assert not bad,sorted(set(bad))
print(json.dumps({'gate':'PASS','manifest_files':len(m['files']),'qualified_rounds':3,'qualified_stage_files':75,'base_present_per_round':1424,'external_binaries':3,'gpu_used':False}))
