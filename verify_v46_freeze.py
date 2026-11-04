"""Verify v46 source/model/evidence hashes. Never initialize CUDA."""
from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parent
manifest=json.loads((R/'FREEZE_V46.json').read_text());bad=[]
for name,expected in manifest['files'].items():
 p=R/name
 if not p.is_file()or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:bad.append(name)
print(json.dumps({'gate':'PASS'if not bad else'FAIL','files_checked':len(manifest['files']),'bad':bad,'GPU_used':False}))
raise SystemExit(1 if bad else 0)
