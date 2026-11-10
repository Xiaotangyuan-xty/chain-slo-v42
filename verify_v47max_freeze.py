from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;M=json.loads((R/'FREEZE_V47MAX.json').read_text())
for name,expected in M['files'].items():
 assert hashlib.sha256((R/name).read_bytes()).hexdigest()==expected,name
D=R/M['bundle'];P=json.loads((D/'PURE_PARENT_MANIFEST.json').read_text())
for n,h in P['files'].items():
 assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h,n
 assert hashlib.sha256((R/'stage48_v47pro_event_service_20261108_v1'/n).read_bytes()).hexdigest()==h,n
Q=json.loads((R/'evidence/v47max/CPU_QUALIFICATION.json').read_text());assert Q['gate']==Q['unit']['gate']==Q['native']['gate']=='PASS'
assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in Q['native']['candidate_files'].items())
print(json.dumps({'gate':'PASS','release':'v47max','parent':M['parent_commit'],'original_programs_assets_byte_identical':True,'CPU_qualification':'PASS','GPU_performance_claim':False}))
