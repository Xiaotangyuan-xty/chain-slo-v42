"""Build an unpublished GPU0 development packet; never launches anything."""
from pathlib import Path
import ast,hashlib,json,sys
P=Path(__file__).resolve().parents[1];H=P/'harness'
label=sys.argv[1];horizon=float(sys.argv[2]);kv=float(sys.argv[3]);out=Path(sys.argv[4])
assert 0<horizon<=600 and kv in [1.,2.,4.,8.,16.,19.875]
assert label.replace('_','').isalnum()
contracts=json.loads((P/'tools/qualification_contract.json').read_text())
programs={p.name:p.read_text() for p in H.glob('*.py')}
publication={str(p.relative_to(P)):p.read_text() for folder in ['programs','assets'] for p in (P/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
for n,s in list(programs.items())+list(publication.items()):
 if n.endswith('.py'):ast.parse(s,n)
component='''import subprocess,sys,re,json,torch,os
count=0
for name in ['test_service_forecast','test_native_runtime','test_install']:
 r=subprocess.run([sys.executable,'-m','unittest',name,'-v'],env=os.environ.copy(),text=True,capture_output=True,timeout=240)
 print(r.stdout,flush=True);print(r.stderr,flush=True)
 assert r.returncode==0,(name,r.returncode)
 match=re.search(r'Ran (\\d+) tests? in',r.stderr);assert match,name;count+=int(match.group(1))
assert not torch.cuda.is_initialized()
print(json.dumps({'gate':'PASS','tests':count,'CPU_only':True,'CUDA_context_initialized':False}))
'''
packet={'dry_run':False,'root_name':'stage48_v47pro_GPU0_'+label+'_20261108_v1','methods':['v47pro'],'horizon':horizon,'KV_GiB':kv,'component':label,'http_base':33100,'prediction_base':6690,
        'user_instruction':'用户2026-11-08直接授权：暂停GPU0旧实验，以v47为基础改进v47pro，开发中途可以使用GPU0实测，提交GitHub后暂不启动GPU实验。此为未发布开发快照的一次固定混合前缀验证，完整执行所有已到达请求并排空；不重试，不操作GPU1/GPU2，不更改输入、SLO、0.5秒到达gate；JCT仅报告。',
        'v47pro_commit':'UNPUBLISHED_DEVELOPMENT_SNAPSHOT','publication':publication,'publication_hashes':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in publication.items()},'programs':programs,'runner.py':programs['runner.py'],'start.py':programs['start.py'],**contracts,'component_cpu_script':component,
        'published_cpu':{'gate':'PASS','scope':'Actual hash-bound CPU/native qualification reruns during preparation','GPU_used':False}}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(packet,ensure_ascii=False)+'\n');print(out)
