"""Six authorized v50/A/Astar mixed replays; preparation is CPU only."""
from pathlib import Path
import ast,json,os,shutil,socket,subprocess,sys
os.sched_setaffinity(0,{28,30,84,86})
R=Path('/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1')
sys.path.insert(0,str(R/'stage29_llama_Astar_v46_normalKV_20261106_v1/common/programs'));import study_runner as L
packet=json.load(sys.stdin);assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
ROOT=(R/'cpu_candidates/v50_deployment_dry_run'/packet['dryrun_id'])if packet.get('dry_run')else R/'stage44_v50_A_Astar_llama8_nativeToT_mixed_gpu01_20261108_v1'
assert not ROOT.exists(),'Existing study: inspect, never repeat'
src=R/'stage42_v49pro_llama8_nativeToT_mixed_gpu01_20261107_v1/gpu0/cases/01_llama8_mixed_v49pro_gpu0'
w=L.read(src/'WORKLOAD.json');sourceq=L.read(src/'receipts/CPU_QUALIFICATION.json')
assert len(w['main'])==1800 and sum(len(s)for ph in ['warmup','main']for j in w[ph]for s in j['stages'])==2541
assert all(L.sha(R/n)==h for n,h in sourceq['base_files'].items())
assert L.sha(src/'WORKLOAD.json')=='4566c0f7294f79936903b010acbec567238b8ca12c1683a701dfd01bdd77fd09'
for name,s in packet['publication'].items():
 if name.endswith('.py'):ast.parse(s,filename=name)
 assert __import__('hashlib').sha256(s.encode()).hexdigest()==packet['publication_hashes'][name]
for name,s in packet['programs'].items():ast.parse(s,filename=name)
# Refresh both prior terminal states and every old registered identity.
previous={gpu:R/f'stage42_v49pro_llama8_nativeToT_mixed_gpu01_20261107_v1/gpu{gpu}'for gpu in [0,1]}
for gpu,old in previous.items():
 assert L.read(old/'STUDY_STATUS.json')['state']=='FAILED_PRESERVED'
 identities=[L.read(old/'RUNNER_LAUNCH.json')]
 for spec in L.read(old/'MATRIX.json')['cases']:
  A=Path(spec['stage'])/'attempts'/spec['attempt'];r=L.read(A/'controller_receipt.json')
  assert r['gpu_released']and not r['own_gpu_processes_remaining']and r['failure']is None and L.read(A/'CLIENT_DONE.json')['complete'];identities+=r['processes']
  assert 'FileNotFoundError'in(Path(spec['stage'])/'GATE.log').read_text()
 for i in identities:
  p=Path('/proc')/str(i['pid'])/'stat'
  if p.exists():
   st=p.read_text().rsplit(')',1)[1].split();assert int(st[19])!=i['proc_start_ticks']or st[0]in['Z','X']
ROOT.mkdir(parents=True,exist_ok=False);summary=[]
if packet.get('dry_run'):L.dump(ROOT/'CPU_PREVIEW_ONLY.json',{'GPU_launch_forbidden':True,'scope':'CPU-only matrix/asset/native-ABI qualification'})
for gpu in [0,1]:
 S=ROOT/f'gpu{gpu}';S.mkdir();(S/'cases').mkdir();(S/'common').mkdir()
 for name in ['runner.py','start.py']:(S/name).write_text(packet[name])
 scope={};exec(compile(packet['programs']['study_contract.py'],'study_contract.py','exec'),scope)
 cases=scope['case_specs'](ROOT,gpu)
 model=L.read(src/'MODEL_PROFILE.json');model['resource_binding']={'GPU':gpu,'KV_GiB':19.875,'gpu_blocks':10176,'CPU_swap_GiB':1,'same_all_methods_cards':True}
 for spec in cases:
  C=Path(spec['stage']);name=spec['name'];method=spec['version'];arm=spec['arm'];attempt=spec['attempt'];pos=spec['position'];http=spec['http_port'];pred=spec['prediction_port']
  for folder in ['attempts','receipts']:(C/folder).mkdir(parents=True,exist_ok=False)
  for folder in ['programs','assets']:shutil.copytree(src/folder,C/folder,ignore=shutil.ignore_patterns('__pycache__'))
  for obsolete in ['install_v49.py','install_v49pro.py','credit_curve.py','comparison_runtime_v47.py','comparison_shared_prediction.py']:(C/'programs'/obsolete).unlink(missing_ok=True)
  for name0,source in packet['publication'].items():(C/name0).parent.mkdir(parents=True,exist_ok=True);(C/name0).write_text(source)
  for name0,source in packet['programs'].items():(C/'programs'/name0).write_text(source)
  assert all(L.sha(C/n)==h for n,h in packet['publication_hashes'].items())
  for name0 in ['WORKLOAD.json','TRACE_MANIFEST.json']:shutil.copy2(src/name0,C/name0)
  L.dump(C/'MODEL_PROFILE.json',model)
  L.dump(C/'DEVICE_AUTHORIZATION.json',{'physical_gpu_index':gpu,'gpu_uuid':L.UUIDS[gpu],'allowed_physical_gpu_indices':[gpu],'max_attempt_wall_seconds':28800,'http_request_timeout_seconds':28200,'user_instruction':packet['user_instruction'],'only_registered_processes_may_be_signaled':True})
  L.dump(C/'LONG_EXECUTION_APPROVAL.json',{'authorized':True,'long_experiment_authorized':True,'user_instruction':packet['user_instruction'],'utc':L.utc(),'devices':[{'physical_gpu_index':gpu,'gpu_uuid':L.UUIDS[gpu]}],'scope':name,'workload_sha256':L.sha(C/'WORKLOAD.json')})
  p=L.read(src/'PROTOCOL.json')
  for obsolete in ['v49_commit','v49pro_commit','v48_commit','v47_commit']:p.pop(obsolete,None)
  p.update(schema='stage44-v50-comparison/v1',experiment_root=str(R),study_mode='stage44_v50_llama8_mixed_comparison',case=name,case_kind='CPU_PREVIEW'if packet.get('dry_run')else'formal',case_index=pos,role='formal_mixed',candidate_version=method,runtime_arm=arm,C_version='v50_anytime_stage_window_credit'if arm=='C'else'official',mixed_variant='v50_cached_counterfactual_window_v1'if arm=='C'else None,planned_arms=[arm],allowed_arms=[arm],attempts={arm:attempt},attempt_variants={attempt:'native'},arm_devices={arm:{'physical_gpu_index':gpu,'gpu_uuid':L.UUIDS[gpu],'authorization_file':'DEVICE_AUTHORIZATION.json'}},forbidden_gpu_indices=[g for g in range(3)if g!=gpu],http_ports={arm:http},prediction_ports={arm:pred},predictor_protocol='length_framed_pickle_v1',shared_prediction_version='shared_framed_dynamic_prediction_v1',cpu_affinity={arm:L.CPUS[gpu]},model_profile_sha256=L.sha(C/'MODEL_PROFILE.json'),resources={**p['resources'],'physical_gpu_index':gpu,'gpu_uuid':L.UUIDS[gpu]},long_approval_sha256=L.sha(C/'LONG_EXECUTION_APPROVAL.json'),workload_sha256=L.sha(C/'WORKLOAD.json'),baseline_budget_adapter_sha256=L.sha(C/'programs/baseline_budget.py')if method=='Astar'else None,full_trace_authorization=packet['user_instruction'],parallel=False,parallel_with_other_assigned_card=True,round='single',order=1,order_position=pos,comparability='Per-card v50/A/Astar once each in this order; same fixed input/arrival/SLO/resources and common prediction/executor observer. Two cards are not independent seeds.',C_semantics='Published v50 '+packet['v50_commit']+'; corrected stage budgets plus anytime same-window marginal token-credit projection.',A_semantics='Native A stage budgets with identical shared prediction and executor observation.',study_scope=f'Stage44 GPU{gpu} Llama8B mixed: v50 then A then Astar once each',workload_kind='mixed',observation_horizon_seconds=3600.,source_input_case=str(src),v50_commit=packet['v50_commit'],stop_rules=['Only assigned GPU0/1; never GPU2.','Exactly v50,A,Astar once each on this GPU; no repeats/retries/request drops.','Final technical gate failure preserves evidence and stops subsequent cases on this GPU.','Arrival error<=0.5s; JCT report only.'])
  if method=='v50':
   sys.path.insert(0,str(C/'programs'));from planner_config import contract;p['candidate_planner_contract']=contract();sys.path.pop(0)
  L.dump(C/'PROTOCOL.json',p)
  sys.path.insert(0,str(C/'programs'));from model_contract import verify_external;verify_external(C);from protocol_paths import experiment_root;experiment_root(C,R);sys.path.pop(0)
  for port in [http,pred]:
   with socket.socket()as sock:sock.bind(('127.0.0.1',port))
  spec['workload_sha256']=p['workload_sha256']
 C=Path(cases[0]['stage'])
 prep=L.read(R/'PREPARE.json');env=L.env();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONPATH']=os.pathsep.join([str(C/'programs'),str(R/'programs'),prep['repo'],str(Path(prep['repo'])/'third_party/vllm'),prep['sklearn_deps'],prep['support_deps']])
 tests=[]
 def cpu(name,script,data=None):
  r=subprocess.run([L.PYTHON,'-c',script],input=json.dumps(data)if data is not None else None,env=env,text=True,capture_output=True,timeout=240)
  (S/'common'/f'{name}.stdout').write_text(r.stdout);(S/'common'/f'{name}.stderr').write_text(r.stderr)
  assert r.returncode==0,(name,r.stdout[-1500:],r.stderr[-3000:])
  result=json.loads(r.stdout.strip().splitlines()[-1]);assert result['gate']=='PASS';tests.append({'name':name,**result});print(json.dumps({'GPU':gpu,'test':name,'result':result}),flush=True)
 cpu('TOKENIZER',packet['tokenizer_script'],{'case':str(C),'count':2541})
 cpu('NATIVE_STOP',packet['stop_script'],{'payload_source':packet['programs']['stage_replay.py'],'workload':w})
 for method in ['v50','A','Astar']:cpu('NATIVE_ROUTE_'+method,packet['route_script'],{'method':method,'observer':packet['programs']['native_graph_observation.py'],'tot_warmup':next(j for j in w['warmup']if j['application']=='tot_style')})
 cpu('SHARED_DYNAMIC_CACHE',packet['shared_cpu_script'])
 for spec in cases:
  C=Path(spec['stage'])
  q={'gate':'PASS','utc':L.utc(),'GPU_used':False,'base_files':sourceq['base_files'],'publication_files':packet['publication_hashes'],'v50_commit':packet['v50_commit'],'stage_files':{str(f.relative_to(C)):L.sha(f)for f in C.rglob('*')if f.is_file()and'receipts'not in f.relative_to(C).parts and'__pycache__'not in f.parts},'tests':tests,'published_CPU_qualification':packet['published_cpu']}
  L.dump(C/'receipts/CPU_QUALIFICATION.json',q);spec['qualification_sha256']=L.sha(C/'receipts/CPU_QUALIFICATION.json');L.dump(C/'CASE_SPEC.json',spec)
 matrix={'schema':'stage44-three-cases-per-card/v1','GPU':gpu,'runtime_version':'v50_anytime_stage_window_credit','v50_commit':packet['v50_commit'],'models':{'llama8':{'model_path':model['model_path'],'gpu_blocks':10176,'KV_GiB':19.875,'qrf_training_model':model['qrf_training_model']}},'root_rate':.5,'arrival_window_seconds':3600,'mix':[840,840,60,60],'cases':cases,'no_retries':True,'stop_on_any_invalid_case':True}
 scope['validate_matrix'](matrix,gpu);L.dump(S/'MATRIX.json',matrix)
 L.dump(S/'AUTHORIZATION.json',{'utc':L.utc(),'user_instruction':packet['user_instruction'],'GPU':[gpu],'formal_cases':3,'methods_in_order':['v50','A','Astar'],'model':'Llama-3.1-8B-Instruct','KV_GiB':19.875,'retry':False,'source_case':str(src),'v50_commit':packet['v50_commit']})
 L.dump(S/'SOURCE_INPUTS.json',{'source_case':str(src),'files':{n:L.sha(src/n)for n in ['WORKLOAD.json','TRACE_MANIFEST.json']},'byte_identical_workload_and_manifest':True})
 L.dump(S/'OLD_TERMINAL_STATE.json',{'study':str(previous[gpu]),'state':L.read(previous[gpu]/'STUDY_STATUS.json'),'old_registered_processes_exited':True,'old_queue_not_resumed':True})
 L.dump(S/'CPU_QUALIFICATION.json',{'gate':'PASS','GPU_used':False,'utc':L.utc(),'runner_sha256':L.sha(S/'runner.py'),'matrix_sha256':L.sha(S/'MATRIX.json'),'start_sha256':L.sha(S/'start.py'),'CPU_affinity':sorted(set(L.CPUS[gpu]+[28,30,84,86])),'tests':tests,'case_qualifications':{c['name']:c['qualification_sha256']for c in cases}})
 L.dump(S/'STUDY_STATUS.json',{'state':'PREPARED','phase':'CPU_QUALIFIED','utc':L.utc(),'pending':[c['name']for c in cases]})
 summary.append({'study':str(S),'GPU':gpu,'gate':'PASS','cases':[c['name']for c in cases],'workload_sha256':cases[0]['workload_sha256'],'CPU_qualification_sha256':L.sha(S/'CPU_QUALIFICATION.json')})
L.dump(ROOT/'TWO_CARD_MATRIX.json',{'GPUs':[0,1],'methods':['v50','A','Astar'],'cases_per_gpu':3,'studies':summary,'same_frozen_array_not_independent_seeds':True,'v50_commit':packet['v50_commit']})
print(json.dumps({'gate':'PASS','studies':summary},ensure_ascii=False))
