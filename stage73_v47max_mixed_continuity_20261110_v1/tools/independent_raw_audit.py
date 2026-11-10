"""Read-only terminal audit with separate raw credit arithmetic; no GPU context."""
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib,json,math,os,statistics,sys
os.sched_setaffinity(0,{28,30,84,86});sys.dont_write_bytecode=True
R=Path('/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1')
packet=json.load(sys.stdin);S=R/packet['root_name']/'gpu0'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(r):return r['phase'],r['collection_id'],r['stage_id'],r['request_id']
spec=read(S/'MATRIX.json')['cases'][0];C=Path(spec['stage']);A=C/'attempts'/spec['attempt']
state=read(S/'STUDY_STATUS.json');g=read(C/'receipts'/('RUN_GATE_'+spec['attempt']+'.json'))
assert state['state']=='COMPLETED' and g['gate']=='PASS' and all(g['checks'].values()),'No partial/failed result promoted'
protocol=read(C/'PROTOCOL.json');q=read(C/'receipts/CPU_QUALIFICATION.json');receipt=read(A/'controller_receipt.json');common=read(A/'common_audit_summary.json')
checks={'original_gate_and_terminal_queue':True,'controller_cleanup':receipt['failure'] is None and receipt['gpu_released'] and not receipt['own_gpu_processes_remaining']}
changed=[]
for base,files in [(R,q['base_files']),(C,q['stage_files']),(C,q['publication_files'])]:
 for rel,expected in files.items():
  if sha(base/rel)!=expected:changed.append(str(base/rel))
checks['hash_bound_source_unchanged']=not changed
work=read(C/'WORKLOAD.json');calls=read(A/'client_calls.json');manifest=read(C/'TRACE_MANIFEST.json')['requests']
expected={}
for phase in ['warmup','main']:
 for job in work[phase]:
  for sid,members in enumerate(job['stages']):
   for member in members:expected[(phase,job['collection_id'],sid,member['request_id'])]=(job,member)
flat_list=[r for c in calls for s in c['stages'] for r in s['requests']];flat={key(r):r for r in flat_list};engine={key(r):r for r in common['requests']}
checks['all_identities_exact']=len(flat_list)==len(flat)==len(engine)==len(expected)==len(manifest) and set(flat)==set(engine)==set(expected)
bad=[]
for k,(job,member) in expected.items():
 e=engine[k];r=flat[k];m=manifest[':'.join(map(str,k))]
 gen={'temperature':0.,'top_p':1.,'top_k':-1,'seed':42,'ignore_eos':True,'n':1,'max_tokens':256 if job.get('application')=='tot_style' else 1024,'target_output_length':member['target_output_tokens']}
 valid=(r['success'] and e['finished'] and r['target_output_tokens']==e['output_tokens']==member['target_output_tokens']==m['target_output_tokens'] and r['received_slo']==e['received_slo']==job['received_slo'] and r['input_prompt_sha256']==e['prompt_sha256']==m['prompt_sha256']==hashlib.sha256(member['prompt'].encode()).hexdigest() and e['prompt_token_ids_sha256']==m['prompt_token_ids_sha256'] and all(e['generation'].get(n)==v for n,v in gen.items()))
 if not valid:bad.append(str(k))
checks['inputs_SLO_sampling_lengths_complete']=not bad
raw_seen=set();raw_bad=[]
for line in (A/'engine_token_ids.jsonl').open():
 r=json.loads(line);k=key(r)
 if k in raw_seen or hashlib.sha256(json.dumps(r['prompt_token_ids']).encode()).hexdigest()!=r['prompt_token_ids_sha256'] or hashlib.sha256(json.dumps(r['output_token_ids']).encode()).hexdigest()!=r['output_token_ids_sha256'] or r['output_token_ids_sha256']!=engine[k]['token_ids_sha256'] or r['prompt_token_ids_sha256']!=engine[k]['prompt_token_ids_sha256']:raw_bad.append(str(k))
 raw_seen.add(k)
checks['raw_token_hashes_coverage']=raw_seen==set(expected) and not raw_bad
lifetimes={};life_bad=[]
for line in (A/'LIFETIME_TIMINGS.jsonl').open():
 r=json.loads(line);k=key(r);ts=r['token_append_monotonic']
 if k in lifetimes or len(ts)!=engine[k]['output_tokens'] or r['request_uuid']!=engine[k]['request_uuid'] or [b-a for a,b in zip(ts,ts[1:])]!=r['TBT'] or r['TBT']!=flat[k]['server_tbt'] or r['TTFT']!=flat[k]['server_ttft']:life_bad.append(str(k))
 lifetimes[k]=ts
checks['complete_lifetime_TBT']=set(lifetimes)==set(expected) and not life_bad
checks['barriers']=all(max(r['client_request_end_monotonic'] for r in prev['requests'])<=min(r['client_request_start_monotonic'] for r in nxt['requests']) for c in calls for prev,nxt in zip(c['stages'],c['stages'][1:]))
main=[c for c in calls if c['phase']=='main'];jobs={j['collection_id']:j for j in work['main']};origin=read(A/'MAIN_START.json')['origin_monotonic'];h=protocol['observation_horizon_seconds'];duration=max(c['client_chain_end_monotonic'] for c in main)-origin
arrival=max(abs(c['client_chain_start_monotonic']-origin-jobs[c['collection_id']]['arrival_offset_seconds']) for c in main)
checks['arrival_error_le_0_5']=arrival<=.5
checks['full_prefix_and_observation']=len(main)==len(jobs) and {c['collection_id'] for c in main}==set(jobs) and all(c['success'] for c in calls) and read(A/'CLIENT_DONE.json')['observation_end_monotonic']>=origin+h
sys.path.insert(0,str(C/'programs'))
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(protocol['model_path'],local_files_only=True,trust_remote_code=False)
ledger=[];full=[0,0,0];inside=[0,0,0];apps=defaultdict(lambda:{'window_credit':0,'full_credit':0,'roots':0,'attained':0});native_total=[0,0,0]
from jitserve_mixed_goodput import native_function
oracle=native_function(R)
def add(t,amount,kind,cid,sid=None,rid=None):
 if amount:ledger.append({'time_seconds':t-origin,'weighted_tokens':int(amount),'type':kind,'collection_id':cid,'stage_id':sid,'request_id':rid})
for c in main:
 cid=c['collection_id'];kind=c['request_type'];job=jobs[cid];slo=c['received_slo'];terminal=c['client_chain_end_monotonic'];jct=terminal-c['client_chain_start_monotonic'];chain_met=jct<=slo[2]*len(job['stages']);root_credit=0;outputs=[]
 app=job.get('application','dr');apps[app]['roots']+=1
 for stage in c['stages']:
  for r in stage['requests']:
   k=key(r);prompt=expected[k][1]['prompt'];pin=len(tokenizer.encode(prompt,add_special_tokens=False));pout=len(tokenizer.encode(r['generated_text'],add_special_tokens=False));ts=lifetimes[k]
   if kind==0:
    first=r.get('client_first_text_monotonic')
    apps[app]['attained']+=bool(first is not None and first-r['client_request_start_monotonic']<=slo[0] and all(dt<=slo[1] for dt in r['server_tbt']))
    add(ts[0],pin if r['server_ttft']<=slo[0] else 0,0,cid,r['stage_id'],r['request_id'])
    elapsed=r['server_ttft']
    for i,dt in enumerate(r['server_tbt']):
     elapsed+=dt
     if elapsed<=slo[0]+slo[1]*(i+1):add(ts[i+1],8,0,cid,r['stage_id'],r['request_id'])
   elif kind==1:
    met=r['client_request_end_monotonic']-r['client_request_start_monotonic']<=slo[2]
    add(r['client_request_end_monotonic'],pin+8*pout if met else 0,1,cid,r['stage_id'],r['request_id']);apps[app]['attained']+=met
   else:root_credit+=pin+8*pout
   outputs.append(NS(request_type=kind,collection_id=cid,success=True,request_input=prompt,request_output=r['generated_text'],request_ttft=r['server_ttft'],request_tbt=r['server_tbt'],request_latency=r['client_request_end_monotonic']-r['client_request_start_monotonic'],request_finish_time=r['client_request_end_monotonic'],request_service_gain=r.get('logged_service_gain',0),finish_before_ddl=chain_met))
 if kind==2:
  add(terminal,root_credit if chain_met else 0,2,cid);apps[app]['attained']+=chain_met
 native=oracle([outputs],max(h,duration),tokenizer,[],slo_constraint=tuple(v/(cid%4+1) for v in slo))
 native_total[kind]+=int(sum(z['token_goodput'] for z in native.request_details))
ledger.sort(key=lambda e:(e['time_seconds'],e['type'],e['collection_id']))
for e in ledger:
 kind=e['type'];amount=e['weighted_tokens'];full[kind]+=amount;app=jobs[e['collection_id']].get('application','dr');apps[app]['full_credit']+=amount
 if 0<=e['time_seconds']<h:inside[kind]+=amount;apps[app]['window_credit']+=amount
original=[json.loads(line) for line in (A/'CREDIT_LEDGER.jsonl').open()]
checks['separate_raw_ledger_exact']=ledger==original
checks['valid_settlement_times']=all(math.isfinite(e['time_seconds']) and e['time_seconds']>=0 for e in ledger)
checks['native_oracle_credit_exact']=native_total==full
checks['raw_window_and_full_metrics_exact']=(sum(inside)==g['metrics']['fixed_hour_credited_weighted_tokens'] and sum(inside)/h==g['metrics']['fixed_hour_token_goodput'] and sum(full)==g['metrics']['jitserve_credited_weighted_tokens'] and sum(full)/max(h,duration)==g['metrics']['complete_replay_token_goodput'])
plans=[];masks={};plan_ids=set();mode_counts=Counter();reasons=Counter();preemptions=Counter();partial=[]
for line in (A/'V47PRO_DECISIONS.jsonl').open():
 e=json.loads(line);kind=e['kind']
 if kind=='plan':
  plans.append(e);plan_ids.add(e['decision_id']);reasons.update(e['reasons']);mode_counts[e['mode']]+=1
  partial.extend({'decision_id':e['decision_id'],**r} for r in e['rows'] if r['selected'] and len(r['selected'])<len(r['members']))
 elif kind=='execution_mask':masks[e['mask_id']]=e
 elif kind=='preempt':preemptions[e['preemption']]+=1
exec_count=0;exec_bad=[];schedule_seconds=[];swap_dependency_rounds=0
for line in (A/'EXECUTOR_EVENTS.jsonl').open():
 e=json.loads(line);exec_count+=1;schedule_seconds.append(e.get('scheduler_seconds',0.))
 if e['error'] or not e['seconds']>0:exec_bad.append(exec_count)
 if e.get('decision_id',0) not in plan_ids|{0}:exec_bad.append(exec_count)
 if e.get('candidate_filter'):
  mask=masks.get(e.get('mask_id'))
  if mask is None or mask['monotonic']>e['start_monotonic'] or mask['mode_epoch']!=e['mode_epoch'] or e['members']>len(mask['allowed']):exec_bad.append(exec_count)
 swap_dependency_rounds+=bool(e.get('swap_out_blocks') and not e['members'])
checks['executor_mask_decision_integrity']=not exec_bad and exec_count==read(A/'EXECUTOR_OBSERVATION.json')['events']
checks['no_outside_allowed_execution']=common['stats'].get('outside_selected_executed_groups',0)==0
processes=[]
for reg in receipt['processes']+[read(S/'RUNNER_LAUNCH.json')]:
 try:
  f=Path('/proc')/str(reg['pid']);stat=(f/'stat').read_text().rsplit(')',1)[1].split();same=int(stat[19])==reg['proc_start_ticks'] and stat[0] not in ['Z','X']
 except FileNotFoundError:same=False
 processes.append({'pid':reg['pid'],'same_live_identity':same})
checks['registered_processes_exited']=not any(p['same_live_identity'] for p in processes)
checks['KV_release_and_caps']=common['final_kv_free']=={'gpu_blocks':spec['gpu_blocks'],'cpu_blocks':512} and all(common['resources'].get(k)==v for k,v in protocol['expected_resource_caps'].items())
import torch
checks['CPU_only_no_CUDA']=not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES']==''
summary={'gate':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'failed_checks':[k for k,v in checks.items() if not v],'study':str(S),'horizon_seconds':h,'KV_GiB':spec['KV_GiB'],'main_roots':len(main),'main_calls':len([r for r in flat.values() if r['phase']=='main']),'window_credit':sum(inside),'window_goodput':sum(inside)/h,'credit_by_class':inside,'full_credit':sum(full),'complete_replay_goodput':sum(full)/max(h,duration),'main_duration_seconds':duration,'drain_seconds':max(0,duration-h),'arrival_error_seconds':arrival,'applications':dict(apps),'gate_reported_metrics':g['metrics'],
         'trace':{'plans':len(plans),'replan_reasons':dict(reasons),'plan_modes':dict(mode_counts),'preemptions':dict(preemptions),'partial_stage_rows':len(partial),'partial_stage_examples':partial[:5],'scheduler_host_seconds_mean':statistics.fmean(schedule_seconds),'scheduler_host_seconds_p95':sorted(schedule_seconds)[min(len(schedule_seconds)-1,int(.95*len(schedule_seconds)))],'scheduler_host_seconds_max':max(schedule_seconds),'swap_dependency_empty_rounds':swap_dependency_rounds,'executor_calls':exec_count},'processes':processes,'source_publication_hashes':q['publication_files'],
         'raw_files':{str(p.relative_to(S)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in [C/'PROTOCOL.json',C/'WORKLOAD.json',C/'TRACE_MANIFEST.json',C/'receipts'/('RUN_GATE_'+spec['attempt']+'.json'),A/'client_calls.json',A/'LIFETIME_TIMINGS.jsonl',A/'engine_token_ids.jsonl',A/'V47PRO_DECISIONS.jsonl',A/'EXECUTOR_EVENTS.jsonl',A/'V47PRO_RUNTIME.json',A/'CREDIT_LEDGER.jsonl']}}
print(json.dumps(summary,ensure_ascii=False))
