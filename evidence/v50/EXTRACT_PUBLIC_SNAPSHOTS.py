"""Causal public-state snapshots from completed v49pro traces; no GPU calls."""
from pathlib import Path
from collections import defaultdict
import bisect,json,os,sys
os.sched_setaffinity(0,{28,30,84,86});assert os.environ['CUDA_VISIBLE_DEVICES']==''
from step_cost import StepCost
R=Path('/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1');D=R/'stage42_v49pro_llama8_nativeToT_mixed_gpu01_20261107_v1'
fixtures=[]
for gpu in [0,1]:
 C=D/f'gpu{gpu}/cases/01_llama8_mixed_v49pro_gpu{gpu}';A=next((C/'attempts').iterdir())
 read=lambda p:json.loads(p.read_text())
 work=read(C/'WORKLOAD.json');jobs={j['collection_id']:j for j in work['main']};calls=read(A/'client_calls.json');roots={c['collection_id']:c for c in calls if c['phase']=='main'}
 engine=[r for r in read(A/'common_audit_summary.json')['requests']if r['phase']=='main'];lifetimes={}
 for line in(A/'LIFETIME_TIMINGS.jsonl').open():
  v=json.loads(line);lifetimes[v['request_uuid']]=v
 warm=next(v for v in lifetimes.values()if v['phase']=='warmup');offset=warm['first_token_wall']-warm['token_append_monotonic'][0]
 prediction={}
 for line in(A/'SHARED_PREDICTION_EVENTS.jsonl').open():
  v=json.loads(line)
  if v['kind']=='prediction_success':prediction[v['collection_id'],v['stage_id'],v['request_id']]=v
 plans=[json.loads(l)for l in(A/'V49PRO_DECISIONS.jsonl').open()]
 chosen=[plans[min(len(plans)-1,i*len(plans)//32)]for i in range(32)]
 costs=StepCost();stream=iter((A/'EXECUTOR_EVENTS.jsonl').open());pending=json.loads(next(stream));mismatches=0
 for index,plan in enumerate(chosen):
  t=plan['monotonic']
  while pending is not None and pending['end_monotonic']<=t:
   costs.observe_execution(pending,pending['seconds'])
   try:pending=json.loads(next(stream))
   except StopIteration:pending=None
  active=[r for r in engine if r['start_monotonic']<=t<r['finish_monotonic']]
  running=set(plan['running_before']);members=[];grouped=defaultdict(list);actions=[]
  for r in active:
   rid=r['request_uuid'];life=lifetimes[rid];gen=bisect.bisect_right(life['token_append_monotonic'],t)
   pr=prediction.get((r['collection_id'],r['stage_id'],r['request_id']));pred=pr['prediction']if pr and pr['monotonic']<=t else 1024
   cap=r['generation']['max_tokens'];prefill=r['prompt_tokens']if gen==0 else 0
   state='running'if rid in running else'swapped'if gen else'waiting'
   elapsed=max(0.,t+offset-life['arrival_wall']);remaining=max(1,min(cap,pred)-gen)
   unit=plan['detail'].get('cost_model',{}).get('fallback_unit',.04);service=max(.001,remaining*unit)
   croot=roots[r['collection_id']];job=jobs[r['collection_id']]
   slack=20*len(job['stages'])-(t-croot['client_chain_start_monotonic'])if r['request_type']==2 else r['received_slo'][2]-elapsed
   reward=r['prompt_tokens']+8*min(cap,pred)
   if r['request_type']==2:ratio=min(1.,max(0.,(20*(r['stage_id']+1)-(t-croot['client_chain_start_monotonic']))/service))
   elif r['request_type']==1:ratio=min(1.,max(0.,slack/service))
   else:ratio=1.
   priority=(1 if slack<=0 else 0,-reward*ratio**100/service,service)
   member={'request':rid,'member':r['request_id'],'prompt':r['prompt_tokens'],'declared_input':next(z['declared_input_tokens']for z in job['stages'][r['stage_id']]if z['request_id']==r['request_id']),
       'generated':gen,'max_tokens':cap,'prefill_tokens':prefill,'reward_input':max(0,r['prompt_tokens']-1),
       'arrival':life['arrival_wall'],'native_priority':priority,'predicted_remaining':remaining,'state':state,'can_swap':True}
   members.append(member)
   if r['request_type']==2:grouped[r['collection_id']].append((r,member,pred,slack))
   else:
    context=(r['request_type'],member['reward_input'],gen,pred,cap,prefill,elapsed,r['received_slo'])
    actions.append({'id':'request:'+rid,'kind':'foreground','members':[rid],'foreground':context,'arrival':life['arrival_wall'],'native_priority':priority,
         'cheap_credit':reward if (elapsed<(r['received_slo'][2]if r['request_type']==1 else r['received_slo'][0]+r['received_slo'][1]*(cap-1)))else 0.,'cheap_work':remaining})
  for cid,gs in grouped.items():
   stage=gs[0][0]['stage_id'];job=jobs[cid];observed={}
   for r in engine:
    if r['collection_id']!=cid or r['finish_monotonic']>t:continue
    observed.setdefault(r['stage_id'],{})[r['request_id']]={'declared_input':next(z['declared_input_tokens']for z in job['stages'][r['stage_id']]if z['request_id']==r['request_id']),
         'output':r['output_tokens'],'reward_input':max(0,r['prompt_tokens']-1),'reward_output':r['output_tokens']}
   meta={'stages':len(job['stages']),'widths':[len(s)for s in job['stages']],'observed':observed,'future_max_tokens':256 if job['application']=='tot_style'else 1024}
   prefix=sum(z['reward_input']+8*z['reward_output']for ms in observed.values()for z in ms.values())
   views=[{k:m[k]for k in ['request','member','declared_input','prompt','generated','max_tokens','prefill_tokens','reward_input']}for r,m,pr,slack in gs]
   cheap=prefix+sum(m['reward_input']+8*(m['generated']+m['predicted_remaining'])for r,m,pr,slack in gs)
   actions.append({'id':'chain:'+str(cid),'kind':'compound','members':[m['request']for r,m,pr,slack in gs],
      'arrival':min(m['arrival']for r,m,pr,slack in gs),'native_priority':min(m['native_priority']for r,m,pr,slack in gs),
      'cheap_credit':cheap if gs[0][3]>0 else 0.,'cheap_work':sum(m['predicted_remaining']for r,m,pr,slack in gs),
      'slack':gs[0][3],'domain':'tot'if job['application']=='tot_style'else'dr','meta':meta,'stage':stage,'views':views})
  snapshot=costs.snapshot(plan['detail'].get('cost_model',{}).get('fallback_unit',.04),max(1,len(running)))
  mismatch=len(active)!=plan['physical_candidates'];mismatches+=mismatch
  fixtures.append({'GPU':gpu,'index':index,'monotonic':t,'actions':actions,'members':members,'running':sorted(running&{m['request']for m in members}),
     'capacity':plan['capacity'],'slots':plan['slots'],'unit':snapshot.fallback,'cost_strata':snapshot.strata,'cost_reference':snapshot.reference,
     'recorded_physical_count':plan['physical_candidates'],'reconstructed_physical_count':len(active),'membership_mismatch':mismatch})
 print(json.dumps({'progress_gpu':gpu,'snapshots':32,'membership_mismatches':mismatches}),flush=True)
print(json.dumps({'SNAPSHOTS':fixtures,'scope':'Open-loop public observed state/timing qualification only; native priority is a corrected-uniform-budget proxy, not an exact counterfactual GPU execution. No future output lengths or future arrival arrays supplied to the planner.'}),flush=True)
