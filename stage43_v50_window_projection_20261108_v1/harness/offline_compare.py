"""Independent calibration metrics and structural gates; SLO wins are outcomes."""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(r):return(r['phase'],r['collection_id'],r['stage_id'],r['request_id'])
def percentile(xs,p):
    if not xs:return None
    xs=sorted(xs);i=(len(xs)-1)*p;j=int(i);return xs[j]+(xs[min(j+1,len(xs)-1)]-xs[j])*(i-j)

def analyze(root,stage,attempt_name):
    from protocol_paths import experiment_root
    root=experiment_root(stage,root)
    protocol=read(stage/'PROTOCOL.json');work=read(stage/'WORKLOAD.json');attempt=stage/'attempts'/attempt_name
    calls=read(attempt/'client_calls.json');common=read(attempt/'common_audit_summary.json');receipt=read(attempt/'controller_receipt.json')
    engine={key(r):r for r in common['requests']};expected=read(stage/'TRACE_MANIFEST.json')['requests']
    flat={key(r):r for c in calls for s in c['stages']for r in s['requests']}
    bad=[]
    for k,r in flat.items():
        e=engine.get(k);m=expected.get(':'.join(map(str,k)))
        if not e or not m:bad.append(['identity',k]);continue
        if not e['finished']or e['output_tokens']!=m['target_output_tokens']or e['prompt_sha256']!=m['prompt_sha256']or e['prompt_token_ids_sha256']!=m['prompt_token_ids_sha256']:bad.append(['input_output',k])
        if r['input_prompt_sha256']!=m['prompt_sha256']or r['received_slo']!=e['received_slo']:bad.append(['client_engine',k])
    raw_seen=set();raw_bad=[]
    for line in(attempt/'engine_token_ids.jsonl').open():
        r=json.loads(line);k=key(r);e=engine.get(k)
        ih=hashlib.sha256(json.dumps(r['prompt_token_ids']).encode()).hexdigest();oh=hashlib.sha256(json.dumps(r['output_token_ids']).encode()).hexdigest()
        if k in raw_seen or not e or ih!=r['prompt_token_ids_sha256']or oh!=r['output_token_ids_sha256']or ih!=e['prompt_token_ids_sha256']or oh!=e['token_ids_sha256']:raw_bad.append(str(k))
        raw_seen.add(k)
    main=[c for c in calls if c['phase']=='main'];origin=read(attempt/'MAIN_START.json')['origin_monotonic']
    raw_duration=max(c['client_chain_end_monotonic']for c in main)-origin
    duration=max(3600.,raw_duration)if protocol['workload_kind']=='mixed'else raw_duration
    arr=max(abs(c['arrival_offset_seconds']-c['planned_arrival_offset_seconds'])for c in main)
    classes={str(t):{'jobs':0,'requests':0,'attained_jobs':0,'actual_output_tokens':0,'actual_input_tokens':0,'gated_weighted_tokens':0,'official_formula_weighted_tokens':0,'client_ttft_text':[],'server_ttft':[],'tbt':[],'e2el':[],'complete_tbt_coverage_requests':0}for t in [0,1,2]}
    chains=[];barrier_bad=[];signatures={}
    for c in main:
        t=c['request_type'];group=classes[str(t)];group['jobs']+=1;rs=[r for s in c['stages']for r in s['requests']]
        group['requests']+=len(rs);jct=c['client_chain_end_monotonic']-c['client_chain_start_monotonic'];budget=c['received_slo'][2]*len(c['stages'])
        if t==2:attained=c['success']and jct<=budget
        elif t==1:attained=c['success']and jct<=c['received_slo'][2]
        else:
            r=rs[0];e=engine[key(r)];covered=len(r['server_tbt'])==max(0,e['output_tokens']-1)
            first=r.get('client_first_text_monotonic');text_ttft=None if first is None else first-r['client_request_start_monotonic']
            attained=r['success']and text_ttft is not None and text_ttft<=r['received_slo'][0]and covered and all(x<=r['received_slo'][1]for x in r['server_tbt'])
        group['attained_jobs']+=int(bool(attained));chains.append({'collection_id':c['collection_id'],'request_type':t,'jct':jct,'budget':budget if t==2 else c['received_slo'][2],'attained':bool(attained)})
        for previous,nxt in zip(c['stages'],c['stages'][1:]):
            if max(r['client_request_end_monotonic']for r in previous['requests'])>min(r['client_request_start_monotonic']for r in nxt['requests']):barrier_bad.append(c['collection_id'])
        for r in rs:
            e=engine[key(r)];inp,out=e['prompt_tokens'],e['output_tokens'];group['actual_input_tokens']+=inp;group['actual_output_tokens']+=out
            group['e2el'].append(r['client_request_end_monotonic']-r['client_request_start_monotonic'])
            ttft=r.get('server_ttft');tbts=r['server_tbt'];group['server_ttft'].append(ttft);group['tbt'].extend(tbts)
            if r.get('client_first_text_monotonic')is not None:group['client_ttft_text'].append(r['client_first_text_monotonic']-r['client_request_start_monotonic'])
            group['complete_tbt_coverage_requests']+=int(len(tbts)==max(0,out-1))
            if attained:group['gated_weighted_tokens']+=inp+8*out
            if t==0:
                cumulative=ttft;good=0
                for i,interval in enumerate(tbts):
                    cumulative+=interval;good+=cumulative<=r['received_slo'][0]+r['received_slo'][1]*(i+1)
                group['official_formula_weighted_tokens']+=(inp if ttft<=r['received_slo'][0]else 0)+8*good
            elif attained:group['official_formula_weighted_tokens']+=inp+8*out
            signatures[str(key(r))]={k:e[k]for k in ['prompt_sha256','prompt_token_ids_sha256','generation','received_slo','output_tokens','finish_reasons','stop_reasons']}
    main_jobs={j['collection_id']:j for j in work['main']}
    checks={'cpu_gate_pass':read(stage/'receipts/CPU_QUALIFICATION.json')['gate']=='PASS',
            'exact_frozen_workload_sha':sha(stage/'WORKLOAD.json')==protocol['workload_sha256'],
            'all_main_jobs_complete':len(main)==len(work['main'])and {c['collection_id']for c in main}==set(main_jobs)and all(c['success']for c in main),
            'all_request_identities_preserved':len(flat)==len(engine)==len(expected)and raw_seen==set(engine)and set(flat)==set(engine),
            'actual_inputs_lengths_and_slo_preserved':not bad,'raw_token_hashes_verified':not raw_bad,'stage_barriers_preserved':not barrier_bad,
            'arrival_error_le_0_5':arr<=.5,
            'controller_cleanup':receipt['failure']is None and receipt['client_returncode']==0 and receipt['gpu_released']and not receipt['own_gpu_processes_remaining'],
            'only_assigned_GPU':common['device']['physical_gpu_index']==protocol['arm_devices'][receipt['arm']]['physical_gpu_index'] and common['device']['actual_uuid']==protocol['arm_devices'][receipt['arm']]['gpu_uuid'],
            'same_resource_caps':all(common['resources'].get(k)==v for k,v in protocol['expected_resource_caps'].items()),
            'all_KV_released':common['final_kv_free']=={'gpu_blocks':protocol['expected_resource_caps']['gpu_blocks'],'cpu_blocks':protocol['expected_resource_caps']['cpu_blocks']},
            'model_assets_verified_before_GPU':receipt.get('model_asset_verification',{}).get('model_path')==protocol['model_path'] and receipt.get('model_asset_verification',{}).get('gate')=='PASS',
            'client_completed_finite_replay':read(attempt/'CLIENT_DONE.json')['complete'],
            'native_executor':common['resources']['native_executor']}
    checks['complete_TBT_main']=all(v['complete_tbt_coverage_requests']==v['requests'] for v in classes.values())
    checks['common_statistics_repair_declared']=(attempt/'SAFE_STATISTICS_RUNTIME.json').exists()
    checks['frozen_arrival_window_and_drain']=work['main'][-1]['arrival_offset_seconds']==protocol['expected_arrival_window_seconds']and duration>=work['main'][-1]['arrival_offset_seconds']
    if protocol['workload_kind']=='mixed':checks['fixed_hour_observation']=all(0<=j['arrival_offset_seconds']<3600. for j in work['main'])and read(attempt/'CLIENT_DONE.json')['observation_end_monotonic']>=origin+3600.
    from validate_lifetime import validate as validate_lifetime
    checks.update(validate_lifetime(attempt,flat,engine))
    mixed=read(attempt/'MIXED_RUNTIME.json')if(attempt/'MIXED_RUNTIME.json').exists()else None
    if receipt['arm']=='A':checks['native_A_scheduler_policy']=common['resources']['scheduler_type'].endswith('.SLOScheduler')and common['resources']['policy_type'].endswith('.SLOPolicy')
    else:
        checks['declared_C_adapter']=mixed is not None and mixed['variant']==protocol['mixed_variant']
        checks['declared_candidate_version']=common['resources']['mechanism_version']==protocol['C_version']
        checks['declared_candidate_corrected_budget']=(attempt/'V50_BUDGET_ADAPTER.json').exists()and read(attempt/'V50_BUDGET_ADAPTER.json')['semantics_equal_Astar']and not(attempt/'BASELINE_BUDGET_ADAPTER.json').exists()
        checks['no_superseded_execution_cadence_adapter']=not (attempt/'V46_EXECUTION_OBSERVATION.json').exists()
        checks['no_outside_candidate_selection']=common['stats'].get('outside_selected_executed_groups',0)==0
        checks['candidate_selection_audit_exercised']=common['stats'].get('selection_checked_execute_calls',0)>0
        runtime=read(attempt/'V50_RUNTIME.json')
        plans=sum(s['mixed_counts'].get('joint_plans',0)for s in runtime['schedulers'])
        logged=sum(1 for line in(attempt/'V50_DECISIONS.jsonl').open()if json.loads(line)is not None)
        checks['v50_all_decision_plans_streamed']=runtime['logged_plan_events']==plans==logged and all(s['unstreamed_plan_events_lost']==0 for s in runtime['schedulers'])
        checks['v50_shared_comparison_ready']=runtime['prediction_comparison_ready']and runtime['model_key']==protocol['model_key']
    # Common observer validation covers A/Astar too; only the candidate owns a
    # cost model. Baseline receipts must not be compared to nonexistent samples.
    observation=read(attempt/'EXECUTOR_OBSERVATION.json')
    events=0;failed=0;malformed=[]
    for line in(attempt/'EXECUTOR_EVENTS.jsonl').open():
        event=json.loads(line);events+=1;failed+=event['error']is not None
        if not event['seconds']>0 or abs(event['seconds']-(event['end_monotonic']-event['start_monotonic']))>1e-9:malformed.append(events)
    checks['executor_observation_finalized_and_complete']=events==observation['events']and events>0
    if receipt['arm']=='C':
        accepted=sum(s['step_cost_stats'].get('accepted_samples',0)for s in runtime['schedulers'])
        checks['candidate_cost_samples_cover_executor']=events==runtime['execution_observer']['events']==accepted
    checks['executor_calls_valid_without_failure']=not malformed and failed==observation['failed_calls']==0
    checks['no_added_CUDA_synchronization']=not observation['CUDA_synchronization_added']
    if protocol.get('candidate_version')=='Astar':
        adapter=read(attempt/'BASELINE_BUDGET_ADAPTER.json')
        shares=[]
        for line in(attempt/'server_events.jsonl').open():
            event=json.loads(line)
            if event.get('kind')=='graph_stage_share'and event.get('stage_id')==0:shares.append(event['share'])
        checks['Astar_corrected_stage_budget']=adapter['variant']=='cumulative_through_current_declared_depth_v1'and bool(shares)and all(0<x<=1 for x in shares)
        checks['Astar_core_native_policy_retained']=common['resources']['scheduler_type']=='jitserve.scheduler.slo_scheduler.SLOScheduler'and common['resources']['policy_type']=='jitserve.scheduler.policy.SLOPolicy'
    for v in classes.values():
        for k in ['client_ttft_text','server_ttft','tbt','e2el']:
            vals=v[k];v[k]={'count':len(vals),'mean':statistics.fmean(vals)if vals else None,'p50':percentile(vals,.5),'p95':percentile(vals,.95),'max':max(vals)if vals else None}
        v['attained_job_rate']=v['attained_jobs']/v['jobs'] if v['jobs'] else None;v['gated_weighted_goodput']=v['gated_weighted_tokens']/duration;v['official_formula_weighted_goodput']=v['official_formula_weighted_tokens']/duration
    result={'schema':'stage19-calibration-arm/v1','arm':receipt['arm'],'attempt':attempt_name,'gate':'PASS'if all(checks.values())else'FAIL','checks':checks,
            'metrics':{'main_duration':duration,'main_root_jobs':len(main),'main_requests':sum(v['requests']for v in classes.values()),
                       'main_output_tokens':sum(v['actual_output_tokens']for v in classes.values()),'output_tokens_per_second':sum(v['actual_output_tokens']for v in classes.values())/duration,
                       'max_arrival_error':arr,'compound_attained':classes['2']['attained_jobs'],'compound_total':classes['2']['jobs']},
            'classes':classes,'chains':chains,'mixed_runtime':mixed,'execution_stats':common['stats'],'scheduler_stats':read(attempt/'server_audit_summary.json')['scheduler_stats'],
            'request_signatures':signatures,'scope':protocol['comparability'],'bad_inputs':bad,'raw_token_errors':raw_bad}
    from native_graph_observation import shared_checks,route_checks
    checks.update(shared_checks(attempt,protocol,len(expected)))
    checks['warmup_end_to_end_gate_pass']=read(attempt/'WARMUP_NATIVE_TOT_GATE.json')['gate']=='PASS'
    route_gate,route_bad=route_checks(attempt,work,protocol['candidate_version'],'main');checks.update(route_gate)
    if protocol['workload_kind']=='mixed':
        from alignment_metrics import analyze as aligned_analyze
        gp=aligned_analyze(root,stage,attempt_name,work,calls,raw_duration)
        checks.update(gp['checks'])
        result['gate']='PASS'if all(checks.values())else'FAIL'
        result['metrics'].update(gp['metrics'])
        result['aligned_goodput']=gp['summary']
        result['application_contributions']=gp['applications']
        for k,values in classes.items():
            values['engine_token_formula_weighted_tokens']=values.pop('official_formula_weighted_tokens')
            values['engine_token_formula_weighted_goodput']=values.pop('official_formula_weighted_goodput')
            values['native_full_credited_weighted_tokens']=gp['summary']['complete_replay']['credit_by_class'][int(k)]
            values['native_fixed_hour_credited_weighted_tokens']=gp['summary']['fixed_hour']['credit_by_class'][int(k)]
        result['metric_attribution']='credit settlement in fixed[0,3600), complete replay separately'
    else:
        from jitserve_goodput import analyze as primary_analyze
        gp=primary_analyze(root,stage,attempt_name,duration)
        assert gp['compound_attained']==result['metrics']['compound_attained']
        result['metrics'].update(jitserve_token_goodput=gp['jitserve_token_goodput'],jitserve_credited_weighted_tokens=gp['credited_weighted_tokens'],mean_JCT=statistics.fmean(c['jct']for c in chains),p95_JCT=percentile([c['jct']for c in chains],.95),drain_seconds=duration-work['main'][-1]['arrival_offset_seconds'],request_failures=0)
        result['jitserve_goodput']=gp
        sampling=protocol['generation']
        checks['actual_engine_sampling_matches_frozen_contract']=all(e['generation'].get('top_k')==sampling['top_k'] and e['generation'].get('temperature')==0. and e['generation'].get('target_output_length')==expected[':'.join(map(str,key(e)))]['target_output_tokens']for e in engine.values())
    result['gate']='PASS'if all(checks.values())else'FAIL'
    (stage/'receipts'/('RUN_GATE_'+attempt_name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    stage=Path(sys.argv[2]);r=analyze(Path(sys.argv[1]),stage,sys.argv[3]);print(json.dumps({k:r[k]for k in ['arm','gate','checks','metrics','classes','mixed_runtime']},indent=2));raise SystemExit(0 if r['gate']=='PASS'else 1)
