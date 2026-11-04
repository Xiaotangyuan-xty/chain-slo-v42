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

def analyze(stage,attempt_name):
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
    duration=max(c['client_chain_end_monotonic']for c in main)-origin
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
            'same_resource_caps':all(common['resources'][k]==v for k,v in {'gpu_blocks':512,'cpu_blocks':1170,'block_size':16,'max_num_seqs':32,'max_num_batched_tokens':2048,'max_model_len':8192,'single_engine':True,'enable_prefix_caching':False}.items()),
            'client_completed_finite_replay':read(attempt/'CLIENT_DONE.json')['complete'],
            'native_executor':common['resources']['native_executor']}
    checks['complete_TBT_main']=all(v['complete_tbt_coverage_requests']==v['requests'] for v in classes.values())
    checks['common_statistics_repair_declared']=(attempt/'SAFE_STATISTICS_RUNTIME.json').exists()
    checks['frozen_arrival_window_and_drain']=work['main'][-1]['arrival_offset_seconds']==protocol['expected_arrival_window_seconds']and duration>=work['main'][-1]['arrival_offset_seconds']
    if protocol['workload_kind']=='mixed':checks['full_one_hour_arrivals']=work['main'][-1]['arrival_offset_seconds']==3600.0
    from validate_lifetime import validate as validate_lifetime
    checks.update(validate_lifetime(attempt,flat,engine))
    mixed=read(attempt/'MIXED_RUNTIME.json')if(attempt/'MIXED_RUNTIME.json').exists()else None
    from validate_v46 import validate as validate_v46
    checks.update(validate_v46(attempt,protocol,receipt))
    if receipt['arm']=='A':checks['native_A_scheduler_policy']=common['resources']['scheduler_type'].endswith('.SLOScheduler')and common['resources']['policy_type'].endswith('.SLOPolicy')
    else:
        checks['declared_C_adapter']=mixed is not None and mixed['variant']==protocol['mixed_variant']
        checks['no_outside_v45_selection']=common['stats']['outside_selected_executed_groups']==0
    for v in classes.values():
        for k in ['client_ttft_text','server_ttft','tbt','e2el']:
            vals=v[k];v[k]={'count':len(vals),'mean':statistics.fmean(vals)if vals else None,'p50':percentile(vals,.5),'p95':percentile(vals,.95),'max':max(vals)if vals else None}
        v['attained_job_rate']=v['attained_jobs']/v['jobs'];v['gated_weighted_goodput']=v['gated_weighted_tokens']/duration;v['official_formula_weighted_goodput']=v['official_formula_weighted_tokens']/duration
    result={'schema':'stage19-calibration-arm/v1','arm':receipt['arm'],'attempt':attempt_name,'gate':'PASS'if all(checks.values())else'FAIL','checks':checks,
            'metrics':{'main_duration':duration,'main_root_jobs':len(main),'main_requests':sum(v['requests']for v in classes.values()),
                       'main_output_tokens':sum(v['actual_output_tokens']for v in classes.values()),'output_tokens_per_second':sum(v['actual_output_tokens']for v in classes.values())/duration,
                       'max_arrival_error':arr,'compound_attained':classes['2']['attained_jobs'],'compound_total':classes['2']['jobs']},
            'classes':classes,'chains':chains,'mixed_runtime':mixed,'execution_stats':common['stats'],'scheduler_stats':read(attempt/'server_audit_summary.json')['scheduler_stats'],
            'request_signatures':signatures,'scope':protocol['comparability'],'bad_inputs':bad,'raw_token_errors':raw_bad}
    (stage/'receipts'/('RUN_GATE_'+attempt_name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    stage=Path(sys.argv[2]);r=analyze(stage,sys.argv[3]);print(json.dumps({k:r[k]for k in ['arm','gate','checks','metrics','classes','mixed_runtime']},indent=2));raise SystemExit(0 if r['gate']=='PASS'else 1)
