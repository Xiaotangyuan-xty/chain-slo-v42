"""Read-only endpoint recomputation and three-round trace reduction."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket,statistics,subprocess
from pathlib import Path

ROOT=Path('/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1')
MATRIX=ROOT/'stage15_stable_chain_v45_parallel_20261101_v2'
spec=json.loads((MATRIX/'MATRIX.json').read_text())
out={'created_utc':datetime.now(timezone.utc).isoformat(),'rounds':[],'errors':[]}
winsets=defaultdict(list);sources={}
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def pct(xs,p):
    xs=sorted(xs);i=(len(xs)-1)*p;a=int(i)
    return xs[a]+(xs[min(a+1,len(xs)-1)]-xs[a])*(i-a)

for row in spec['rounds']:
    stage=MATRIX/row['directory'];protocol=read(stage/'PROTOCOL.json')
    comparison=read(stage/'receipts/AC_COMPARISON.json');audit=read(stage/'results/FINAL_AUDIT.json')
    q=read(stage/'receipts/CPU_QUALIFICATION.json')
    sources.update({ROOT/n:h for n,h in q['base_files'].items()})
    sources.update({stage/n:h for n,h in q['stage_files'].items()})
    result={'round':row['round'],'A_gpu':row['A_gpu'],'C_gpu':row['C_gpu'],'arms':{},
            'formal_comparison_valid':comparison['comparison_valid'],'formal_audit_gate':audit['gate']}
    if not comparison['comparison_valid'] or audit['gate']!='PASS':out['errors'].append(['formal_gate',row['round']])
    for arm in ['A','C']:
        attempt=stage/'attempts'/protocol['attempts'][arm]
        calls=read(attempt/'client_calls.json');main=[c for c in calls if c['phase']=='main']
        flat=[r for c in calls for s in c['stages']for r in s['requests']]
        if len(main)!=200 or len(flat)!=1204 or {c['collection_id']for c in main}!=set(range(200)):
            out['errors'].append(['completeness',row['round'],arm])
        origin=statistics.median(c['client_chain_start_monotonic']-c['arrival_offset_seconds']for c in main)
        rows=[]
        for c in main:
            cid=c['collection_id'];n=len(c['stages'])
            jct=c['client_chain_end_monotonic']-c['client_chain_start_monotonic']
            slos={r['received_slo'][2]for s in c['stages']for r in s['requests']}
            if len(slos)!=1 or n!=protocol['stage_counts'][str(cid)]:out['errors'].append(['structure_or_slo',row['round'],arm,cid])
            budget=next(iter(slos))*n
            success=c['success']and all(s['success']and all(r['success']for r in s['requests'])for s in c['stages'])
            if not success:out['errors'].append(['failed_chain',row['round'],arm,cid])
            rows.append({'chain':cid,'stages':n,'jct':jct,'budget':budget,'attained':success and jct<=budget,
                         'strict':success and jct<=20*n,'requests':sum(len(s['requests'])for s in c['stages']),
                         'completion':c['client_chain_end_monotonic']-origin})
        wins=[r for r in rows if r['attained']];ids={r['chain']for r in wins};winsets[arm].append(ids)
        duration=max(r['completion']for r in rows)
        metrics={'official_scaled_attained':len(wins),'strict_attained':sum(r['strict']for r in rows),
                 'mean_jct':statistics.fmean(r['jct']for r in rows),
                 'jct_p90':pct([r['jct']for r in rows],.9),'jct_p95':pct([r['jct']for r in rows],.95),
                 'episode_duration':duration,'request_goodput':sum(r['requests']for r in wins)/duration}
        diffs={k:[v,comparison['metrics'][arm][k]]for k,v in metrics.items()if abs(v-comparison['metrics'][arm][k])>1e-7}
        if diffs:out['errors'].append(['metric_differences',row['round'],arm,diffs])
        arrival=max(abs(c['arrival_offset_seconds']-protocol['target_arrivals'][c['collection_id']])for c in main)
        if arrival>.5:out['errors'].append(['arrival_gate',row['round'],arm,arrival])
        raw_bad=[]
        for name in ['client_calls.json','engine_token_ids.jsonl']:
            if sha(attempt/name)!=audit['raw_trace_inventory'][arm][name]['sha256']:raw_bad.append(name)
        if raw_bad:out['errors'].append(['raw_file_changed_since_audit',row['round'],arm,raw_bad])
        result['arms'][arm]={'metrics':metrics,'metric_differences':diffs,'main_chains':len(main),'requests_including_warmup':len(flat),
            'max_arrival_error':arrival,'attained_ids':sorted(ids),
            'by_stages':{str(n):{'total':sum(r['stages']==n for r in rows),'attained':sum(r['attained']for r in rows if r['stages']==n)}for n in [4,6]},
            'by_deadline':{str(b):{'total':sum(r['budget']==b for r in rows),'attained':sum(r['attained']for r in rows if r['budget']==b)}for b in sorted({r['budget']for r in rows})},
            'win_margin_seconds':{'min':min(r['budget']-r['jct']for r in wins),'median':statistics.median(r['budget']-r['jct']for r in wins),
                                  'within_1s':sum(r['budget']-r['jct']<=1 for r in wins),'within_5s':sum(r['budget']-r['jct']<=5 for r in wins)},
            'attained_completions_by_time':{str(t):sum(r['completion']<=t for r in wins)for t in [120,200,300,480,600,700]},
            'last_attained_completion_seconds':max(r['completion']for r in wins),
            'scheduler_stats':comparison['arms'][arm]['scheduler_stats']}
        if arm=='C':
            counts=Counter();free=[];max_excess=0
            for line in(attempt/'reservation_events.jsonl').open():
                e=json.loads(line);t=e['monotonic']-origin
                if t<0 or e['mode']!='completion_density_horizon':continue
                counts['active_snapshots']+=1
                counts['extra_lease_hold_snapshots']+=bool(e['lease_holds'])
                counts['estimated_over_capacity']+=e['reserved_blocks']>e['capacity']
                max_excess=max(max_excess,e['reserved_blocks']-e['capacity'])
                if t<200:free.append(e['free_gpu'])
            result['C_admission_trace']={'counts':dict(counts),'max_estimated_block_excess':max_excess,
                'snapshot_mean_free_gpu_first200':statistics.fmean(free),
                'outside_selected_execution_calls':audit['C_reservation_alignment']['executor_calls_outside_selection'],
                'interpretation':'Horizon estimates can exceed capacity between planning frames; this is not physical over-allocation or a causal ablation.'}
    a,c=(set(result['arms'][x]['attained_ids'])for x in ['A','C'])
    result['overlap']={'both':len(a&c),'A_only':len(a-c),'C_only':len(c-a),'neither':200-len(a|c)}
    result['net_C_gain']=len(c)-len(a)
    out['rounds'].append(result)

bad_sources=[str(p)for p,h in sources.items()if not p.is_file()or sha(p)!=h]
out['qualified_source_audit']={'unique_files_checked':len(sources),'mismatches':bad_sources}
if bad_sources:out['errors'].append(['sources',bad_sources])
out['aggregate']={}
for arm in ['A','C']:
    metrics=[r['arms'][arm]['metrics']for r in out['rounds']]
    out['aggregate'][arm]={'attainment_counts':[m['official_scaled_attained']for m in metrics],
        'mean_attained':statistics.fmean(m['official_scaled_attained']for m in metrics),
        'mean_attainment_rate':sum(m['official_scaled_attained']for m in metrics)/600,
        'mean_jct':statistics.fmean(m['mean_jct']for m in metrics),
        'mean_episode_duration':statistics.fmean(m['episode_duration']for m in metrics),
        'all_three_attained_ids':sorted(set.intersection(*winsets[arm])),
        'ever_attained_unique_ids':sorted(set.union(*winsets[arm])),
        'swap_out_blocks':sum(r['arms'][arm]['scheduler_stats']['swap_out_block_groups']for r in out['rounds'])}
out['all_three_valid_and_C_wins']=not out['errors']and all(r['net_C_gain']>0 for r in out['rounds'])
out['gpu_state_now']={}
for gpu in [0,1]:
    state=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
    out['gpu_state_now'][str(gpu)]={'state':state,'compute_processes':processes}
out['ports_free_now']={}
for port in [18080,18081,65433,65434]:
    try:
        with socket.socket()as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))
        out['ports_free_now'][str(port)]=True
    except OSError:out['ports_free_now'][str(port)]=False
out['scope']='Three repeated observations of the same200 DEV-informed chains/arrivalseed42, not600 unique chains. GPU assignment crossed once, same host, fixed-length replay. No heldout/generalization/semantic-equivalence or v45-over-v44 causal claim.'
out['gate']='PASS' if not out['errors']else 'REVIEW'
print(json.dumps(out,indent=2,ensure_ascii=False))
