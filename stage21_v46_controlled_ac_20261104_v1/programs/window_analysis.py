"""CPU-only window accounting; all admitted roots remain in the final cohort."""
from collections import Counter
import json
import math
from pathlib import Path
import sys

def read(p):return json.loads(p.read_text())
def key(r):return(r['phase'],r['collection_id'],r['stage_id'],r['request_id'])
def cohort_index(offset,window):
    return min(max(0,int(offset//300)),max(1,math.ceil(window/300))-1)

def credits(request,engine,kind,attained):
    assert request['server_ttft']is not None and math.isfinite(request['server_ttft'])and request['server_ttft']>=0
    assert all(math.isfinite(x)and x>=0 for x in request['server_tbt'])
    inp,out=engine['prompt_tokens'],engine['output_tokens']
    gated=inp+8*out if attained else 0
    if kind!=0:return gated,gated
    slo=request['received_slo'];ttft=request['server_ttft'];cumulative=ttft;good=0
    for n,interval in enumerate(request['server_tbt']):
        cumulative+=interval;good+=int(cumulative<=slo[0]+slo[1]*(n+1))
    return gated,(inp if ttft<=slo[0]else 0)+8*good

def analyze(stage):
    protocol=read(stage/'PROTOCOL.json');work=read(stage/'WORKLOAD.json')
    window=work['main'][-1]['arrival_offset_seconds'];result={}
    for arm,name in protocol['attempts'].items():
        directory=stage/'attempts'/name
        calls=[c for c in read(directory/'client_calls.json')if c['phase']=='main']
        gate=read(stage/'receipts'/('RUN_GATE_'+name+'.json'));assert gate['gate']=='PASS'
        attained={j['collection_id']:j['attained']for j in gate['chains']}
        origin=read(directory/'MAIN_START.json')['origin_monotonic']
        engine={key(r):r for r in read(directory/'common_audit_summary.json')['requests']}
        online={str(t):Counter()for t in [0,1,2]};full={str(t):Counter()for t in [0,1,2]}
        for job in calls:
            kind=job['request_type'];row=online[str(kind)]
            if job['client_chain_end_monotonic']-origin<=window:
                row['completed_jobs']+=1;row['attained_jobs']+=int(attained[job['collection_id']])
            for item in [r for s in job['stages']for r in s['requests']]:
                e=engine[key(item)];gated,literal=credits(item,e,kind,attained[job['collection_id']])
                values={'completed_calls':1,'output_tokens':e['output_tokens'],'input_tokens':e['prompt_tokens'],
                        'gated_weighted_tokens':gated,'official_formula_weighted_tokens':literal}
                full[str(kind)].update(values)
                if item['client_request_end_monotonic']-origin<=window:row.update(values)
        for kind in ['0','1','2']:
            assert full[kind]['gated_weighted_tokens']==gate['classes'][kind]['gated_weighted_tokens']
            assert full[kind]['official_formula_weighted_tokens']==gate['classes'][kind]['official_formula_weighted_tokens']
            for field in ['gated_weighted_tokens','official_formula_weighted_tokens']:
                online[kind][field+'_per_second']=online[kind][field]/window
        times=sorted(set([0.,window]+[float(x)for x in range(300,int(window)+1,300)]))
        backlog=[]
        for offset in times:
            live=[j for j in calls if j['client_chain_start_monotonic']<=origin+offset<j['client_chain_end_monotonic']]
            backlog.append({'offset_seconds':offset,'inflight_roots_by_type':dict(Counter(str(j['request_type'])for j in live))})
        cohorts=[]
        for bucket in range(max(1,math.ceil(window/300))):
            jobs=[j for j in calls if cohort_index(j['planned_arrival_offset_seconds'],window)==bucket]
            row={'arrival_bucket_start':bucket*300,'by_type':{}}
            for kind in [0,1,2]:
                group=[j for j in jobs if j['request_type']==kind]
                row['by_type'][str(kind)]={'admitted':len(group),'eventually_attained':sum(attained[j['collection_id']]for j in group)}
            cohorts.append(row)
        assert sum(v['admitted']for row in cohorts for v in row['by_type'].values())==len(calls)
        result[arm]={'arrival_window_seconds':window,'finite_replay_seconds':gate['metrics']['main_duration'],
                     'all_admitted_compound_chains':gate['metrics']['compound_total'],
                     'eventually_attained_compound_chains':gate['metrics']['compound_attained'],
                     'arrival_window_request_completion_credit_by_type':{k:dict(v)for k,v in online.items()},
                     'full_drained_request_credit_by_type':{k:dict(v)for k,v in full.items()},
                     'root_backlog_during_arrivals':backlog,'arrival_cohorts_300s':cohorts}
    return {'schema':'mixed-window-accounting/v1','gate':'PASS','arms':result,
            'primary_rule':'Every admitted compound chain is evaluated after full drain against its own actual received chain SLO, including late-arriving chains completing after the arrival window.',
            'window_credit_rule':'Credit a request at client completion within the arrival window. Collective credits are conditional on eventual whole-chain attainment. This is completion-based credit, not individual token delivery timestamps.',
            'cohort_rule':'300-second planned-arrival cohorts; the final endpoint belongs to the last cohort. Every admitted root is included.',
            'official_formula_rule':'Literal cumulative-TBT token formula with actual engine input/output lengths; source benchmark retokenization and exact paper workload are not claimed.'}

if __name__=='__main__':
    stage=Path(sys.argv[1]);result=analyze(stage)
    Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'gate':result['gate'],'arrival_windows':{a:r['arrival_window_seconds']for a,r in result['arms'].items()}}))
