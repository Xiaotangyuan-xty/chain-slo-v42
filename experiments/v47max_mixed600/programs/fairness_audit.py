"""Observe actual native routing and correction calls; returned budgets unchanged."""
import atexit,functools,inspect,json,math,threading
from pathlib import Path

VERSION='stage74_native_implicit_tot_budget_audit_v3'

def install(server,output,protocol):
    from jitserve.request_analyzer.graph_context import GraphMatchContext
    import native_tot_adapter,baseline_budget
    output=Path(output);method=protocol['candidate_version'];tls=threading.local()
    stream=(output/'FAIRNESS_EVENTS.jsonl').open('x');counts={}
    def emit(kind,**data):
        counts[kind]=counts.get(kind,0)+1
        stream.write(json.dumps({'kind':kind,**data},allow_nan=False)+'\n');stream.flush()
    for name,module,attribute in [('tot',native_tot_adapter,'tot_cumulative_ratio'),('dr',baseline_budget,'cumulative_ratio')]:
        prior=getattr(module,attribute)
        def marker(*args,_prior=prior,_name=name,**kw):
            if getattr(tls,'request',None)is not None:
                tls.calls[_name]=tls.calls.get(_name,0)+1
            return _prior(*args,**kw)
        setattr(module,attribute,functools.wraps(prior)(marker))
    previous_ratio=GraphMatchContext._online_ratio
    native_budget_function=inspect.getsourcefile(previous_ratio).endswith('/graph_context.py')
    @functools.wraps(previous_ratio)
    def ratio(self,num_stages,requests_per_stage,prompt,collection_id,is_deepresearch,stage_id):
        active=getattr(tls,'request',None)
        if active is None:return previous_ratio(self,num_stages,requests_per_stage,prompt,collection_id,is_deepresearch,stage_id)
        tls.calls={}
        structure=self.collection_graph_unfinished_dict.get(collection_id)if not is_deepresearch else None
        actual_stage=getattr(structure,'stage_finished',stage_id)
        result=previous_ratio(self,num_stages,requests_per_stage,prompt,collection_id,is_deepresearch,stage_id)
        emit('budget',**active,is_deepresearch=is_deepresearch,declared_stages=num_stages,
             actual_stage=actual_stage,ratio=result,helper_calls=dict(tls.calls),
             native_pattern=getattr(structure,'request_per_stage',None),
             use_all_node=getattr(structure,'use_all_node',None))
        return result
    GraphMatchContext._online_ratio=ratio
    previous_route=server._handle_collective_request
    @functools.wraps(previous_route)
    def route(info,data,prompt):
        active={'phase':'warmup'if info.client_id==900000 else'main','collection_id':info.collection_id,
                'stage_id':info.stage_id,'request_id':info.request_id}
        wire=data.get('requests_per_stage');depth=data.get('num_stages')
        tls.request=active
        try:result=previous_route(info,data,prompt)
        finally:tls.request=None
        emit('route',**active,is_deepresearch=result[0],wire_widths=wire,declared_stages=depth,
             annotated_widths=list(getattr(info,'chain_stage_widths',[])or[]),
             future_cap=getattr(info,'chain_future_max_tokens',None),received_slo=list(info.slo_constraint))
        return result
    server._handle_collective_request=route
    receipt={'version':VERSION,'method':method,'native_budget_preserved':native_budget_function,
             'Astar_DR_and_ToT_expected':method=='Astar','return_values_modified_by_observer':False,
             'future_truth_used':False,'same_audit_all_methods':True}
    def write():
        (output/'FAIRNESS_RUNTIME.json').write_text(json.dumps({**receipt,'counts':dict(counts)},indent=2)+'\n')
    def close():stream.close();write()
    write();atexit.register(close)
    return receipt

def validate_rows(rows,work,method,phase):
    jobs=[j for j in work[phase]if j['request_type']==2]
    expected={(j['collection_id'],sid,r['request_id']):(j,sid)for j in jobs for sid,stage in enumerate(j['stages'])for r in stage}
    subset=[r for r in rows if r['phase']==phase]
    budgets=[r for r in subset if r['kind']=='budget'];routes=[r for r in subset if r['kind']=='route']
    key=lambda r:(r['collection_id'],r['stage_id'],r['request_id'])
    coverage=len(budgets)==len(routes)==len(expected)and {key(r)for r in budgets}==set(expected)=={key(r)for r in routes}
    checks={'fair_'+phase+'_complete_budget_and_route_coverage':coverage};bad=[]
    for r in routes:
        if key(r)not in expected:bad.append('unknown route');continue
        job,sid=expected[key(r)];tot=job['application']=='tot_style'
        if r['received_slo']!=job['received_slo']or r['declared_stages']!=len(job['stages']):bad.append('SLO/depth changed')
        if tot:
            if r['is_deepresearch']or r['wire_widths']is not None or r['annotated_widths']!=[3,1,3,1]or r['future_cap']!=256:bad.append('native implicit ToT metadata/route')
        elif not r['is_deepresearch']or r['wire_widths']!=list(map(len,job['stages'])):bad.append('DR route')
    for r in budgets:
        if key(r)not in expected:bad.append('unknown budget');continue
        job,sid=expected[key(r)];tot=job['application']=='tot_style'
        if r['is_deepresearch']==tot or r['declared_stages']!=len(job['stages']):bad.append('budget graph kind/depth')
        if not math.isfinite(r['ratio'])or not 0<=r['ratio']<=1:bad.append('nonfinite budget')
        if tot and (r['native_pattern']!=[3,1,3,1]or r['actual_stage']!=sid or r['use_all_node']is not False):bad.append('native stage progress/pattern/aggregation')
        expected_calls={('tot'if tot else'dr'):1}if method=='Astar'else{}
        if r['helper_calls']!=expected_calls:bad.append('correction helper not actually executed or changed native arm')
        if method=='Astar'and r['ratio']+1e-12<(sid+1)/len(job['stages']):bad.append('current-stage-inclusive cumulative budget floor')
        if method=='Astar'and sid==len(job['stages'])-1 and abs(r['ratio']-1.)>1e-12:bad.append('last-stage cumulative budget')
    checks['fair_'+phase+'_native_routes_metadata_and_effective_budgets']=not bad
    checks['fair_'+phase+'_ToT_coverage_matches_frozen_workload']=sum(not r['is_deepresearch'] for r in routes)==sum(len(s) for j in jobs if j['application']=='tot_style' for s in j['stages'])
    checks['fair_'+phase+'_DR_coverage_matches_frozen_workload']=sum(r['is_deepresearch'] for r in routes)==sum(len(s) for j in jobs if j['application']=='dr' for s in j['stages'])
    return checks

def checks(output,work,method,phase):
    output=Path(output);rows=[json.loads(l)for l in(output/'FAIRNESS_EVENTS.jsonl').read_text().splitlines()]
    result=validate_rows(rows,work,method,phase);receipt=json.loads((output/'FAIRNESS_RUNTIME.json').read_text())
    result['fair_'+phase+'_observer_and_native_A_budget_identity']=receipt['version']==VERSION and receipt['method']==method and receipt['return_values_modified_by_observer']is False and receipt['native_budget_preserved']==(method!='Astar')
    result['fair_'+phase+'_Astar_ToT_adapter_presence']=(output/'ASTAR_TOT_BUDGET_ADAPTER.json').exists()==(method=='Astar')
    return result
