"""Wall-clock native closed loop; hidden replay fields stay behind make_view."""
from dataclasses import dataclass,replace
import collections,heapq,math,random,statistics,time
from contracts import CacheEntry,CacheView,CallView,quantile
from v2_contracts import View,GoalView,ObservedCall
from v2_reference import forecasts
from pool_policies import route,jit_batch,features
from pool_reference import project
from official_adapter import OfficialGMAX

@dataclass(frozen=True)
class PoolCache(CacheView):
    owners:tuple=()
    def get(self,key):
        i=dict(self.owners).get(key,0)
        e=next((x for x in self.entries if x.key==f'{i}:{key}'),CacheEntry(key))
        return replace(e,key=key)

def public_view(now,metadata,states,histories,calls,pool,owners,tools,prefixes,virtual):
    entries=tuple(replace(e,key=f'{i}:{e.key}') for i,b in enumerate(pool.instances) for e in b.ledger.view().entries)
    cache=PoolCache(16384,16384,entries,tuple(sorted(owners.items())))
    goals=[]
    for pid,state in states.items():
        if state=='DONE':continue
        p=metadata[pid];c=calls.get(pid);cv=None
        if state=='READY':
            cv=CallView(c['key'],pid,c['ordinal'],len(c['prompt']),len(c['emitted']),prefixes.get(pid,0),
                c['level'],c['quantum'],now-c['ready_since'],c['latency_service'],virtual[pid],False,c['wait'])
        goals.append(GoalView(pid,p['arrival'],p['deadline'],state,tuple(histories.get(pid,[])),cv,
            tools.get(pid) if state=='AWAY' else None,now-tools[pid] if state=='AWAY' else None,prefixes.get(pid,0)))
    return View(now,tuple(goals),cache)

def candidate_action(view,instance,eligible,pred,perf,owners):
    start=time.perf_counter();q=64*perf.decode_seconds
    cs=[c for c in view.calls if c.key in eligible]
    pi0=min(cs,key=lambda c:(c.level,-c.ready_wait,c.key)).key;h0=4*perf.decode_seconds
    audit={'pi0':pi0,'actions':[],'accepted':False,'fallback':None}
    try:
        js=forecasts(view,pred,perf)
        bypid={g.program:g for g in view.goals}
        for j in js:
            g=bypid[j['program']];j['instance']=owners[g.program]
            j['context']=g.ready.prompt_tokens+g.ready.emitted if g.ready else g.history[-1].prompt_tokens+g.history[-1].outputs
        bykey={j['key']:j for j in js};acts=[(pi0,h0)]
        for c in cs:
            for h in [h0,max(perf.decode_seconds,min(q,bykey[c.key]['current']))]:
                if (c.key,h) not in acts:acts.append((c.key,h))
        for k,h in acts:
            score=project(js,(k,h),view.now,perf,instance,start+.02)
            audit['actions'].append({'key':k,'h':h,**score})
        winner=max(range(len(acts)),key=lambda i:((audit['actions'][i]['good'],-audit['actions'][i]['mean']),-i))
        if winner==0:audit['fallback']='no_model_gain'
        audit['accepted']=winner!=0;audit['seconds']=time.perf_counter()-start
        return acts[winner],audit
    except (ValueError,RuntimeError,TimeoutError) as e:
        audit.update(fallback=str(e),seconds=time.perf_counter()-start)
        return (pi0,h0),audit

def run_episode(inp,arm,pool,perf,pred,qrf,graphs,median_output,router=None):
    pool.reset();start=time.perf_counter();clock=lambda:time.perf_counter()-start
    if hasattr(perf,'reset_online'):perf.reset_online()
    official=OfficialGMAX(qrf,graphs,clock) if arm=='JITServe_official_GMAX_core' else None
    ps={p['id']:p for p in inp['programs']};metadata={p['id']:{'arrival':p['arrival'],'deadline':p['deadline']} for p in inp['programs']}
    pending=[];seq=0
    for p in inp['programs']:
        heapq.heappush(pending,(p['arrival'],seq,p['id'],0));seq+=1
    states={};calls={};all_calls=[];hist={};context={};prefixes={};owners={};tools={};done={}
    virtual=collections.defaultdict(float);age=collections.defaultdict(int);previous=[[],[]];frames=[50,50]
    leases=[None,None];remaining=[0.,0.];steps=[0,0];trace=[];routes=[];batches=[];policy_seconds=0.;physical_busy=0.;timing=collections.Counter()
    def current_visible():return [{'instance':c['instance'],'prompt':len(c['prompt']),'emitted':len(c['emitted'])} for c in calls.values()]
    def admit(release,pid,ordinal):
        spec=ps[pid]['calls'][ordinal];old=context.get(pid,[]);rng=random.Random(ps[pid]['token_seed']+ordinal)
        prompt=old+[rng.randrange(100,10000) for _ in range(spec['prompt_tokens']-len(old))]
        if len(prompt)!=spec['prompt_tokens']:raise RuntimeError('observed prefix exceeds new prompt')
        t0=time.perf_counter()
        estimate=None
        if official is not None:
            cv=CallView(pid+':'+str(ordinal),pid,ordinal,len(prompt),0,prefixes.get(pid,0),0,0.,0.,0.,virtual[pid],False)
            g=GoalView(pid,metadata[pid]['arrival'],metadata[pid]['deadline'],'READY',tuple(hist.get(pid,[])),cv)
            estimate=official.prepare(g,release).request_info.output_len
        elif arm=='JITServe_GMAX_port':
            cv=CallView(pid+':'+str(ordinal),pid,ordinal,len(prompt),0,prefixes.get(pid,0),0,0.,0.,0.,virtual[pid],False)
            g=GoalView(pid,metadata[pid]['arrival'],metadata[pid]['deadline'],'READY',tuple(hist.get(pid,[])),cv)
            estimate=float(qrf.predict([features(g)],quantiles=.95)[0])
        route_begin=time.perf_counter()
        if router is None:
            route_arm='JITServe_GMAX_port' if official is not None else arm
            route_estimate=estimate
        else:
            from study_design import router_parameters
            route_arm,route_estimate=router_parameters(router)
        inst,audit=route(len(prompt),pid,current_visible(),[b.ledger.view() for b in pool.instances],perf,route_arm,median_output,route_estimate)
        if router is not None:audit['registered_common_router']=router; audit['common_router_length_prior']=route_estimate
        timing['routing_seconds']+=time.perf_counter()-route_begin
        if inst is None:raise RuntimeError('unexpected capacity reject; preserve full denominator')
        now=clock();owners[pid]=inst
        if official is not None:official.assign(pid+':'+str(ordinal),inst)
        pool.before_call(inst,pid,prompt,prefixes.get(pid,0))
        c={'key':pid+':'+str(ordinal),'program':pid,'ordinal':ordinal,'prompt':prompt,'emitted':[],
            'instance':inst,'release':release,'admitted':now,'ready_since':now,'level':0,'quantum':0.,
            'latency_service':0.,'GPU_share':0.,'wait':0.,'token_times':[],'spec':spec}
        calls[pid]=c;all_calls.append(c);states[pid]='READY';tools.pop(pid,None)
        routes.append({'time':now,'program':pid,'ordinal':ordinal,'instance':inst,'prompt':len(prompt),'audit':audit})
        return time.perf_counter()-t0
    while pending or calls:
        if clock()>max(p['arrival'] for p in inp['programs'])+900:raise TimeoutError('episode limit; never drop unfinished chains')
        did_work=False
        for inst in range(2):
            now=clock()
            while pending and pending[0][0]<=now:
                release,_,pid,ordinal=heapq.heappop(pending);policy_seconds+=admit(release,pid,ordinal);now=clock()
            own=[c for c in calls.values() if c['instance']==inst]
            if not own:continue
            did_work=True;bykey={c['key']:c for c in own};t0=time.perf_counter()
            view_begin=time.perf_counter()
            view=public_view(now,metadata,states,hist,calls,pool,owners,tools,prefixes,virtual)
            timing['public_view_seconds']+=time.perf_counter()-view_begin
            oldest=max(own,key=lambda c:now-c['ready_since'])
            q=64*perf.decode_seconds;forced=oldest['key'] if now-oldest['ready_since']>8*q else None
            floor=min(virtual[c['program']] for c in own)
            eligible=[c for c in own if virtual[c['program']]<=floor+2*q]
            legal={c['key'] for c in eligible};audit={};anchor=None
            if arm=='PackServe_port':
                held=[k for k in previous[inst] if k in legal]
                chosen=held+[c['key'] for c in sorted(eligible,key=lambda c:(c['release'],c['key'])) if c['key'] not in held]
                chosen=chosen[:4]
            elif official is not None:
                chosen,audit=official.choose(view,inst,legal,previous[inst])
            elif arm=='JITServe_GMAX_port':
                held=[k for k in previous[inst] if k in legal]
                if frames[inst]>=50 or not held:
                    chosen,audit=jit_batch(view.goals,now,legal,qrf,graphs,perf,age,held,pool.instances[inst].ledger.view());frames[inst]=0
                    for c in eligible:age[c['key']]+=1
                else:
                    chosen=(held+[c['key'] for c in sorted(eligible,key=lambda c:(c['release'],c['key'])) if c['key'] not in held])[:4]
            else:
                anchor=leases[inst]
                if anchor not in legal or remaining[inst]<=0:
                    if arm=='v3_probability_tie':
                        from probability_tie import choose
                        (anchor,h),audit=choose(view,inst,legal,pred,perf,owners)
                    elif arm=='v2_fixed_h4':
                        from h4_policy import fixed_h4_action
                        (anchor,h),audit=fixed_h4_action(view,inst,legal,pred,perf,owners)
                    else:
                        (anchor,h),audit=candidate_action(view,inst,legal,pred,perf,owners)
                    timing['candidate_selector_seconds']+=audit.get('seconds',0.)
                    leases[inst]=anchor;remaining[inst]=h
                chosen=[anchor]+[c['key'] for c in sorted(eligible,key=lambda c:(c['level'],-(now-c['ready_since']),c['key'])) if c['key']!=anchor]
                chosen=chosen[:4]
            if forced and forced not in chosen:chosen=([forced]+chosen)[:4]
            if not chosen or len(chosen)!=len(set(chosen)) or not set(chosen)<=legal|({forced} if forced else set()):raise RuntimeError('illegal/non-work-conserving batch')
            if audit or steps[inst]%4==0:
                trace.append({'time':now,'instance':inst,'view':view.to_dict(),'owners':dict(owners),'selected':chosen,
                    'legal':sorted(legal),'forced':forced,'audit':audit,'arm':arm,'performance_ema':dict(getattr(perf,'factors',{}))})
            policy_seconds+=time.perf_counter()-t0
            selected=[bykey[k] for k in chosen];begin=clock()
            for c in selected:c['wait']+=begin-c['ready_since']
            before_feedback=len(getattr(pool.instances[inst],'stats',{}).get('native_decode_batches',[]))
            logits=pool.instances[inst].step_batch([(c['program'],c['prompt'],c['emitted']) for c in selected])
            end=clock();span=end-begin;physical_busy+=span
            feedback=getattr(pool.instances[inst],'stats',{}).get('native_decode_batches',[])[before_feedback:]
            model_feedback=[perf.observe(x['B'],x['S'],x['wall']) for x in feedback] if hasattr(perf,'observe') else []
            batches.append({'begin':begin,'end':end,'instance':inst,'keys':chosen,'B':len(chosen),
                'prefill_keys':[c['key'] for c in selected if not c['emitted']],'model_feedback':model_feedback})
            for c in selected:
                c['latency_service']+=span;c['GPU_share']+=span/len(selected);virtual[c['program']]+=span/len(selected)
                c['ready_since']=end;c['quantum']+=perf.decode_seconds
                if c['quantum']>=q*2**c['level']:c['level']=min(3,c['level']+1);c['quantum']=0.
                token=random.Random(c['spec']['output_token_seed']+len(c['emitted'])).randrange(100,10000)
                c['emitted'].append(token);c['token_times'].append(end)
                if leases[inst]==c['key']:remaining[inst]-=perf.decode_seconds
                if len(c['emitted'])==c['spec']['outputs']:
                    pid=c['program'];ordinal=c['ordinal'];c['finish']=end
                    if official is not None:
                        ts=time.perf_counter();official.finish(c['key']);policy_seconds+=time.perf_counter()-ts
                    hist.setdefault(pid,[]).append(ObservedCall(ordinal,len(c['prompt']),len(c['emitted']),end,c['latency_service'],c['wait']))
                    context[pid]=c['prompt']+c['emitted'];prefixes[pid]=len(context[pid])-1;calls.pop(pid)
                    if ordinal+1<len(ps[pid]['calls']):
                        tools[pid]=end;states[pid]='AWAY';heapq.heappush(pending,(end+c['spec']['away'],seq,pid,ordinal+1));seq+=1
                    else:states[pid]='DONE';done[pid]=end;pool.release(pid)
            logits=None;previous[inst]=chosen;steps[inst]+=1;frames[inst]+=1
        if not did_work and pending:time.sleep(max(0,min(.05,pending[0][0]-clock())))
    if len(done)!=len(ps):raise RuntimeError('full chain denominator')
    jobs=[];callmetrics=[]
    for p in inp['programs']:
        cs=[c for c in all_calls if c['program']==p['id']]
        service=sum(c['latency_service'] for c in cs);wait=sum(c['wait'] for c in cs)
        lag=sum(c['admitted']-c['release'] for c in cs);away=sum(c['away'] for c in p['calls']);jct=done[p['id']]-p['arrival']
        residual=jct-service-wait-lag-away
        if abs(residual)>1e-5:raise RuntimeError('per-chain latency accounting '+str(residual))
        jobs.append({'id':p['id'],'kind':p['kind'],'jct':jct,'attained':done[p['id']]<=p['deadline'],
            'finish':done[p['id']],'arrival':p['arrival'],'deadline':p['deadline'],'service':service,'wait':wait,'lag':lag,'away':away,'residual':residual,
            'outputs':sum(len(c['emitted']) for c in cs)})
    for c in all_calls:
        tpot=(c['token_times'][-1]-c['token_times'][0])/max(1,len(c['token_times'])-1)
        callmetrics.append({'program':c['program'],'ordinal':c['ordinal'],'release':c['release'],'finish':c['finish'],
            'TTFT':c['token_times'][0]-c['release'],'TPOT':tpot,'outputs':len(c['emitted']),'GPU_share':c['GPU_share']})
    costs=[b.summary() for b in pool.instances]
    if sum(c['logical_outputs'] for c in costs)!=inp['total_decode']:raise RuntimeError('logical work conservation')
    if abs(sum(c['GPU_share'] for c in all_calls)-physical_busy)>1e-5:raise RuntimeError('GPU busy double billing')
    return {'schema':'jitpack-pool-native/v1','seed':inp['seed'],'condition':inp['condition'],'arm':arm,'complete':True,
        'physical_GPU':__import__('study_gate').GPU,'jobs':jobs,'call_metrics':callmetrics,'trace':trace,'routes':routes,'batches':batches,
        'costs':costs,'cache_events':[b.events for b in pool.instances],'cache_commits':[b.ledger.events for b in pool.instances],
        'official_gmax':official.snapshot() if official is not None else None,'timing_breakdown':dict(timing),
            'policy_seconds':policy_seconds,'GPU_busy_seconds':physical_busy,'wall_seconds':clock(),
        'metrics':{'goodput':statistics.fmean(j['attained'] for j in jobs),'attained':sum(j['attained'] for j in jobs),'chains':len(jobs),
            'mean_JCT':statistics.fmean(j['jct'] for j in jobs),'P50_TPOT':quantile([c['TPOT'] for c in callmetrics],.5),
            'P90_TPOT':quantile([c['TPOT'] for c in callmetrics],.9),'mean_TTFT':statistics.fmean(c['TTFT'] for c in callmetrics)},
        'backend_busy_wall_includes_host_submission_not_sum_of_CUDA_kernels':True,'no_natural_language_quality_claim':True,'no_multi_GPU_fleet_efficiency_claim':True}
