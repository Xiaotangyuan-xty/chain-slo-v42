"""Small common-random-number resource rollout. Receives no replay objects."""
from dataclasses import dataclass, replace
import copy, hashlib, math, random, statistics, time
from contracts import CacheLedger, CallView, Intent

def guard_target(calls, current, q, continuous):
    """Same work-conserving guard for candidate and every beta arm.

    Deadline-like thresholds choose the oldest request; they are not promises
    that arbitrary overloaded arrivals can meet a maximum waiting time.
    """
    if not calls: return None
    floor=min(c.virtual_service for c in calls)
    queued=[c for c in calls if c.key != current and c.virtual_service<=floor+2*q+1e-9]
    overdue=[c for c in queued if c.ready_wait >= 8*q]
    if overdue:return max(overdue,key=lambda c:(c.ready_wait,c.key)).key
    if queued and current is not None and continuous >= 8*q:
        return min(queued,key=lambda c:(c.level,-c.ready_wait,c.key)).key
    return None

def legal_targets(calls, q):
    floor=min(c.virtual_service for c in calls)
    return {c.key for c in calls if c.virtual_service <= floor+2*q+1e-9}

@dataclass
class ProxyCall:
    view: CallView
    remaining: int
    record: tuple
    ready_at: float = 0.
    successor_allowed: bool = True

def reference(calls, now, current):
    ready=[c for c in calls if c.ready_at <= now]
    if not ready:return None
    # Retain running call until its ordinary quantum is exhausted, unless a
    # queued request has a strictly higher priority.
    best=min(ready,key=lambda c:(c.view.level,-c.view.ready_wait,c.view.key))
    old=next((c for c in ready if c.view.key==current),None)
    if old is not None and old.view.level <= best.view.level:return old
    return best

def rollout(view, predictor, profile, action, q, sample, deadline):
    ledger=CacheLedger(view.cache.gpu_capacity,view.cache.host_capacity)
    ledger.entries={e.key:e for e in view.cache.entries}
    ledger.clock=max([0]+[max(e.gpu_touch,e.host_touch) for e in view.cache.entries])
    calls=[]
    for v in view.calls:
        seed=int(hashlib.sha256((v.key+':'+str(sample)).encode()).hexdigest()[:16],16)
        rng=random.Random(seed)
        rem,record=predictor.remaining(v.prompt_tokens,v.ordinal,v.emitted,rng)
        calls.append(ProxyCall(v,rem,record))
    now=0.;cost=0.;current=None;continuous=0.;first=True;events=0
    # Drain all sampled present calls and one predicted successor each, rather
    # than making restoration after the first h disappear at a cutoff.
    while calls:
        events+=1
        if events>4096 or time.perf_counter()>deadline:raise TimeoutError('rollout budget')
        ready=[c for c in calls if c.ready_at <= now]
        if not ready:now=min(c.ready_at for c in calls);continue
        forced=guard_target(tuple(c.view for c in ready),current,q,continuous)
        legal=legal_targets(tuple(c.view for c in ready),q)
        if first:
            chosen=next(c for c in ready if c.view.key==action.key)
            length=action.horizon
        else:
            chosen=reference(calls,now,current);length=q
        if forced:chosen=next(c for c in ready if c.view.key==forced)
        elif chosen.view.key not in legal:
            chosen=min((c for c in ready if c.view.key in legal),key=lambda c:(c.view.level,-c.view.ready_wait,c.view.key))
        if chosen.view.key != current:continuous=0.
        current=chosen.view.key;v=chosen.view
        length=min(length,8*q-continuous)
        if length <= 1e-9:current=None;continuous=0.;continue
        if not first and v.quantum_used >= q*2**v.level:
            v=replace(v,level=min(3,v.level+1),quantum_used=0.)
        # Restore only already-produced KV; initial new prompt is necessary work.
        previous=v.prompt_tokens+max(0,v.emitted-1) if v.emitted else v.produced_prefix
        target_prefix=v.prompt_tokens+max(0,v.emitted-1)
        count=min(chosen.remaining,max(1,int(length/profile.decode_seconds)))
        required=v.prompt_tokens+v.emitted+count-1
        for event,key,tokens in ledger.plan(v.program,required):
            if event=='swap_out':cost+=tokens*profile.copy_seconds_per_token
            ledger.commit(event,key,tokens)
        entry=ledger.view().get(v.program)
        restore=profile.recovery(entry,previous)
        cost+=restore
        prompt_cost=profile.prefill(max(0,v.prompt_tokens-max(entry.gpu,entry.host)),
                                    max(entry.gpu,entry.host)) if not v.emitted else 0.
        elapsed=restore+prompt_cost+count*profile.decode_seconds
        ledger.commit('gpu_produced',v.program,required)
        for other in ready:
            if other is not chosen:other.view=replace(other.view,ready_wait=other.view.ready_wait+elapsed)
        chosen.remaining-=count;now+=elapsed;continuous+=elapsed
        chosen.view=replace(v,emitted=v.emitted+count,produced_prefix=required,
                            quantum_used=v.quantum_used+elapsed,ready_wait=0.,
                            virtual_service=v.virtual_service+elapsed)
        first=False
        if chosen.remaining==0:
            calls.remove(chosen);current=None;continuous=0.
            out,continues,next_prompt,away=chosen.record
            if chosen.successor_allowed and continues:
                key=v.key+'~predicted'
                rng=random.Random(int(hashlib.sha256((key+str(sample)).encode()).hexdigest()[:16],16))
                rem,rec=predictor.remaining(next_prompt,v.ordinal+1,0,rng)
                pre=profile.prefill(max(0,next_prompt-required),min(next_prompt,required))
                level=next((i for i in range(4) if q*2**i>=pre),3)
                nv=replace(chosen.view,key=key,ordinal=v.ordinal+1,prompt_tokens=next_prompt,
                           emitted=0,level=level,quantum_used=0.,ready_wait=0.,running=False)
                calls.append(ProxyCall(nv,rem,rec,now+away,False))
            else:ledger.commit('release',v.program)
    return cost

def decide(view,predictor,profile,pi0,q,hcap,continuous,timeout=.01):
    start=time.perf_counter();deadline=start+timeout
    current=next((c.key for c in view.calls if c.running),None)
    forced=guard_target(view.calls,current,q,continuous)
    if forced:return Intent(forced,min(q,hcap),'common_guard'),{'fallback':'guard'}
    legal=legal_targets(view.calls,q)
    if pi0 not in legal:
        pi0=min((c for c in view.calls if c.key in legal),key=lambda c:(c.level,-c.ready_wait,c.key)).key
    fallback=Intent(pi0,min(q,hcap),'pi0')
    if len(view.calls)>32:return fallback,{'fallback':'online_state_over_32_call_rollout_budget'}
    try:
        keys=[pi0]
        if current in legal and current not in keys:keys.append(current)
        oldest=max(view.calls,key=lambda c:(c.ready_wait,c.key)).key
        if oldest in legal and oldest not in keys:keys.append(oldest)
        actions=[]
        for key in keys[:3]:
            v=next(c for c in view.calls if c.key==key)
            rem90=predictor.p90(v.prompt_tokens,v.ordinal,v.emitted)
            length=min(hcap,max(profile.decode_seconds,rem90*profile.decode_seconds))
            actions.append(Intent(key,length,'cost_rollout'))
        # Reference action always included with exactly its original duration.
        actions=[fallback]+[a for a in actions if (a.key,a.horizon)!=(fallback.key,fallback.horizon)]
        costs=[[rollout(view,predictor,profile,a,q,s,deadline) for s in range(4)] for a in actions]
        means=[statistics.fmean(x) for x in costs]
        winner=min(range(len(actions)),key=lambda i:(means[i],i))
        diffs=[a-b for a,b in zip(costs[0],costs[winner])]
        gain=statistics.fmean(diffs)
        uncertainty=3*statistics.stdev(diffs)/2+2*profile.relative_error*max(means)
        if time.perf_counter()>deadline:raise TimeoutError('decision timeout')
        accepted=winner!=0 and gain>uncertainty
        return (actions[winner] if accepted else fallback),{
            'fallback':None if accepted else 'no_clear_model_difference',
            'means':means,'paired_estimated_gain':gain,'uncertainty':uncertainty,
            'decision_seconds':time.perf_counter()-start,'accepted':accepted}
    except (ValueError,TimeoutError,RuntimeError) as e:
        return fallback,{'fallback':str(e),'decision_seconds':time.perf_counter()-start}
