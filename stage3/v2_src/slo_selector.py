"""Finite first action then shared EDF fluid drain; no replay truth."""
from dataclasses import dataclass
import time
from contracts import Intent
from selector import guard_target,legal_targets

@dataclass(frozen=True)
class SLOView:
    now:float
    calls:tuple
    cache:object
    # Only currently released calls' client-requested metadata.
    deadlines:tuple

def forecast(view,predictor,profile):
    deadlines=dict(view.deadlines);jobs=[]
    for c in view.calls:
        b=predictor.predict(c.prompt_tokens,c.ordinal,c.emitted)
        prior=c.prompt_tokens+max(0,c.emitted-1) if c.emitted else c.produced_prefix
        recovery=profile.recovery(view.cache.get(c.program),prior)
        jobs.append({'key':c.key,'deadline':deadlines[c.program],
            'current':b['current_upper']*profile.decode_seconds+recovery,
            'work':b['chain_upper']*profile.decode_seconds+recovery,
            'point':b['chain_mean']*profile.decode_seconds+recovery})
    return jobs

def score(jobs,key,h,now):
    """Identical EDF reference after the bounded first action for all actions.

    AWAY is folded into intrinsic chain work. This is explicitly a surrogate,
    not an execution simulator and has no claimed optimality/coverage for JCT.
    """
    rem={j['key']:j['work'] for j in jobs};finished={};elapsed=0.
    selected=next(j for j in jobs if j['key']==key)
    step=min(h,selected['current'],rem[key]);rem[key]-=step;elapsed+=step
    if rem[key]<=1e-9:finished[key]=now+elapsed
    for j in sorted(jobs,key=lambda j:(j['deadline'],j['key'])):
        k=j['key']
        if k in finished:continue
        elapsed+=rem[k];finished[k]=now+elapsed
    good=sum(finished[j['key']]<=j['deadline'] for j in jobs)
    # Full denominator is retained, including already predicted-late chains.
    mean=sum(finished.values())/len(jobs)-now
    return good,-mean

def decide(view,predictor,profile,pi0,q,continuous,diagnostic=False,timeout=.02):
    begin=time.perf_counter();calls=view.calls
    current=next((c.key for c in calls if c.running),None)
    legal=legal_targets(calls,q)
    if pi0 not in legal:pi0=min((c for c in calls if c.key in legal),key=lambda c:(c.level,-c.ready_wait,c.key)).key
    fallback=Intent(pi0,4*profile.decode_seconds,'pi0')
    audit={'accepted':False,'legal':sorted(legal),'pi0':pi0,'actions':[]}
    try:
        jobs=forecast(view,predictor,profile)
        eligible=[j for j in jobs if j['key'] in legal]
        if diagnostic:
            target=min(eligible,key=lambda j:(j['deadline']<view.now+j['work'],
                       j['deadline']-view.now-j['work'],j['work'],j['key']))['key']
            action=Intent(target,4*profile.decode_seconds,'same_info_laxity')
        else:
            actions=[fallback]
            for j in eligible:
                h=max(profile.decode_seconds,min(q,j['current']))
                for length in (4*profile.decode_seconds,h):
                    if (j['key'],length) not in [(a.key,a.horizon) for a in actions]:
                        actions.append(Intent(j['key'],length,'slo_lookahead'))
            scores=[score(jobs,a.key,a.horizon,view.now) for a in actions]
            winner=max(range(len(actions)),key=lambda i:(scores[i],-i))
            action=actions[winner]
            audit['actions']=[{'key':a.key,'h':a.horizon,'pred_good':s[0],'pred_mean':-s[1]} for a,s in zip(actions,scores)]
        if time.perf_counter()-begin>timeout:raise TimeoutError('decision_timeout')
        audit.update(accepted=action.source!='pi0',fallback=None if action.source!='pi0' else 'no_model_gain',
                     estimates=jobs,decision_seconds=time.perf_counter()-begin)
        return action,audit
    except (ValueError,RuntimeError,TimeoutError) as e:
        audit.update(fallback=str(e),decision_seconds=time.perf_counter()-begin)
        return fallback,audit
