"""This mechanism's own nonpreemptive compute/credit admission and required work.

Uses public credit estimates, executor host-cost observations and queue aging.
No native JITServe score, graph ratio or baseline pack is called. The cost guard
is a control target, not an additional experiment validity/SLO constraint.
"""
import math
from chain_admission import blocks

def choose(actions,members,running,capacity,slots,costs,*,target_step=.080,horizon=50,required=()):
    by={m['request']:m for m in members};owner={r:a for a in actions for r in a['members']}
    live=set(by);running=set(running)&live;selected=set(running);required=set(required)&live
    charges={r:blocks(m,16,horizon)for r,m in by.items()}
    def context(rs):return sum(by[r]['prompt']+by[r]['generated']for r in rs)
    def fits(rs):return len(rs)<=slots and sum(charges[r]for r in rs)<=capacity
    def order(r):
        a=owner[r];return (0 if r in required else 1,1 if a.get('hard_credit_expired',False)else 0,-a.get('cheap_credit',0)/max(.001,a.get('cheap_work',1.)),a['arrival'],r)
    # Expired required work gets a bounded opportunity even when live credit
    # remains. Eviction eligibility is decided by Commitments, not this helper.
    for r in sorted(required,key=order):
        if fits(selected|{r}):selected.add(r)
    recoverable=any(not a.get('hard_credit_expired',False)for a in actions)
    if recoverable and any(owner[r].get('cheap_credit',0)>0 for r in live-selected):
        # Modeled zero-credit residents may defer to recoverable work; all
        # removal still passes runtime quantum/cooldown/restoration checks.
        selected={r for r in selected if not owner[r].get('hard_credit_expired',False)or r in required}
    pending=sorted(live-selected,key=order)
    if not recoverable:
        # Enumerate widths: a nonmonotone batching curve must not trap a
        # greedy single-add rule. Keep current residents, fill by age.
        best=set(selected);best_rate=len(best)/max(1e-9,costs.seconds(max(1,len(best)),context(best),0))
        pack=set(selected)
        for r in pending:
            if not fits(pack|{r}):continue
            pack.add(r);rate=len(pack)/max(1e-9,costs.seconds(len(pack),context(pack),0))
            if rate>=best_rate:best=set(pack);best_rate=rate
        return best,{'mode':'required_drain','compute_limited':False,'target_step_seconds':target_step,'projected_step_seconds':costs.seconds(max(1,len(best)),context(best),0)}
    blocked=0;exploration=0
    supported=max([k[0]for k,_,_ in getattr(costs,'strata',())if not k[2]and not k[3]and not k[4]and not k[5]]or[0])
    initial_count=len(running)
    for r in pending:
        if owner[r].get('hard_credit_expired',False)and r not in required:continue
        candidate=selected|{r}
        if not fits(candidate):continue
        step=costs.seconds(len(candidate),context(candidate),0)
        probe=(len(candidate)>supported and len(candidate)<=8 and len(candidate)<=max(1,initial_count+1)and (not running or costs.seconds(len(running),context(running),0)<=target_step))
        if selected and step>target_step and r not in required and not probe:blocked+=1;continue
        exploration+=int(step>target_step and probe)
        selected=candidate
    # Always make progress from a cold start; a sparse-cost fallback cannot
    # reject the only runnable request indefinitely.
    if not selected and pending:
        r=next((r for r in pending if fits({r})),None)
        if r is not None:selected={r}
    return selected,{'mode':'credit_compute_admission','compute_limited':bool(blocked),'blocked_admissions':blocked,'uncertain_compute_probe_admissions':exploration,'target_step_seconds':target_step,'projected_step_seconds':costs.seconds(max(1,len(selected)),context(selected),0)}
