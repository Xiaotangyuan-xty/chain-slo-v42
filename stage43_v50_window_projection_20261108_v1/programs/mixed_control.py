"""Bounded set search over paired, stage-aware terminal-credit rollouts.

Cheap public-state pruning precedes donor materialization. Every live request
remains registered with the native executor. A missing model, incumbent pool
overflow, prediction/event/time budget exhaustion or nonpositive paired gain
restores native work-conserving scheduling; none is treated as zero true utility.
"""
import itertools
import math
import time
from chain_admission import blocks
from step_cost import StepCost
from paired_rollout import PairedRollout
from stage_scenarios import Scenarios,Scenario,Stage

def eligible_decode_count(generated, remaining, elapsed, prefill_seconds, unit, ttft, tbt):
    """Native cumulative deadlines, including possible recovery after late TTFT."""
    lo, hi = max(1, 2 - generated), remaining
    if hi < lo:
        return 0
    intercept = elapsed + prefill_seconds - ttft - (generated - 1) * tbt
    slope = unit - tbt
    if abs(slope) < 1e-12:
        return hi - lo + 1 if intercept <= 1e-12 else 0
    boundary = -intercept / slope
    if slope > 0:
        hi = min(hi, math.floor(boundary + 1e-10))
    else:
        lo = max(lo, math.ceil(boundary - 1e-10))
    return max(0, hi - lo + 1)



def search_sets(pool, seeds, fits, evaluate, *, beam_width=6, max_evaluations=96,
                expired=lambda:False, exact_limit=8):
    """Small pools exhaustive; larger pools keep losing intermediate states.

    Empty/incumbent/native seeds are explicit. Pair-add, remove, 1-for-1 and
    1-for-2 neighborhoods escape a single-add greedy/KV packing trap.
    """
    cache={};stopped=False
    def visit(ids):
        nonlocal stopped
        ids=frozenset(ids)
        if ids in cache or not fits(ids):return
        if len(cache)>=max_evaluations or expired():stopped=True;return
        value=evaluate(ids)
        cache[ids]=float(value)
    for ids in seeds:visit(ids)
    if len(pool)<=exact_limit:
        for n in range(len(pool)+1):
            for ids in itertools.combinations(pool,n):
                visit(ids)
                if stopped:break
            if stopped:break
    else:
        for aid in pool:visit([aid])
        # Pair seeds survive even if their score is below every singleton.
        for pair in itertools.combinations(pool,2):
            visit(pair)
            if stopped:break
        for _ in range(len(pool)):
            before=len(cache)
            beam=sorted(cache,key=lambda x:(-cache[x],len(x),tuple(sorted(x))))[:beam_width]
            for ids in beam:
                outside=[x for x in pool if x not in ids]
                proposals=[]
                for old in ids:proposals.append(ids-{old})
                for n in [1,2]:
                    for add in itertools.combinations(outside,n):
                        proposals.append(ids|set(add))
                        for old in ids:proposals.append((ids-{old})|set(add))
                for candidate in proposals:
                    visit(candidate)
                    if stopped:break
                if stopped:break
            if stopped or len(cache)==before:break
    best=max(cache,key=lambda x:(cache[x],-len(x),tuple(sorted(x))))if cache else frozenset()
    return best,{'evaluations':len(cache),'search_exhausted_budget':stopped,
                 'exact_pool_enumeration':len(pool)<=exact_limit and not stopped,
                 'sets':cache}


def plan(actions,members,running,capacity,slots,unit,block_size=16,*,costs=None,
         previous=(),lease_remaining=None,horizon=50,candidate_multiplier=3,
         prefill_unit=256.,pool_limit=12,max_evaluations=96,
         max_seconds=.006,scenario_limit=8,plan_start=None):
    started=time.perf_counter()if plan_start is None else plan_start
    elapsed=lambda:max(0.,time.perf_counter()-started)
    expired=lambda:max_seconds is not None and elapsed()>=max_seconds
    by_id={m['request']:m for m in members}
    if len(by_id)!=len(members)or slots<1 or capacity<1:raise ValueError('Invalid physical candidate/budget')
    action_map={a['id']:a for a in actions};owner={}
    if len(action_map)!=len(actions):raise ValueError('Duplicate action identity')
    for a in actions:
        if not a['members']:raise ValueError('Empty action')
        for rid in a['members']:
            if rid not in by_id or rid in owner:raise ValueError('Invalid/duplicate action membership')
            owner[rid]=a['id']
    if set(owner)!=set(by_id):raise ValueError('Every live physical member needs an owner')
    charges={rid:blocks(m,block_size,horizon)for rid,m in by_id.items()}
    def physical(ids):return {rid for aid in ids for rid in action_map[aid]['members']}
    def fits(ids):
        rs=physical(ids);return len(rs)<=slots and sum(charges[r]for r in rs)<=capacity
    costs=costs or StepCost().snapshot(unit,max(1,len(running)))
    detail={'reason':'native_required_work_fallback','decisions':[],'backfill':[],
            'evicted':[],'selected_charge':0,'all_actions':len(actions),'action_pool':0,
            'planning_budget_seconds':max_seconds,'materialized_scenarios':0}
    def fallback(reason):
        if detail.get('actions'):detail['rejected_proposal_actions']=detail.pop('actions')
        detail.update(reason='native_required_work_fallback',fallback_reason=reason,actions=[],
                      selected_charge=0,evicted=[],plan_seconds=elapsed(),cost_model=costs.describe())
        return None,detail
    # The prefilter uses only request metadata/observed prefix and QRF point
    # length. It does not call a donor model or build any width-dependent curve.
    def cheap(a):
        return float(a.get('cheap_credit',0.))/max(1.,float(a.get('cheap_work',1.)))
    ranked=sorted(actions,key=lambda a:(a.get('native_priority',(a['arrival'],)),a['arrival'],a['id']))
    incumbent={a['id']for a in actions if set(a['members'])<=set(previous)&set(running)}
    if not fits(incumbent):incumbent=set()
    reference=set()
    for a in ranked:
        if fits(reference|{a['id']}):reference.add(a['id'])
    if not any(cheap(a)>0 for a in actions):return fallback('no_recoverable_public_state')
    limit=max(1,min(pool_limit,max(1,slots*candidate_multiplier)))
    # Never compute gains while silently omitting a resident/reference loss.
    required=reference|{owner[r]for r in running if r in owner}
    if len(required)>limit:return fallback('resident_reference_exceeds_model_pool')
    pool=list(sorted(required))
    for a in sorted(actions,key=lambda a:(-cheap(a),a['arrival'],a['id'])):
        if a['id']not in pool and cheap(a)>0 and len(pool)<limit:pool.append(a['id'])
    detail['action_pool']=len(pool)
    if expired():return fallback('cheap_pruning_budget_exhausted')
    material=[]
    for aid in pool:
        a=dict(action_map[aid])
        if 'scenario_factory'in a:a['scenarios']=a['scenario_factory']()
        if 'scenarios'in a and not a['scenarios'].rows and a['scenarios'].slack<=0:
            # A real expired deadline proves zero terminal credit, not zero
            # executor work. Keep its currently required service envelope in
            # both rollouts without consulting any donor or future output.
            ms=[by_id[r]for r in a['members']]
            stage=Stage(len(ms),max(max(1,m.get('predicted_remaining',50))for m in ms),
                        sum(m.get('prefill_tokens',0)for m in ms),
                        sum(m['prompt']+m['generated']for m in ms))
            a['scenarios']=Scenarios([Scenario((stage,),0.,'expired_current_QRF_envelope')],a['scenarios'].slack)
        if 'scenarios'in a:
            detail['materialized_scenarios']+=len(a['scenarios'].rows)
            if not a['scenarios'].supported or not a['scenarios'].rows:
                return fallback('unsupported_or_expired_reference_scenario')
        a['resident']=bool(set(a['members'])&set(running))
        a['resident_context']=sum(by_id[r]['prompt']+by_id[r]['generated']for r in a['members']if r in running)
        a['swapped']=any(by_id[r].get('state')=='swapped'for r in a['members'])
        a['resume_prefill']=sum(by_id[r]['prompt']+by_id[r]['generated']for r in a['members']if r in running and not by_id[r].get('can_swap',True))
        a['continuation_priority']=(0 if cheap(a)>0 else 1,*a.get('native_priority',(a['arrival'],)))
        material.append(a)
        if expired():return fallback('scenario_materialization_budget_exhausted')
    reference&=set(pool)
    running_context=sum(by_id[r]['prompt']+by_id[r]['generated']for r in running if r in by_id)
    window=max(.05,horizon*costs.seconds(max(1,len(running)),running_context))
    # Reserve the whole bounded planning allowance in every deadline/rollout.
    # If actual planning overruns it, no candidate intervention is committed.
    charge=max_seconds if max_seconds is not None else elapsed()
    rollout=PairedRollout(material,reference,costs,capacity,slots,window,
                          planning_seconds=charge,scenario_limit=scenario_limit,expired=expired)
    def objective(ids):return rollout.evaluate(ids)['credit']
    best,search=search_sets(pool,[reference,incumbent&set(pool),set()],fits,objective,
                            max_evaluations=max_evaluations,expired=expired)
    detail.update({k:v for k,v in search.items()if k!='sets'})
    if expired():return fallback('rollout_wall_budget_exhausted')
    pair=rollout.paired(best)
    if not pair['complete_envelope']:return fallback('rollout_event_envelope_exhausted')
    if not best or pair['delta_credit']<=1e-9 or best==reference:return fallback('no_positive_paired_terminal_gain')
    selected=physical(best)
    detail.update(reason='stage_aware_paired_terminal_credit',paired=pair,
                  selected_charge=sum(charges[r]for r in selected),evicted=sorted(set(running)-selected),
                  selected_step_seconds=costs.seconds(len(selected),sum(by_id[r]['prompt']+by_id[r]['generated']for r in selected)),objective_credit=pair['credit'],
                  actions=[{'id':a['id'],'members':a['members'],'kind':a['kind'],'arrival':a['arrival'],
                            'selected':a['id']in best,'expected_credit':pair['credit_by_action'][a['id']],
                            'probability':pair['probability_by_action'][a['id']],
                            'density':cheap(a)}for a in material],
                  plan_seconds=elapsed(),cost_model=costs.describe())
    if expired():return fallback('finalization_wall_budget_exhausted')
    return selected,detail
