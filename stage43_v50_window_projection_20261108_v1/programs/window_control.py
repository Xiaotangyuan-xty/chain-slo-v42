"""Anytime token-credit intervention around corrected native execution.

All resident/reference losses are scored. Only challenger generation is capped.
A completed improvement survives later search-budget exhaustion. Unknown or
incomplete comparisons never become zero-loss proposals.
"""
import itertools
import time
from chain_admission import blocks
from step_cost import StepCost
from planner_config import CONFIG
from stage_scenarios import Stage,Scenario,Scenarios
from window_projection import WindowProjector,ProjectionBudgetExceeded


def candidate_sets(reference,incumbent,ranked,low,challengers):
    """Native/incumbent/rank packs, add/remove pairs and bounded exchanges."""
    yield frozenset(reference)
    yield frozenset(incumbent)
    yield frozenset(ranked)
    for n in [1,2]:
        for removed in itertools.combinations(low,n):yield frozenset(reference-set(removed))
    for n in [2,1]:
        for added in itertools.combinations(challengers[:3],n):yield frozenset(reference|set(added))
    for remove_n,add_n in [(1,1),(1,2),(2,1),(2,2)]:
        for removed in itertools.combinations(low,remove_n):
            for added in itertools.combinations(challengers[:3],add_n):
                yield frozenset((reference-set(removed))|set(added))


def plan(actions,members,running,capacity,slots,unit,block_size=16,*,costs=None,
         previous=(),lease_remaining=None,horizon=CONFIG.horizon,candidate_multiplier=3,
         pool_limit=CONFIG.challenger_count,max_evaluations=CONFIG.max_evaluations,max_seconds=CONFIG.max_seconds,scenario_limit=CONFIG.scenario_limit,
         plan_start=None,max_evicted_members=CONFIG.max_evicted_members,min_gain=CONFIG.min_gain,relative_gain=CONFIG.relative_gain,swap_capacity_blocks=512):
    started=time.perf_counter()if plan_start is None else plan_start
    elapsed=lambda:max(0.,time.perf_counter()-started)
    expired=lambda:max_seconds is not None and elapsed()>=max_seconds
    reserve=max_seconds+CONFIG.final_margin_seconds if max_seconds is not None else 0.
    by_id={m['request']:m for m in members};amap={a['id']:a for a in actions}
    if len(by_id)!=len(members)or len(amap)!=len(actions)or slots<1 or capacity<1:raise ValueError('Invalid candidate identities/budgets')
    owner={}
    for a in actions:
        if not a['members']:raise ValueError('Empty action')
        for rid in a['members']:
            if rid not in by_id or rid in owner:raise ValueError('Duplicate/unknown member')
            owner[rid]=a['id']
    if set(owner)!=set(by_id):raise ValueError('Every live member must be represented')
    costs=costs or StepCost().snapshot(unit,max(1,len(running)))
    charges={rid:blocks(m,block_size,horizon)for rid,m in by_id.items()}
    def physical(ids):return {rid for aid in ids for rid in amap[aid]['members']}
    def fits(ids):
        rs=physical(ids);return len(rs)<=slots and sum(charges[r]for r in rs)<=capacity
    def cheap(a):return a.get('cheap_credit',0.)/max(1.,a.get('cheap_work',1.))
    detail={'reason':'native_required_work_fallback','decisions':[],'backfill':[],
            'evicted':[],'actions':[],'all_actions':len(actions),'action_pool':0,
            'selected_charge':0,'materialized_scenarios':0,'complete_sets':0,
            'planning_budget_seconds':max_seconds,'reserved_planning_seconds':reserve,
            'reference_semantics':'corrected_stage_budget_native_pack_with_expiry_tier'}
    def fallback(reason):
        detail.update(fallback_reason=reason,reason='native_required_work_fallback',actions=[],
                      selected_charge=0,evicted=[],plan_seconds=elapsed(),cost_model=costs.describe())
        return None,detail
    if not actions or not any(cheap(a)>0 for a in actions):return fallback('no_recoverable_public_state')
    native=sorted(actions,key=lambda a:(a.get('native_priority',(a['arrival'],)),a['arrival'],a['id']))
    reference=set()
    for a in native:
        if fits(reference|{a['id']}):reference.add(a['id'])
    if not reference:return fallback('no_atomic_physical_reference')
    incumbents={owner[r]for r in running if r in owner}
    challenger_order=sorted((a for a in actions if a['id']not in reference and cheap(a)>0),key=lambda a:(-cheap(a),a['arrival'],a['id']))
    challengers=[a['id']for a in challenger_order[:pool_limit]]
    model_ids=reference|incumbents|set(challengers)|{a['id']for a in actions if a['kind']=='compound'and cheap(a)>0}
    # Foreground rows are O(1); a large foreground batch is no reason to discard
    # every comparison. Only unbounded *compound* materialization is refused.
    if sum(amap[x]['kind']=='compound'and cheap(amap[x])>0 for x in model_ids)>8:return fallback('compound_model_budget')
    material=[]
    for a0 in actions:
        if a0['id']not in model_ids and a0['kind']=='compound':continue
        a=dict(a0)
        if 'scenario_factory'in a:a['scenarios']=a['scenario_factory']()
        if 'scenarios'in a:
            b=a['scenarios']
            if not b.rows and b.slack<=0:
                ms=[by_id[r]for r in a['members']]
                st=Stage(len(ms),max(max(1,m.get('predicted_remaining',50))for m in ms),sum(m.get('prefill_tokens',0)for m in ms),sum(m['prompt']+m['generated']for m in ms))
                a['scenarios']=Scenarios([Scenario((st,),0.,'expired_current_public_prediction')],b.slack)
            elif not b.supported or not b.rows:return fallback('unsupported_reference_scenario')
            detail['materialized_scenarios']+=len(a['scenarios'].rows)
        a['resident']=any(r in running for r in a['members'])
        a['swapped']=any(by_id[r].get('state')=='swapped'for r in a['members'])
        a['resume_prefill']=sum(by_id[r]['prompt']+by_id[r]['generated']for r in a['members']if r in running and not by_id[r].get('can_swap',True))
        material.append(a)
        if expired():return fallback('materialization_budget')
    live_model={a['id']for a in material};reference&=live_model;incumbents&=live_model
    detail['action_pool']=len(model_ids);detail['reference_actions']=len(reference);detail['challenger_count']=len(challengers)
    context=sum(by_id[r]['prompt']+by_id[r]['generated']for r in physical(reference))
    window=max(.05,horizon*costs.seconds(max(1,len(physical(reference))),context))
    try:
        projector=WindowProjector(material,reference,costs,capacity,slots,by_id,window,reserve,expired,swap_capacity_blocks,block_size)
        baseline=projector.evaluate(reference)
    except ProjectionBudgetExceeded:return fallback('reference_projection_budget')
    detail['complete_sets']=1;best=frozenset(reference);best_value=baseline['credit'];seen={best};budget_stop=False
    # Cheap density is a *proposal* only. The final objective is paired native
    # token credit; no width/work multiplier enters that objective.
    ranked=set()
    for a in sorted(material,key=lambda a:(-cheap(a),a['arrival'],a['id'])):
        if fits(ranked|{a['id']}):ranked.add(a['id'])
    low=sorted(reference,key=lambda aid:(cheap(amap[aid]),amap[aid]['arrival'],aid))[:3]
    # Focus is bounded in time and scored against losses of *every* peer.
    # Fresh latency prefills stay in the packet; stage/barrier completions and
    # new arrivals still trigger re-evaluation by the runtime.
    focus=[]
    roots=sorted((a for a in material if a['kind']=='compound'and a['cheap_credit']>0),
                 key=lambda a:(-baseline['credit_by_action'][a['id']]/max(1.,a['cheap_work']),a['arrival']))[:2]
    fresh={a['id']for a in material if 'foreground'in a and a['foreground'][0]==0 and a['foreground'][2]==0 and a['id']in reference}
    for root in roots:
        packet=fresh|{root['id']}
        for aid in sorted(reference,key=lambda x:(-cheap(amap[x]),amap[x]['arrival'],x)):
            if len(physical(reference)-physical(packet))<=max_evicted_members and len(set(running)-physical(packet))<=max_evicted_members:break
            packet.add(aid)
        if fits(packet):focus.append(frozenset(packet))
    proposals=itertools.chain([frozenset(reference),frozenset(incumbents)],focus,
                              candidate_sets(reference,incumbents,ranked,low,challengers))
    for proposal in proposals:
        if proposal in seen or not proposal or not proposal<=live_model or not fits(proposal):continue
        seen.add(proposal)
        # Limit physical churn; unlike v49, every peer's delayed credit and
        # aggregate swap/recompute cost are included in the comparison.
        if len(physical(reference)-physical(proposal))>max_evicted_members:continue
        if len(set(running)-physical(proposal))>max_evicted_members:continue
        if detail['complete_sets']>=max_evaluations or expired():budget_stop=True;break
        try:value=projector.evaluate(proposal)
        except ProjectionBudgetExceeded:budget_stop=True;break
        detail['complete_sets']+=1
        if value['credit']>best_value+1e-9:best=proposal;best_value=value['credit']
    detail['search_budget_stopped']=budget_stop
    # Paired results are already memoized; retrieving a completed best result
    # must not turn it into a timeout failure merely because search stopped.
    if best==frozenset(reference):return fallback('no_completed_positive_gain')
    before=projector.memo[frozenset(reference)];after=projector.memo[best]
    delta={a['id']:after['credit_by_action'][a['id']]-before['credit_by_action'][a['id']]for a in material}
    pair={**after,'reference_credit':before['credit'],'delta_credit':after['credit']-before['credit'],
          'resident_loss':sum(max(0.,-delta[a['id']])for a in material if a['resident']),
          'candidate_gain':sum(max(0.,v)for v in delta.values()),
          'paired_action_delta':{k:v for k,v in delta.items()if v},
          'zero_deltas_omitted':True,'same_window_seconds':window,'planning_delay_seconds':reserve}
    # Unaffected credit is a constant in the counterfactual objective; it
    # must not increase the intervention threshold merely because the queue
    # has more unrelated work. Scale hysteresis by changed credit only.
    required=max(min_gain,relative_gain*max(1.,pair['resident_loss']+pair['candidate_gain']))
    if pair['delta_credit']<=required:return fallback('gain_below_switch_margin')
    if max_seconds is not None and elapsed()>reserve:return fallback('planning_reserve_exceeded')
    selected=physical(best)
    details=[]
    for a in material:
        if a['id']in best or delta[a['id']]:
            details.append({'id':a['id'],'kind':a['kind'],'members':a['members'],'arrival':a['arrival'],
                'selected':a['id']in best,'expected_credit':after['credit_by_action'][a['id']],
                'probability':after['probability_by_action'][a['id']],'density':cheap(a)})
    detail.update(reason='anytime_stage_window_token_credit',paired=pair,actions=details,
        selected_charge=sum(charges[r]for r in selected),evicted=sorted(set(running)-selected),
        objective_credit=best_value,required_credit_gain=required,
        budget_exhaustion_kept_completed_improvement=budget_stop,
        plan_seconds=elapsed(),cost_model=costs.describe())
    return selected,detail
