"""Bounded root-terminal-credit controller; no A/Astar priority is imported.

Physical request IDs define no-change. Cheap conditional token/work estimates
only propose sets; final decisions compare proposal-specific event rollouts and
charge every modeled peer, restoration, integer prefill and transfer round.
"""
import itertools,time
from chain_admission import blocks
from planner_config import CONFIG
from step_cost import StepCost
from stage_scenarios import Stage,Scenario,Scenarios
from window_projection import WindowProjector,ProjectionBudgetExceeded

def plan(actions,members,running,capacity,slots,unit,block_size=16,*,costs=None,previous=(),lease_remaining=None,
         horizon=50,candidate_multiplier=3,pool_limit=4,max_evaluations=5,max_seconds=.008,scenario_limit=4,
         plan_start=None,max_evicted_members=3,min_gain=8.,relative_gain=.01,swap_capacity_blocks=512):
    started=time.perf_counter()if plan_start is None else plan_start
    elapsed=lambda:time.perf_counter()-started
    expired=lambda:max_seconds is not None and elapsed()>=max_seconds
    by={m['request']:m for m in members};amap={a['id']:a for a in actions};running=set(running)&set(by)
    if len(by)!=len(members)or len(amap)!=len(actions):raise ValueError('Duplicate physical/action identity')
    owner={r:a['id']for a in actions for r in a['members']}
    if len(owner)!=sum(len(a['members'])for a in actions)or set(owner)!=set(by):raise ValueError('Every physical member must have exactly one owner')
    costs=costs or StepCost().snapshot(unit,max(1,len(running)))
    charges={r:blocks(m,block_size,horizon)for r,m in by.items()}
    fits=lambda rs:len(rs)<=slots and sum(charges[r]for r in rs)<=capacity
    cheap=lambda a:a.get('cheap_credit',0.)/max(.001,a.get('cheap_work',1.))
    detail={'reason':'physical_no_change','decisions':[],'backfill':[],'evicted':[],'actions':[],'all_actions':len(actions),'action_pool':0,'selected_charge':sum(charges[r]for r in running),'materialized_scenarios':0,'complete_sets':0,'planning_budget_seconds':max_seconds,'reserved_planning_seconds':(max_seconds or 0)+.002,'reference_semantics':'exact_physical_running','reference_members':sorted(running),'cost_model':costs.describe()}
    def unchanged(reason):
        detail.update(fallback_reason=reason,plan_seconds=elapsed())
        return set(running),detail
    if not actions:return unchanged('empty')
    ranked=sorted(actions,key=lambda a:(-cheap(a),a['arrival'],a['id']))
    challengers=ranked[:pool_limit];required_ids={owner[r]for r in running}|{a['id']for a in challengers}
    # Unmodeled waiting compounds cannot silently become zero-loss victims.
    # In that case refuse active intervention; runtime still admits work using
    # its separate nonpreemptive required-work controller.
    material=[]
    for a0 in actions:
        if expired():return unchanged('materialization_budget')
        a=dict(a0)
        if 'scenario_factory'in a:
            if a.get('slack',1.)<=0:
                ms=[by[r]for r in a['members']];st=Stage(len(ms),max(max(1,m.get('predicted_remaining',50))for m in ms),sum(m.get('prefill_tokens',0)for m in ms),sum(m['prompt']+m['generated']for m in ms))
                a['scenarios']=Scenarios([Scenario((st,),0.)],a['slack'])
            else:a['scenarios']=a['scenario_factory']()
        if'scenarios'in a:
            if not a['scenarios'].supported or not a['scenarios'].rows:return unchanged('unsupported_live_scenario')
            detail['materialized_scenarios']+=len(a['scenarios'].rows)
        a['resident']=bool(set(a['members'])&running);material.append(a)
    detail['action_pool']=len(material)
    if expired():return unchanged('materialization_budget')
    window=1.5
    try:
        projector=WindowProjector(material,running,costs,capacity,slots,by,window,(max_seconds or 0)+.002,expired,swap_capacity_blocks,block_size,scenario_limit=scenario_limit)
        baseline=projector.evaluate(running)
    except ProjectionBudgetExceeded:return unchanged('no_change_projection_budget')
    detail['complete_sets']=1;best=set(running);best_value=baseline['credit'];best_result=baseline;seen={frozenset(best)};stopped=False
    protected={r for r,n in(lease_remaining or{}).items()if n>0 and r in running}
    proposals=[]
    # Keep physical siblings distinct. Adding an action is a proposal, not an
    # implicit expansion of the no-change incumbent root.
    pack=set(running)
    for a in challengers:
        extra=set(a['members'])-pack
        if fits(pack|extra):pack|=extra
    proposals.append(pack)
    # Joint additions/exchanges are evaluated before single additions so a
    # local single-action loss cannot hide a feasible 1-for-2 or batching gain.
    fresh=[a for a in challengers if not set(a['members'])<=running]
    pair_count=0
    for left,right in itertools.combinations(fresh[:4],2):
        desired=set(left['members'])|set(right['members'])
        rs=running|desired
        if fits(rs):proposals.append(rs);pair_count+=1
        victims=sorted(running-desired-protected,key=lambda r:(cheap(amap[owner[r]]),r))[:max_evicted_members]
        for count in [1,2]:
            for removed in itertools.combinations(victims,count):
                rs=(running-set(removed))|desired
                if fits(rs):proposals.append(rs);pair_count+=1
    detail['joint_candidates_generated']=pair_count
    for a in challengers:
        desired=set(a['members']);keep=running|desired
        if fits(keep):proposals.append(keep)
        victim=sorted(running-desired-protected,key=lambda r:(cheap(amap[owner[r]]),by[r].get('arrival',0.),r))
        for count in [1,2,3]:
            take=victim[:count]
            rs=(running-set(take))|desired
            if take and fits(rs):proposals.append(rs)
    dense=set(protected)
    for a in ranked:
        rs=dense|set(a['members'])
        if fits(rs):dense=rs
    proposals.append(dense)
    for proposal in proposals:
        if expired()or len(seen)>=max_evaluations:stopped=True;break
        proposal=set(proposal);key=frozenset(proposal)
        if key in seen or not fits(proposal)or len(running-proposal)>max_evicted_members or protected-proposal:continue
        seen.add(key)
        try:value=projector.evaluate(proposal)
        except ProjectionBudgetExceeded:stopped=True;break
        detail['complete_sets']+=1
        if value['credit']>best_value:best=proposal;best_value=value['credit'];best_result=value
    delta={a['id']:best_result['credit_by_action'][a['id']]-baseline['credit_by_action'][a['id']]for a in material}
    loss=sum(max(0.,-v)for v in delta.values());gain=sum(max(0.,v)for v in delta.values());required=max(min_gain,relative_gain*(loss+gain))
    detail.update(search_budget_stopped=stopped,required_credit_gain=required)
    if best==running or best_value-baseline['credit']<=required:return unchanged('no_completed_positive_gain')
    detail.update(reason='physical_paired_terminal_credit',paired={**best_result,'reference_credit':baseline['credit'],'delta_credit':best_value-baseline['credit'],'resident_loss':sum(max(0.,-delta[a['id']])for a in material if a['resident']),'candidate_gain':gain,'paired_action_delta':{k:v for k,v in delta.items()if v},'same_window_seconds':window,'planning_delay_seconds':(max_seconds or 0)+.002},evicted=sorted(running-best),selected_charge=sum(charges[r]for r in best),budget_exhaustion_kept_completed_improvement=stopped)
    detail['actions']=[{'id':a['id'],'kind':a['kind'],'members':a['members'],'arrival':a['arrival'],'selected':bool(set(a['members'])&best),'expected_credit':best_result['credit_by_action'][a['id']],'probability':best_result['probability_by_action'][a['id']],'density':cheap(a)}for a in material if set(a['members'])&best or delta[a['id']]]
    detail['plan_seconds']=elapsed();return best,detail
