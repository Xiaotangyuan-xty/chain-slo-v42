"""v50pro cached window-projection admission; original v47 pure DR loop is delegated unchanged."""
from collections import Counter, deque
import time

from jitserve.request_info import RequestType
from jitserve.scheduler.policy import SLOPolicy
from jitserve.scheduler.slo_scheduler import SLOScheduler
from vllm.core.scheduler import PreemptionMode, SchedulerOutputs
from vllm.sequence import SequenceStatus
import native_chain
from native_chain import ChainCompletionPolicy, NativeChainScheduler
from window_control import plan
from planner_config import CONFIG
from stage_scenarios import forecast_scenarios
from step_cost import StepCost
from service_commitments import Commitments
from compute_admission import choose as compute_choose
from credit_cues import root_cue,foreground_cue,compute_target
from mixed_forecast import tot_model
from token_reward import REWARD

VERSION = 'v50pro_anytime_stage_window_credit'
VARIANT = 'v50pro_physical_dynamic_commit_v1'


def is_chain(group):
    return group.request_type == RequestType.COLLECTIVE


class MixedPolicy(ChainCompletionPolicy):
    version = VERSION
    mixed_variant = VARIANT

    def __init__(self):
        super().__init__()
        self.joint_mode = False
        self.native_fallback = False
        self.mixed_priorities = {}

    def add_seq_group(self, seq_group):
        group = seq_group
        if not is_chain(group):
            return SLOPolicy.add_seq_group(self, group)
        result = super().add_seq_group(group)
        info = group.request_info
        if getattr(info, 'application', None) == 'tot_native':
            meta = self.chains[group.collection_id]
            meta['mixed_application'] = 'tot_native'
            meta['mixed_model_key'] = getattr(info, 'mixed_model_key', None)
            meta['future_max_tokens'] = info.chain_future_max_tokens
        return result


    def delete_seq_group(self, seq_group):
        group = seq_group
        self.mixed_priorities.pop(group.request_id, None)
        if is_chain(group):
            return super().delete_seq_group(group)
        return SLOPolicy.delete_seq_group(self, group)

    def native_service_priority(self, group):
        # Compatibility name used by inherited execution glue; this is our
        # public token/work priority, not SLOPolicy's native baseline score.
        now=time.perf_counter();info=group.request_info;seq=group.first_seq
        remaining=max(1,min(group.sampling_params.max_tokens,info.output_len)-seq.get_output_len())
        if is_chain(group):
            meta=self.chains[group.collection_id];slack=meta['deadline']-now
            prefix=sum(z.get('reward_input',z['declared_input'])+8*z.get('reward_output',z['output'])for ms in meta['observed'].values()for z in ms.values())
            credit=prefix+getattr(info,'chain_reward_input_tokens',seq.get_prompt_len())+8*(seq.get_output_len()+remaining)
        else:
            slo=info.slo_constraint;elapsed=time.time()-group.arrival_time
            slack=(slo[2]if group.request_type==RequestType.THROUGHPUT else slo[0]+slo[1]*max(0,group.sampling_params.max_tokens-1))-elapsed
            credit=(seq.get_prompt_len()if seq.get_output_len()==0 else 0)+8*remaining
        return (int(slack<=0),-credit/max(1.,remaining)if slack>0 else 0.,group.arrival_time)

    def get_priority(self, seq_group):
        group = seq_group
        if self.joint_mode and self.native_fallback:
            return self.native_service_priority(group)
        if self.joint_mode:
            if group.request_id in self.mixed_priorities:return self.mixed_priorities[group.request_id]
            return self.native_service_priority(group)
        return super().get_priority(group) if is_chain(group) else SLOPolicy.get_priority(self, group)

    def goals(self, groups, now):
        use_domain = any(self.chains[g.collection_id].get('mixed_application') == 'tot_native' for g in groups)
        if not self.joint_mode and not use_domain:
            return super().goals(groups, now)
        grouped = {}
        for g in groups:
            grouped.setdefault(g.collection_id, []).append(g)
        result = []
        unit = max(.01, min(.2, self._get_interval_time()))
        for cid, gs in grouped.items():
            meta = self.chains[cid]; stage = gs[0].request_info.stage_id
            if any(g.request_info.stage_id != stage for g in gs):
                raise RuntimeError('Concurrent stages violate the root barrier')
            views = [{'request':g.request_id, 'member':g.request_info.request_id,
                      'declared_input':g.request_info.input_len, 'prompt':g.first_seq.get_prompt_len(),
                      'generated':g.first_seq.get_output_len(), 'max_tokens':g.sampling_params.max_tokens,
                      'prefill_tokens':g.get_num_uncomputed_tokens() if g.is_prefill() else 0,
                      'arrival':g.arrival_time,
                      'reward_input':getattr(g.request_info,'chain_reward_input_tokens',g.first_seq.get_prompt_len())}
                     for g in gs]
            model = native_chain.MODEL; domain = 'unchanged_DR_FIT295'
            if meta.get('mixed_application') == 'tot_native':
                calibrated = tot_model(meta.get('mixed_model_key'))
                if calibrated is not None:
                    model = calibrated; domain = 'reserved_ToT_calibration'
                else:
                    domain = 'unqualified_ToT_transfer'
            costs=getattr(self,'stage_service_costs',None)or StepCost().snapshot(unit,max(1,len(groups)))
            scenarios=forecast_scenarios(model,meta,stage,views,meta['deadline']-now)
            value=scenarios.isolated(costs,current_members=len(groups))
            work=sum(max(1.,v['max_tokens']-v['generated'])for v in views)
            estimate={'supported':scenarios.supported,'probability':value['probability'],
                      'remaining_work':work,'critical_mean_seconds':sum(value['scenario_completion_seconds'])/max(1,len(scenarios.rows)),
                      'score':value['expected_credit']/max(1.,work),'expected_credited_tokens':value['expected_credit'],
                      'donors':len(scenarios.rows),'stage_aware_service':True}
            for g in gs:self.priorities[g.request_id]=(0 if value['probability']>0 else 1,-estimate['score'],meta['deadline'],g.arrival_time)
            result.append({**estimate, 'chain':cid, 'stage':stage, 'members':views,
                           'deadline':meta['deadline'], 'arrival':min(g.arrival_time for g in gs),
                           'forecast_domain':domain})
        return result



class MixedScheduler(NativeChainScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = MixedPolicy()
        self.mixed_counts = Counter()
        self.last_joint_summary = None
        self._foreground_native = False
        self._native_fallback_active = False
        self._joint_active = False
        self._joint_seen = set()
        self._joint_predictions = None
        self._joint_deadlines = None
        self.mixed_leases = {}
        self._lease_noticed = set()
        self.cost_model = StepCost()
        self.plan_events = deque(maxlen=64)
        self.plan_events_dropped = 0
        self.decision_sink = None
        self._planned_this_step = False
        self.commitments=Commitments()
        self._eviction_order={};self._cue_cache={};self._mixed_ever=False;self._last_cost=None;self._last_cost_sample=-100;self._last_cost_time=0.
        self._last_plan_wall=-float('inf');self._last_service={};self._last_generated={};self._decision_id=0

    def _release_foreground_plan(self):
        if self._foreground_native:
            self._pending_swap_out_queue.clear(); self._current_bin_queue.clear()
            self._exec_order = []; self._bins_by_index = []; self._bin_bounds = []
            self._foreground_native = False

    def _schedule(self):
        started = time.perf_counter()
        self._planned_this_step = False
        output = self._dispatch()
        elapsed = max(0.,time.perf_counter()-started)
        preempted = getattr(output,'preempted',0)
        preempted = len(preempted) if hasattr(preempted,'__len__') else preempted
        self._execution_flags = {'scheduler_seconds':elapsed,
            'planning_seconds':elapsed if self._planned_this_step else 0.,
            'preempted':int(preempted or 0),
            'ignored':len(getattr(output,'ignored_seq_groups',[])),
            'policy_path':'joint' if self.policy.joint_mode else 'pure_or_foreground','decision_id':self._decision_id}
        # Executor observer supplies the samples. Scheduler entry intervals are
        # never inserted into the execution-cost model, in any dispatch path.
        self.mixed_counts['scheduler_nanoseconds'] += int(elapsed*1e9)
        return output

    def _dispatch(self):
        groups = [g for g in list(self.running)+list(self.waiting)+list(self.swapped) if not g.is_finished()]
        chains = [g for g in groups if is_chain(g)]
        foreground = [g for g in groups if not is_chain(g)]
        phase = 'main' if any(g.request_info.client_id != 900000 for g in groups) else 'warmup'
        if foreground:
            self._restore_drain_limits()
        now=time.perf_counter()
        samples=self.cost_model.stats.get('accepted_samples',0)
        if self._last_cost is None or samples-self._last_cost_sample>=16 or now-self._last_cost_time>=.25:
            self._last_cost=self.cost_model.snapshot(.035,max(1,len(self.running)));self._last_cost_sample=samples;self._last_cost_time=now
            self.mixed_counts['cost_snapshot_refreshes']+=1
        self.policy.stage_service_costs=self._last_cost
        self._mixed_ever=self._mixed_ever or bool(foreground)
        if not foreground and not self._mixed_ever and not self.commitments.deferred:
            if self._native_fallback_active:
                self._foreground_native=True;self._native_fallback_active=False
            if self._joint_active:
                self.selected=set();self.last_members=None;self.leases.clear()
                self.last_plan=self.policy.num_schedule_count-50
            self._joint_active = False; self.policy.joint_mode = False
            self.mixed_leases.clear(); self._lease_noticed.clear()
            self._release_foreground_plan()
            self.mixed_counts[phase+'_pure_chain_calls'] += 1
            return NativeChainScheduler._schedule(self)
        self.policy.joint_mode = True
        self._release_foreground_plan()
        self.mixed_counts[phase+'_joint_calls'] += 1
        return self._schedule_joint()

    def _actions(self, groups, now):
        grouped = {}
        for group in groups:
            if is_chain(group):grouped.setdefault(group.collection_id, []).append(group)
        actions = []
        for cid, gs in grouped.items():
            meta = self.policy.chains[cid]; stage = gs[0].request_info.stage_id
            if any(g.request_info.stage_id != stage for g in gs):
                raise RuntimeError('Concurrent stages violate the root barrier')
            views = [{'request':g.request_id,'member':g.request_info.request_id,
                      'declared_input':g.request_info.input_len,'prompt':g.first_seq.get_prompt_len(),
                      'generated':g.first_seq.get_output_len(),'max_tokens':g.sampling_params.max_tokens,
                      'prefill_tokens':g.get_num_uncomputed_tokens() if g.is_prefill() else 0,
                      'reward_input':getattr(g.request_info,'chain_reward_input_tokens',g.first_seq.get_prompt_len())}
                     for g in gs]
            model = native_chain.MODEL; domain = 'unchanged_DR_FIT295'
            if meta.get('mixed_application') == 'tot_native':
                calibrated = tot_model(meta.get('mixed_model_key'))
                if calibrated is not None:model = calibrated; domain = 'reserved_ToT_calibration'
                else:domain = 'unqualified_ToT_transfer'
            slack=meta['deadline']-now
            def make_scenarios(model=model,meta=meta,stage=stage,views=views,slack=slack):
                return forecast_scenarios(model,meta,stage,views,slack,max_scenarios=CONFIG.scenario_limit)
            cue_key=(cid,stage,tuple((g.request_id,g.first_seq.get_output_len(),g.get_num_uncomputed_tokens() if g.is_prefill()else 0,getattr(g.request_info,'prediction_revision',0))for g in gs))
            if cue_key in self._cue_cache:
                old_scenarios,_,_,_=self._cue_cache[cue_key]
                from stage_scenarios import Scenarios
                scenarios=Scenarios(old_scenarios.rows,slack,supported=old_scenarios.supported,reason=old_scenarios.reason)
                cheap_credit,cheap_work,cue_detail=root_cue(scenarios,self.policy.stage_service_costs,self.scheduler_config.max_num_seqs)
            else:
                scenarios=make_scenarios()
                cheap_credit,cheap_work,cue_detail=root_cue(scenarios,self.policy.stage_service_costs,self.scheduler_config.max_num_seqs)
                self._cue_cache[cue_key]=(scenarios,cheap_credit,cheap_work,cue_detail)
                if len(self._cue_cache)>256:self._cue_cache.pop(next(iter(self._cue_cache)))
            def cached_scenarios(scenarios=scenarios):return scenarios
            actions.append({'id':'chain:'+str(cid),'kind':'compound',
                            'members':[g.request_id for g in gs],'scenario_factory':cached_scenarios,'cue':cue_detail,
                            'cheap_credit':cheap_credit if slack>0 else 0.,'cheap_work':cheap_work,
                            'slack':meta['deadline']-now,'arrival':min(g.arrival_time for g in gs),
                            'domain':domain})
        members = []
        for group in groups:
            seq = group.first_seq
            prefill = group.get_num_uncomputed_tokens() if group.is_prefill() else 0
            native_priority = tuple(self.policy.native_service_priority(group))
            members.append({'request':group.request_id,'prompt':seq.get_prompt_len(),
                            'generated':seq.get_output_len(),'max_tokens':group.sampling_params.max_tokens,
                            'prefill_tokens':prefill,'native_priority':native_priority,'arrival':group.arrival_time,
                            'predicted_remaining':max(1,min(group.sampling_params.max_tokens,group.request_info.output_len)-seq.get_output_len()),
                            'state':'running'if group in self.running else 'swapped'if group in self.swapped else 'waiting',
                            'can_swap':self.block_manager.can_swap_out(group)if group in self.running else True})
            if not is_chain(group):
                context = (group.request_type.value,REWARD.count(str(group.prompt),seq.get_prompt_len()),
                           seq.get_output_len(),group.request_info.output_len,group.sampling_params.max_tokens,
                           prefill,max(0.,time.time()-group.arrival_time),group.request_info.slo_constraint)
                fg_credit,fg_service=foreground_cue(context,self.policy.stage_service_costs,seq.get_prompt_len()+seq.get_output_len())
                actions.append({'id':'request:'+group.request_id,'kind':'foreground',
                                'members':[group.request_id],'foreground':context,
                                'penalty':self.policy.penalty_factor,'arrival':group.arrival_time,
                                'domain':'own_native_credit_service_cue',
                                'cheap_credit':fg_credit,'cheap_work':fg_service})
        lookup={m['request']:m for m in members}
        for a in actions:
            a['native_priority']=min(lookup[r]['native_priority']for r in a['members'])
            if a['kind']=='compound':a['hard_credit_expired']=a['slack']<=0
            else:
                c=a['foreground'];latest=c[7][2]if c[0]==1 else c[7][0]+c[7][1]*max(0,c[4]-1)
                a['hard_credit_expired']=c[6]>=latest
        return actions,members

    def _schedule_joint(self):
        native_frame=self.policy.update_schedule_count();self.free_finished_seq_groups()
        groups=[g for g in list(self.running)+list(self.waiting)+list(self.swapped) if not g.is_finished()]
        live={g.request_id for g in groups};tick=self.policy.num_schedule_count
        for rid in list(self.mixed_leases):
            if rid not in live:self.mixed_leases.pop(rid,None);self._lease_noticed.discard(rid)
        remaining={g.request_id:max(0,50-(g.first_seq.get_output_len()-self.mixed_leases[g.request_id]))
                   for g in groups if g.request_id in self.mixed_leases}
        expired={rid for rid,n in remaining.items()if n==0}
        predictions=tuple(sorted((g.request_id,getattr(g.request_info,'prediction_revision',0))for g in groups))
        now=time.perf_counter();wall=time.time()
        def expired_credit(g):
            if is_chain(g):return self.policy.chains[g.collection_id]['deadline']<=now
            slo=g.request_info.slo_constraint
            latest=(slo[2] if g.request_type==RequestType.THROUGHPUT
                    else slo[0]+slo[1]*max(0,g.sampling_params.max_tokens-1))
            return wall-g.arrival_time>=latest
        deadlines=tuple(sorted((g.request_id,expired_credit(g))for g in groups))
        views={g.request_id:{'generated':g.first_seq.get_output_len(),'root':str(g.collection_id),'running':g in self.running,'expired':expired_credit(g)}for g in groups}
        self.commitments.observe(views,now)
        protected_now=self.commitments.protected(views,now)
        remaining={rid:(n if rid in protected_now else 0)for rid,n in remaining.items()}
        for rid,v in views.items():
            if rid not in self._last_service or v['generated']!=self._last_generated.get(rid):self._last_service[rid]=now
            self._last_generated[rid]=v['generated']
        for rid in list(self._last_service):
            if rid not in live:self._last_service.pop(rid,None);self._last_generated.pop(rid,None)
        reasons=[]
        if self.commitments.due(views,now)or self.commitments.restoring:reasons.append('restoration_obligation')
        if not self._joint_active:reasons.append('mixed_entry')
        if live-self._joint_seen:reasons.append('new_request_or_stage')
        if self._joint_seen-live:reasons.append('completion_or_abort')
        if predictions!=self._joint_predictions:reasons.append('prediction_revision')
        if self._joint_deadlines is not None and deadlines!=self._joint_deadlines:reasons.append('deadline_or_membership_change')
        if expired-self._lease_noticed:reasons.append('continuity_quantum_expired')
        if tick-self.last_plan>=50:reasons.append('periodic')
        self._joint_active=True
        urgent=bool(self.commitments.due(views,now)or self.commitments.restoring)or not(self.selected and set(self.selected)&live)
        if reasons and not urgent and now-self._last_plan_wall<.25:
            self.mixed_counts['coalesced_plan_events']+=1;reasons=[]
        if reasons:
            self._last_plan_wall=now
            self._planned_this_step=True;started=time.perf_counter()
            unit=max(.001,min(1.,self.policy._get_interval_time()))
            running={g.request_id for g in self.running}
            capacity=self.block_manager.num_total_gpu_blocks-self.block_manager.watermark_blocks
            recoverable=any(not expired_credit(g)for g in groups)
            costs=self.policy.stage_service_costs
            actions,members=self._actions(groups,started)
            required=[rid for rid in live if now-self._last_service[rid]>=30.]
            # One oldest mandatory-work opportunity per decision; Commitments
            # still enforces service/restoration and root cooldown.
            required=sorted(required,key=lambda r:(self._last_service[r],r))[:1]
            tbt=min((g.request_info.slo_constraint[1]for g in groups if g.request_type.value==0),default=.1)
            target=compute_target(tbt,CONFIG.max_seconds+CONFIG.final_margin_seconds,.010)
            admission,admission_detail=compute_choose(actions,members,running,capacity,self.scheduler_config.max_num_seqs,costs,required=required,target_step=target)
            self.mixed_counts['compute_admission_calls']+=1
            self.mixed_counts['compute_blocked_admissions']+=admission_detail.get('blocked_admissions',0)
            optimizer_proposal=None
            if self.commitments.due(views,now)or self.commitments.restoring:
                self.selected=set(running);detail={'reason':'restore_before_replanning','backfill':[],'evicted':[],'actions':[],'all_actions':len(actions),'action_pool':0,'materialized_scenarios':0,'selected_charge':0,'cost_model':costs.describe()}
            elif recoverable:
                proposed,detail=plan(actions,members,running,capacity,self.scheduler_config.max_num_seqs,unit,self.cache_config.block_size,costs=costs,previous=self.selected or (),lease_remaining=remaining,plan_start=started,swap_capacity_blocks=self.block_manager.get_num_free_cpu_blocks())
                optimizer_proposal=set(proposed)if proposed is not None else None
                # Projections govern displacement. Low-cost nonpreemptive
                # admission covers all fallback paths with the same guard.
                if proposed is None or set(proposed)==running:self.selected=admission
                else:
                    self.selected=set(proposed)
                    ctx=lambda ids:sum(m['prompt']+m['generated']for m in members if m['request']in ids)
                    observed_guard=max(target,costs.seconds(max(1,len(running)),ctx(running),0))
                    if self.selected-running and costs.seconds(len(self.selected),ctx(self.selected),0)>observed_guard:
                        self.selected=admission;detail['compute_guard_rejected_projection']=True
                detail['admission']=admission_detail
            else:
                self.selected=admission
                detail={'reason':'required_work_compute_drain','fallback_reason':'no_recoverable_public_state','backfill':[],'evicted':[],'actions':[],'all_actions':len(actions),'action_pool':0,'materialized_scenarios':0,'selected_charge':0,'cost_model':costs.describe(),'admission':admission_detail}
            by_request={m['request']:m for m in members}
            from chain_admission import blocks
            def commitment_fits(selected):
                return len(selected)<=self.scheduler_config.max_num_seqs and sum(blocks(by_request[r],self.cache_config.block_size)for r in selected if r in by_request)<=capacity
            self.selected,commitment=self.commitments.resolve(self.selected,views,now,commitment_fits,lambda ids:costs.seconds(max(1,len(ids)),sum(by_request[r]['prompt']+by_request[r]['generated']for r in ids),0)<=max(target,costs.seconds(max(1,len(running)),sum(by_request[r]['prompt']+by_request[r]['generated']for r in running),0)))
            self.selected=set(running)if self.selected is None else set(self.selected)
            self.policy.draining=False;self.policy.native_fallback=False
            self.policy.mixed_priorities={r:(0 if a.get('cheap_credit',0)>0 else 1,-a.get('cheap_credit',0)/max(.001,a.get('cheap_work',1.)),a['arrival'])for a in actions for r in a['members']}
            detail['commitment']=commitment
            displaced=[r for r in by_request if r in running and r not in self.selected]
            displaced.sort(key=lambda r:tuple(by_request[r]['native_priority']),reverse=True)
            self._eviction_order={r:i for i,r in enumerate(displaced)}
            detail['actual_eviction_order']=displaced
            detail['evicted']=sorted(displaced)
            previous_leases=self.mixed_leases
            self.mixed_leases={g.request_id:previous_leases.get(g.request_id,g.first_seq.get_output_len())
                               for g in groups if self.selected is not None and g.request_id in self.selected}
            self._lease_noticed=expired & self.mixed_leases.keys()
            self.last_plan=tick;self._joint_seen=live;self._joint_predictions=predictions;self._joint_deadlines=deadlines
            self.mixed_counts['joint_plans']+=1
            self.mixed_counts['completed_candidate_sets']+=detail.get('complete_sets',0)
            proposal_applied=(detail.get('reason')=='physical_paired_terminal_credit'and optimizer_proposal==self.selected and self.selected!=running and not detail.get('compute_guard_rejected_projection'))
            detail['optimizer_proposal_applied']=proposal_applied
            detail['optimizer_proposal_members']=sorted(optimizer_proposal)if optimizer_proposal is not None else None
            self.mixed_counts['candidate_interventions']+=proposal_applied
            self.mixed_counts['nonpreemptive_admitted_members']+=len(set(self.selected or())-running)if not(running-set(self.selected or()))else 0
            self.mixed_counts['kept_completed_gain_after_budget']+=detail.get('budget_exhaustion_kept_completed_improvement',False)
            if detail.get('fallback_reason'):self.mixed_counts['controlled_fallback:'+detail['fallback_reason']]+=1
            self.mixed_counts['event_plans']+='new_request_or_stage'in reasons
            self.mixed_counts['backfill_members']+=len(detail['backfill'])
            self.mixed_counts['planned_evictions']+=len(detail['evicted'])
            self.mixed_counts['compute_filtered_plans']+=self.selected is not None and len(self.selected)<len(live)
            event={'tick':tick,'monotonic':started,'reasons':reasons,'selected':sorted(self.selected)if self.selected is not None else None,
                   'running_before':sorted(running),'physical_candidates':len(live),'capacity':capacity,
                   'slots':self.scheduler_config.max_num_seqs,'detail':detail,
                   'plan_seconds':time.perf_counter()-started}
            self._decision_id+=1;event['decision_id']=self._decision_id
            event['outer_plan_seconds']=event['plan_seconds']
            self.mixed_counts['outer_plans_over_reserve']+=event['plan_seconds']>CONFIG.max_seconds+CONFIG.final_margin_seconds
            self.last_joint_summary=event
            if len(self.plan_events)==self.plan_events.maxlen and self.decision_sink is None:self.plan_events_dropped+=1
            self.plan_events.append(event)
            if self.decision_sink:self.decision_sink(event)
        selected=set(self.selected or ())&live
        self.last_members=live;self.last_recorded_selected=frozenset(selected)
        if not selected and groups:
            # New membership will trigger a full decision on the next call;
            # there is no hidden native fallback in this step.
            self.mixed_counts['empty_selection_steps']+=1
        swaps=[];preemptions=0
        for group in sorted(list(self.running),key=lambda g:self._eviction_order.get(g.request_id,len(self._eviction_order))):
            if group.request_id in selected or preemptions>=self.max_swaps_per_iter:continue
            mode=PreemptionMode.SWAP if self.block_manager.can_swap_out(group)else PreemptionMode.RECOMPUTE
            mode=self._preempt(group,swaps,mode);self.running.remove(group);self.leases.pop(group.request_id,None)
            (self.swapped if mode==PreemptionMode.SWAP else self.waiting).append(group);preemptions+=1
        if swaps or preemptions:return SchedulerOutputs([],0,0,[],swaps,[],[],0,len(self.running),preemptions)
        hidden_waiting=[g for g in self.waiting if g.request_id not in selected]
        hidden_swapped=[g for g in self.swapped if g.request_id not in selected]
        self.waiting=deque(sorted((g for g in self.waiting if g.request_id in selected),key=self.policy.get_priority))
        self.swapped=deque(sorted((g for g in self.swapped if g.request_id in selected),key=self.policy.get_priority))
        self.running=deque(sorted(self.running,key=self.policy.get_priority))
        try:return self._schedule_chunked_prefill()
        finally:self.waiting.extend(hidden_waiting);self.swapped.extend(hidden_swapped)
