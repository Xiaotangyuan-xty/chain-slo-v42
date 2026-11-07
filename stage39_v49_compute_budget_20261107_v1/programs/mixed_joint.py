"""v49 mixed compute admission; original v47 pure DR loop is delegated unchanged."""
from collections import Counter, deque
import time

from jitserve.request_info import RequestType
from jitserve.scheduler.policy import SLOPolicy
from jitserve.scheduler.slo_scheduler import SLOScheduler
from vllm.core.scheduler import PreemptionMode, SchedulerOutputs
from vllm.sequence import SequenceStatus
import native_chain
from native_chain import ChainCompletionPolicy, NativeChainScheduler
from mixed_control import plan
from credit_curve import forecast_curve
from step_cost import StepCost
from mixed_forecast import tot_model
from token_reward import REWARD

VERSION = 'v49_mixed_terminal_credit_compute_budget'
VARIANT = 'v49_observed_compute_terminal_credit_v1'


def is_chain(group):
    return group.request_type == RequestType.COLLECTIVE


class MixedPolicy(ChainCompletionPolicy):
    version = VERSION
    mixed_variant = VARIANT

    def __init__(self):
        super().__init__()
        self.joint_mode = False
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

    def get_priority(self, seq_group):
        group = seq_group
        if self.joint_mode:
            return self.mixed_priorities.get(group.request_id, (2, 0., group.arrival_time))
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
            estimate = model.estimate(meta, stage, views, meta['deadline'] - now, unit)
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
        self._joint_active = False
        self._joint_seen = set()
        self._joint_predictions = None
        self._joint_deadlines = None
        self.mixed_leases = {}
        self._lease_noticed = set()
        self.cost_model = StepCost()
        self._previous_decode = None
        self.plan_events = deque(maxlen=64)
        self.plan_events_dropped = 0
        self.decision_sink = None
        self._planned_this_step = False

    def _release_foreground_plan(self):
        if self._foreground_native:
            self._pending_swap_out_queue.clear(); self._current_bin_queue.clear()
            self._exec_order = []; self._bins_by_index = []; self._bin_bounds = []
            self._foreground_native = False

    def _schedule(self):
        started = time.perf_counter()
        previous = self._previous_decode
        if previous and previous['eligible']:
            self.cost_model.observe(previous['members'], started - previous['started'])
        self._planned_this_step = False
        output = self._dispatch()
        groups = output.scheduled_seq_groups
        eligible = (bool(groups) and not output.num_prefill_groups
                    and not output.blocks_to_swap_in and not output.blocks_to_swap_out
                    and not output.blocks_to_copy and not self._planned_this_step
                    and output.num_batched_tokens == len(groups))
        self._previous_decode = {'started': started, 'members': len(groups), 'eligible': eligible}
        return output

    def _dispatch(self):
        groups = [g for g in list(self.running)+list(self.waiting)+list(self.swapped) if not g.is_finished()]
        chains = [g for g in groups if is_chain(g)]
        foreground = [g for g in groups if not is_chain(g)]
        phase = 'main' if any(g.request_info.client_id != 900000 for g in groups) else 'warmup'
        if foreground:
            self._restore_drain_limits()
        if not foreground:
            if self._joint_active:
                self.selected=set();self.last_members=None;self.leases.clear()
                self.last_plan=self.policy.num_schedule_count-50
            self._joint_active = False; self.policy.joint_mode = False
            self.mixed_leases.clear(); self._lease_noticed.clear()
            self._release_foreground_plan()
            self.mixed_counts[phase+'_pure_chain_calls'] += 1
            return NativeChainScheduler._schedule(self)
        if not chains:
            self._joint_active = False; self.policy.joint_mode = False
            self.mixed_leases.clear(); self._lease_noticed.clear()
            self._foreground_native = True
            self.mixed_counts[phase+'_foreground_only_native_calls'] += 1
            return SLOScheduler._schedule(self)
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
            curve = forecast_curve(model,meta,stage,views,meta['deadline']-now)
            actions.append({'id':'chain:'+str(cid),'kind':'compound',
                            'members':[g.request_id for g in gs],'curve':curve,
                            'slack':meta['deadline']-now,'arrival':min(g.arrival_time for g in gs),
                            'domain':domain,'scenario_count':len(curve.rows)})
        members = []
        for group in groups:
            seq = group.first_seq
            prefill = group.get_num_uncomputed_tokens() if group.is_prefill() else 0
            native_priority = tuple(SLOPolicy.get_priority(self.policy,group))
            members.append({'request':group.request_id,'prompt':seq.get_prompt_len(),
                            'generated':seq.get_output_len(),'max_tokens':group.sampling_params.max_tokens,
                            'prefill_tokens':prefill,'native_priority':native_priority,'arrival':group.arrival_time})
            if not is_chain(group):
                context = (group.request_type.value,REWARD.count(str(group.prompt),seq.get_prompt_len()),
                           seq.get_output_len(),group.request_info.output_len,group.sampling_params.max_tokens,
                           prefill,max(0.,time.time()-group.arrival_time),group.request_info.slo_constraint)
                actions.append({'id':'request:'+group.request_id,'kind':'foreground',
                                'members':[group.request_id],'foreground':context,
                                'penalty':self.policy.penalty_factor,'arrival':group.arrival_time,
                                'domain':'marginal_native_deadlines'})
        return actions,members

    def _schedule_joint(self):
        self.policy.update_schedule_count();self.free_finished_seq_groups()
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
        reasons=[]
        if not self._joint_active:reasons.append('mixed_entry')
        if live-self._joint_seen:reasons.append('new_request_or_stage')
        if self._joint_seen-live:reasons.append('completion_or_abort')
        if predictions!=self._joint_predictions:reasons.append('prediction_revision')
        if self._joint_deadlines is not None and deadlines!=self._joint_deadlines:reasons.append('deadline_or_membership_change')
        if expired-self._lease_noticed:reasons.append('continuity_quantum_expired')
        if tick-self.last_plan>=50:reasons.append('periodic')
        self._joint_active=True
        if reasons:
            self._planned_this_step=True;started=time.perf_counter()
            actions,members=self._actions(groups,started)
            unit=max(.001,min(1.,self.policy._get_interval_time()))
            running={g.request_id for g in self.running}
            capacity=self.block_manager.num_total_gpu_blocks-self.block_manager.watermark_blocks
            costs=self.cost_model.snapshot(unit,max(1,len(running)))
            self.selected,detail=plan(actions,members,running,capacity,self.scheduler_config.max_num_seqs,
                                      unit,self.cache_config.block_size,costs=costs,
                                      previous=self.selected or (),lease_remaining=remaining,
                                      candidate_multiplier=max(1,self.top_k_selection))
            self.policy.draining=self.selected is None
            self.policy.mixed_priorities={}
            for a in detail.get('actions',[]):
                for rid in a['members']:
                    self.policy.mixed_priorities[rid]=(0 if a['expected_credit']>0 else 1,-a['density'],a['arrival'])
            previous_leases=self.mixed_leases
            self.mixed_leases={g.request_id:previous_leases.get(g.request_id,g.first_seq.get_output_len())
                               for g in groups if self.selected is not None and g.request_id in self.selected}
            self._lease_noticed=expired & self.mixed_leases.keys()
            self.last_plan=tick;self._joint_seen=live;self._joint_predictions=predictions;self._joint_deadlines=deadlines
            self.mixed_counts['joint_plans']+=1
            self.mixed_counts['event_plans']+='new_request_or_stage'in reasons
            self.mixed_counts['backfill_members']+=len(detail['backfill'])
            self.mixed_counts['planned_evictions']+=len(detail['evicted'])
            self.mixed_counts['compute_filtered_plans']+=self.selected is not None and len(self.selected)<len(live)
            event={'tick':tick,'monotonic':started,'reasons':reasons,'selected':sorted(self.selected)if self.selected is not None else None,
                   'running_before':sorted(running),'physical_candidates':len(live),'capacity':capacity,
                   'slots':self.scheduler_config.max_num_seqs,'detail':detail,
                   'plan_seconds':time.perf_counter()-started}
            self.last_joint_summary=event
            if len(self.plan_events)==self.plan_events.maxlen and self.decision_sink is None:self.plan_events_dropped+=1
            self.plan_events.append(event)
            if self.decision_sink:self.decision_sink(event)
        selected=set(live)if self.policy.draining else set(self.selected)&live
        self.last_members=live;self.last_recorded_selected=frozenset(selected)
        swaps=[];preemptions=0
        for group in sorted(list(self.running),key=self.policy.get_priority,reverse=True):
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
