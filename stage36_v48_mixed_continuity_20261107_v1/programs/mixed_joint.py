"""v48 mixed continuity; original v47 pure DR scheduling is delegated unchanged."""
from collections import Counter, deque
import time

from jitserve.request_info import RequestType
from jitserve.scheduler.policy import SLOPolicy
from jitserve.scheduler.slo_scheduler import SLOScheduler
from vllm.core.scheduler import PreemptionMode, SchedulerOutputs
from vllm.sequence import SequenceStatus
import native_chain
from native_chain import ChainCompletionPolicy, NativeChainScheduler
from mixed_control import foreground_value, plan
from mixed_forecast import tot_model
from token_reward import REWARD

VERSION = 'v48_mixed_event_continuity_backfill'
VARIANT = 'v48_marginal_credit_resident_first_v1'


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
            meta['v48_application'] = 'tot_native'
            meta['v48_model_key'] = getattr(info, 'v48_model_key', None)
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
        use_domain = any(self.chains[g.collection_id].get('v48_application') == 'tot_native' for g in groups)
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
            if meta.get('v48_application') == 'tot_native':
                calibrated = tot_model(meta.get('v48_model_key'))
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
        self.plan_events = deque(maxlen=4096)
        self.plan_events_dropped = 0

    def _release_foreground_plan(self):
        if self._foreground_native:
            self._pending_swap_out_queue.clear(); self._current_bin_queue.clear()
            self._exec_order = []; self._bins_by_index = []; self._bin_bounds = []
            self._foreground_native = False

    def _schedule(self):
        groups = [g for g in list(self.running)+list(self.waiting)+list(self.swapped) if not g.is_finished()]
        chains = [g for g in groups if is_chain(g)]
        foreground = [g for g in groups if not is_chain(g)]
        phase = 'main' if any(g.request_info.client_id != 900000 for g in groups) else 'warmup'
        if foreground:
            self._restore_drain_limits()
        if not foreground:
            if self._joint_active:
                # Mixed ownership and its priority cache must not defer the
                # first resumed pure plan until an old50-tick boundary.
                self.selected=set();self.last_members=None;self.leases.clear()
                self.last_plan=self.policy.num_schedule_count-50
            self._joint_active = False; self.policy.joint_mode = False
            self._release_foreground_plan()
            self.mixed_counts[phase+'_pure_chain_calls'] += 1
            return NativeChainScheduler._schedule(self)
        if not chains:
            self._joint_active = False; self.policy.joint_mode = False
            self._foreground_native = True
            self.mixed_counts[phase+'_foreground_only_native_calls'] += 1
            return SLOScheduler._schedule(self)
        self.policy.joint_mode = True
        self._release_foreground_plan()
        self.mixed_counts[phase+'_joint_calls'] += 1
        return self._schedule_joint()

    def _actions(self, groups, now):
        unit = max(.01, min(.2, self.policy._get_interval_time()))
        chain_groups = [g for g in groups if is_chain(g)]
        goals = self.policy.goals(chain_groups, now)
        actions, members = [], []
        by_id = {g.request_id:g for g in groups}
        for goal in goals:
            work = max(1., goal.get('remaining_work') or 1.)
            credit = max(0., goal.get('expected_credited_tokens', goal['score']*work))
            probability=goal['probability']
            if goal['deadline']<=now or not goal.get('supported',True):
                credit=0.;probability=0.
            meta = self.policy.chains[goal['chain']]
            prefix = sum(z.get('reward_input',z['declared_input'])+8*z.get('reward_output',z['output'])
                         for stage in meta['observed'].values() for z in stage.values())
            prefix += sum(v['reward_input']+8*v['generated'] for v in goal['members'] if v['generated'])
            current = max(0., goal.get('current_remaining_tokens') or 0.)
            stage_work = max(1., current*len(goal['members'])+
                             sum(v['prefill_tokens'] for v in goal['members'])/256.)
            actions.append({'id':'chain:'+str(goal['chain']), 'kind':'compound',
                            'members':[v['request'] for v in goal['members']],
                            'expected_credit':credit, 'density':credit/work, 'work':work,
                            'stage_work':stage_work, 'probability':probability,
                            'critical_seconds':goal.get('critical_mean_seconds') or 0.,
                            'slack':goal['deadline']-now, 'prefix_at_risk':prefix,
                            'arrival':goal['arrival'], 'domain':goal.get('forecast_domain','DR')})
        for group in groups:
            seq = group.first_seq
            native_priority = tuple(SLOPolicy.get_priority(self.policy, group))
            members.append({'request':group.request_id, 'prompt':seq.get_prompt_len(),
                            'generated':seq.get_output_len(), 'max_tokens':group.sampling_params.max_tokens,
                            'arrival':group.arrival_time, 'native_priority':native_priority})
            if not is_chain(group):
                prompt_credit=REWARD.count(str(group.prompt),seq.get_prompt_len())
                value = foreground_value(group.request_type.value, prompt_credit, seq.get_output_len(),
                                         group.request_info.output_len, group.sampling_params.max_tokens,
                                         group.get_num_uncomputed_tokens() if group.is_prefill() else 0,
                                         max(0., time.time()-group.arrival_time),
                                         group.request_info.slo_constraint, unit,
                                         penalty=self.policy.penalty_factor)
                actions.append({**value, 'id':'request:'+group.request_id,
                                'kind':'foreground', 'members':[group.request_id],
                                'arrival':group.arrival_time, 'domain':'marginal_native_deadlines'})
        for action in actions:
            for rid in action['members']:
                self.policy.mixed_priorities[rid] = (0 if action['expected_credit']>0 else 1,
                                                     -action['density'], by_id[rid].arrival_time)
        return actions, members, unit

    def _schedule_joint(self):
        self.policy.update_schedule_count(); self.free_finished_seq_groups()
        groups = [g for g in list(self.running)+list(self.waiting)+list(self.swapped) if not g.is_finished()]
        live = {g.request_id for g in groups}; tick = self.policy.num_schedule_count
        predictions = tuple(sorted((g.request_id,getattr(g.request_info,'prediction_revision',0)) for g in groups))
        reasons = []
        if not self._joint_active: reasons.append('mixed_entry')
        if live-self._joint_seen: reasons.append('new_request_or_stage')
        if self._joint_seen-live: reasons.append('completion_or_abort')
        if predictions != self._joint_predictions: reasons.append('prediction_revision')
        if tick-self.last_plan >= 50: reasons.append('periodic')
        self._joint_active = True
        if reasons:
            started = time.perf_counter()
            actions,members,unit = self._actions(groups, started)
            running = {g.request_id for g in self.running}
            capacity = self.block_manager.num_total_gpu_blocks-self.block_manager.watermark_blocks
            self.selected, detail = plan(actions,members,running,capacity,
                                         self.scheduler_config.max_num_seqs,unit,self.cache_config.block_size)
            self.policy.draining = self.selected is None or not any(a['expected_credit']>0 for a in actions)
            self.last_plan=tick; self._joint_seen=live; self._joint_predictions=predictions
            self.mixed_counts['joint_plans'] += 1
            self.mixed_counts['backfill_members'] += len(detail['backfill'])
            self.mixed_counts['event_plans'] += 'new_request_or_stage' in reasons
            self.mixed_counts['planned_evictions'] += len(detail['evicted'])
            event={'tick':tick,'monotonic':started,'reasons':reasons,
                   'actions':actions,'selected':sorted(self.selected) if self.selected is not None else None,
                   'capacity':capacity,'slots':self.scheduler_config.max_num_seqs,
                   'running_before':sorted(running),'detail':detail,
                   'plan_seconds':time.perf_counter()-started}
            if len(self.plan_events)==self.plan_events.maxlen: self.plan_events_dropped+=1
            self.plan_events.append(event); self.last_joint_summary=event
        selected = set(live) if self.policy.draining else set(self.selected)&live
        self.last_members=live; self.last_recorded_selected=frozenset(selected)
        swaps=[]; preemptions=0
        for group in sorted(list(self.running),key=self.policy.get_priority,reverse=True):
            if group.request_id in selected or preemptions>=self.max_swaps_per_iter: continue
            mode=PreemptionMode.SWAP if self.block_manager.can_swap_out(group) else PreemptionMode.RECOMPUTE
            mode=self._preempt(group,swaps,mode); self.running.remove(group)
            self.leases.pop(group.request_id,None)
            (self.swapped if mode==PreemptionMode.SWAP else self.waiting).append(group)
            preemptions+=1
        if swaps or preemptions:
            return SchedulerOutputs([],0,0,[],swaps,[],[],0,len(self.running),preemptions)
        hidden_waiting=[g for g in self.waiting if g.request_id not in selected]
        hidden_swapped=[g for g in self.swapped if g.request_id not in selected]
        self.waiting=deque(sorted((g for g in self.waiting if g.request_id in selected),key=self.policy.get_priority))
        self.swapped=deque(sorted((g for g in self.swapped if g.request_id in selected),key=self.policy.get_priority))
        self.running=deque(sorted(self.running,key=self.policy.get_priority))
        try:
            return self._schedule_chunked_prefill()
        finally:
            self.waiting.extend(hidden_waiting); self.swapped.extend(hidden_swapped)
