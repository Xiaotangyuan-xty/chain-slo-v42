"""Resident-preserving mixed coordination; the published pure dispatch is intact.

No trace labels, future prompts, target output lengths or observation horizon
are used. All physical allocation/preemption remains in the original backend.
"""
from collections import Counter, deque
import math
import statistics
import time

from mixed_joint import MixedScheduler as OriginalScheduler, is_chain
from pure_core_v47 import leave_pure, schedule_pure
from native_chain import MODEL
from jitserve.scheduler.policy import SLOPolicy
from vllm.core.scheduler import PreemptionMode, SchedulerOutputs
from service_model import member_blocks, bucket

VERSION = 'v47max_mixed_continuity_v1'
VARIANT = 'v47max_resident_marginal_admission_v1'


def marginal_foreground(policy, group, service, now_wall):
    """Comparable future credit/service; settled latency credit is not resold."""
    reward, remaining = SLOPolicy.reward_estimation(policy, group)
    if group.request_type.value == 0:
        reward = max(0., reward - max(0., group.concord_metrics.service_gain))
    generated = group.first_seq.get_output_len()
    predicted = max(1., group.request_info.output_len - generated)
    view = {'prompt': group.first_seq.get_prompt_len(), 'generated': generated,
            'prefill_tokens': group.get_num_uncomputed_tokens() if group.is_prefill() else 0,
            'resident': group in policy.physical_running,
            'swapped': group in policy.physical_swapped}
    recovery = service.recovery(view)
    cost = max(.001, remaining + recovery)
    age = max(0., now_wall - group.arrival_time)
    if group.request_type.value == 0:
        slack = group.TTFT_constraint + generated * group.TBT_constraint - age
        urgent = generated == 0 and 0 < slack <= max(.1, 2 * service.last_round + recovery)
    else:
        slack = group.deadline - age
        urgent = False
    return {'score': max(0., reward) / cost, 'deadline_slack': slack,
            'urgent': urgent, 'predicted_remaining': predicted, 'cost_seconds': cost}


class MixedScheduler(OriginalScheduler):
    """Override mixed dispatch only, without replacing inherited pure helpers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy.version = VERSION
        self.policy.mixed_variant = VARIANT
        self.max_counts = Counter()
        self._max_goals = {}
        self._max_signature = None
        self._max_next_plan = 0.
        self._max_limit = self.base_max_num_seqs
        self._max_limit_due = 0.
        self._max_policy = None

    def _schedule(self):
        queued = list(self.running) + list(self.waiting) + list(self.swapped)
        # Same operations and same entry as published v47pro3. Do not invoke
        # _begin, a new policy, new preemption, or the new cost/admission here.
        if all(is_chain(g) for g in queued):
            return schedule_pure(self)
        return self._max_schedule()

    def _max_dynamic_limit(self, now):
        if now >= self._max_limit_due:
            n = max(1, len(self.running))
            context = sum(g.first_seq.get_len() for g in self.running)
            bins = [(key, values) for key, values in self.cost_model.samples.items()
                    if key[0] == bucket(n) and key[2] == 0 and not key[3]]
            if bins:
                _, values = min(bins, key=lambda kv: abs(kv[0][1]-bucket(context/n/256)))
                samples = list(values)[-16:]
            else:
                samples = []
            unit = statistics.median(samples) if samples else self.cost_model.last_round
            tbts = [g.TBT_constraint for g in self.running if not is_chain(g)
                    and g.request_type.value == 0 and g.TBT_constraint > 0]
            target = min(tbts, default=.1)
            if unit < .7 * target:
                self._max_limit += 2
            elif unit > .95 * target:
                self._max_limit -= 2
            self._max_limit = min(64, max(8, self._max_limit))
            self._max_limit_due = now + .5
        # A decreasing admission limit never evicts a resident by itself.
        return min(128, max(len(self.running), self._max_limit))

    def _max_refresh_goals(self, chains, now, capacity, limit):
        grouped = {}
        for g in chains:
            grouped.setdefault(g.collection_id, []).append(g)
        for cid in list(self._max_goals):
            if cid not in grouped:
                del self._max_goals[cid]
        dirty = []
        signatures = {}
        for cid, members in grouped.items():
            if all(g in self.running for g in members):
                continue
            signature = tuple(sorted((g.request_id, g.request_info.stage_id,
                getattr(g.request_info, 'prediction_revision', 0), g.request_info.output_len,
                g.first_seq.get_output_len() // 32, g.is_prefill()) for g in members))
            signatures[cid] = signature
            cached = self._max_goals.get(cid)
            if cached is None or cached[0] != signature or now >= cached[1] + .5:
                dirty.extend(members)
        values = self.policy.goals(dirty, now, capacity, limit,
            self.cache_config.block_size, self.scheduler_config.max_num_batched_tokens) if dirty else []
        for goal in values:
            self._max_goals[goal['chain']] = (signatures[goal['chain']], now, goal)
        goals = [self._max_goals[cid][2] for cid, members in grouped.items()
                 if any(g not in self.running for g in members)]
        self.policy.priorities = {}
        for goal in goals:
            for member in goal['members']:
                self.policy.priorities[member['request']] = (
                    0 if goal['probability'] > 0 else 1, -goal['score'],
                    goal['deadline'], member['arrival'])
        self.max_counts['dirty_root_forecasts'] += len(values)
        self.max_counts['cached_root_forecasts'] += len(goals) - len(values)
        return goals

    def _max_plan(self, groups, now, limit):
        capacity = self.block_manager.num_total_gpu_blocks - self.block_manager.watermark_blocks
        size = self.cache_config.block_size
        chains = [g for g in groups if is_chain(g)]
        fg = [g for g in groups if not is_chain(g)]
        goals = self._max_refresh_goals(chains, now, capacity, limit)
        byid = {g.request_id: g for g in groups}
        views = {rid: {**self._view(g), 'remaining': max(1.,
            g.request_info.output_len - g.first_seq.get_output_len())}
            for rid, g in byid.items()}
        selected = {g.request_id for g in self.running}
        used = sum(member_blocks(views[r], size) for r in selected)
        fg_values = {g.request_id: marginal_foreground(self.policy, g, self.cost_model, time.time()) for g in fg}
        items = []
        for g in fg:
            if g.request_id not in selected:
                value = fg_values[g.request_id]
                items.append((not value['urgent'], -value['score'], g.arrival_time, 'fg', g))
        for goal in goals:
            if goal['deadline'] > now and goal['probability'] > 0:
                items.append((True, -goal['score'], goal['arrival'], 'chain', goal))
        items.sort(key=lambda item: item[:3])
        admitted = []; repriced = []; deferred = []
        for _, _, _, kind, value in items:
            if kind == 'fg':
                candidates = [value]
                if fg_values[value.request_id]['score'] <= 1e-9:
                    continue
            else:
                candidates = [byid[m['request']] for m in value['members']
                              if m['request'] in byid and m['request'] not in selected]
            chosen = []
            charge = 0
            for g in sorted(candidates, key=lambda g: g.arrival_time):
                b = member_blocks(views[g.request_id], size)
                if len(selected) + len(chosen) < limit and used + charge + b <= capacity:
                    chosen.append(g); charge += b
            if not chosen:
                continue
            if kind == 'chain':
                own = {m['request'] for m in value['members']}
                first = own & (selected | {g.request_id for g in chosen})
                state = value.get('_forecast')
                if state is not None:
                    state['slack'] = value['deadline'] - now
                    state['slots'] = limit
                background = [views[r] for r in selected - own]
                actual = MODEL.revalue(value, first, background)
                repriced.append({'chain': value['chain'], 'probability': actual['probability'],
                    'score': actual['score'], 'selected_members': sorted(first)})
                if actual['probability'] <= 0:
                    self.max_counts['infeasible_mixed_wave_deferred'] += 1
                    continue
                best_fg = max((v['score'] for rid, v in fg_values.items() if rid not in selected), default=0.)
                if actual['score'] < best_fg:
                    deferred.append(value)
                    continue
            selected.update(g.request_id for g in chosen)
            admitted.extend(g.request_id for g in chosen); used += charge
        # Foreground may have consumed fewer slots than its initial rank implied.
        # Reconsider deferred stages against the now-complete resident/FG pool.
        for goal in deferred:
            chosen = []; charge = 0
            for member in goal['members']:
                rid = member['request']
                if rid not in byid or rid in selected:
                    continue
                b = member_blocks(views[rid], size)
                if len(selected) + len(chosen) < limit and used + charge + b <= capacity:
                    chosen.append(rid); charge += b
            if not chosen:
                continue
            own = {m['request'] for m in goal['members']}
            first = own & (selected | set(chosen))
            actual = MODEL.revalue(goal, first, [views[r] for r in selected - own])
            if actual['probability'] > 0:
                selected.update(chosen); used += charge; admitted.extend(chosen)
                repriced.append({'chain': goal['chain'], 'probability': actual['probability'],
                    'score': actual['score'], 'selected_members': sorted(first)})
        # Fill only when no positive work was admitted/running. All requests
        # remain required; expired work is drained, never discarded.
        if not selected:
            for g in sorted(groups, key=lambda g: (g.get_num_uncomputed_tokens(), g.arrival_time)):
                b = member_blocks(views[g.request_id], size)
                if len(selected) < limit and used + b <= capacity:
                    selected.add(g.request_id); used += b
            if not selected and groups:
                selected.add(min(groups, key=lambda g: g.arrival_time).request_id)
            self.max_counts['required_drain_admissions'] += len(selected)
        self.selected = selected
        self.desired_selected = set(selected)
        self.rows = repriced
        self.policy.draining = False
        self.last_plan = self.policy.num_schedule_count
        self._planned_this_step = True
        self.decision_id += 1
        self.v47_counts['plans'] += 1
        self.max_counts['mixed_plans'] += 1
        self.max_counts['admitted_groups'] += len(admitted)
        self.last_joint_summary = {'limit': limit, 'resident': len(self.running),
            'selected': len(selected), 'newly_admitted': len(admitted),
            'foreground_selected': sum(not is_chain(byid[r]) for r in selected),
            'chain_goals': len(goals), 'actual_mixed_wave_reprices': len(repriced)}
        self._emit({'kind': 'max_plan', 'mode': self.mode, 'selected': sorted(selected),
                    'admitted': admitted, 'limit': limit, 'rows': repriced,
                    'resident_ids': sorted(g.request_id for g in self.running)})

    def _max_urgent_eviction(self, groups, now, limit):
        """Only blocked, positive-value foreground may replace expired work."""
        pending = [g for g in groups if not is_chain(g) and g not in self.running]
        targets = []
        for g in pending:
            value = marginal_foreground(self.policy, g, self.cost_model, time.time())
            from vllm.core.interfaces import AllocStatus
            blocked = len(self.running) >= limit
            if g in self.swapped:
                blocked = blocked or self.block_manager.can_swap_in(g, 0) != AllocStatus.OK
            else:
                blocked = blocked or self.block_manager.can_allocate(g) != AllocStatus.OK
            if value['urgent'] and value['score'] > 0 and blocked:
                targets.append(g)
        if not targets:
            return [], 0
        wall = time.time()
        victims = [g for g in self.running if self._deadline(g, now, wall) <= now
                   and not (not is_chain(g) and g.request_type.value == 0
                            and g.first_seq.get_output_len() > 0)]
        swaps = []; count = 0
        # Reclaim at most one expired resident per blocked arrival, not a
        # periodic resident cleanup. Physical native preemption is unchanged.
        for victim in sorted(victims, key=self.policy.get_priority, reverse=True)[:min(len(targets), self.max_swaps_per_iter)]:
            mode = self._preempt(victim, swaps, self._choose_preemption(victim))
            self.running.remove(victim)
            (self.swapped if mode == PreemptionMode.SWAP else self.waiting).append(victim)
            count += 1
        self.max_counts['blocked_urgent_expired_replacements'] += count
        return swaps, count

    def _max_execute(self, selected, limit, preemptions=0):
        # Planning changes admission, not the resident decode set.
        allowed = set(selected) | {g.request_id for g in self.running}
        self.selected = allowed; self.last_recorded_selected = frozenset(allowed)
        self._execution_flags.update(candidate_filter=True, mode=self.mode,
            mode_epoch=self.mode_epoch, decision_id=self.decision_id)
        hidden_waiting = [g for g in self.waiting if g.request_id not in allowed]
        hidden_swapped = [g for g in self.swapped if g.request_id not in allowed]
        self.waiting = deque(sorted((g for g in self.waiting if g.request_id in allowed), key=self.policy.get_priority))
        self.swapped = deque(sorted((g for g in self.swapped if g.request_id in allowed), key=self.policy.get_priority))
        self.running = deque(sorted(self.running, key=self.policy.get_priority))
        try:
            output = self._chunked(max(limit, len(self.running)))
            output.preempted += preemptions
            return output
        finally:
            self.waiting.extend(hidden_waiting); self.swapped.extend(hidden_swapped)

    def _max_schedule(self):
        if self.policy is not self._max_policy:
            self._max_goals.clear()
            self._max_signature = None
            self._max_next_plan = 0.
        leave_pure(self)
        self._max_policy = self.policy
        self.policy.version = VERSION; self.policy.mixed_variant = VARIANT
        groups, now = self._begin()
        chains = [g for g in groups if is_chain(g)]
        mode = 'joint' if chains else 'foreground'
        if self.mode != mode:
            previous = self.mode
            self.mode = mode; self.mode_epoch += 1
            self._emit({'kind': 'mode_change', 'previous': previous, 'mode': mode})
            # Preserve residents, leases and compatible mixed caches across
            # foreground/joint changes. Pure transitions use original resets.
        self.policy.joint_mode = bool(chains)
        self.policy.physical_running = self.running
        self.policy.physical_swapped = self.swapped
        self.mixed_counts[mode + '_calls'] += 1
        limit = self._max_dynamic_limit(now)
        swaps, preemptions = self._max_urgent_eviction(groups, now, limit)
        if swaps:
            self._max_signature = None
            return SchedulerOutputs([], 0, 0, [], swaps, [], [], 0, len(self.running), preemptions)
        signature = (self.mode_epoch, limit,
            tuple(sorted((g.request_id, g.request_info.stage_id,
                getattr(g.request_info, 'prediction_revision', 0), g.request_info.output_len,
                g.first_seq.status, g.is_prefill()) for g in groups)))
        if signature != self._max_signature or now >= self._max_next_plan:
            self._max_plan(groups, now, limit)
            self._max_signature = signature
            self._max_next_plan = now + .2
        live = {g.request_id for g in groups}
        selected = set(self.desired_selected or ()) & live
        return self._max_execute(selected, limit, preemptions)
