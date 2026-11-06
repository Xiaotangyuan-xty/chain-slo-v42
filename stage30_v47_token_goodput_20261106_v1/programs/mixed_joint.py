"""v45 plus mixed coexistence; pure-chain scheduling delegates to frozen v45."""
from collections import Counter, deque
import time

from jitserve.request_info import RequestType
from jitserve.scheduler.policy import SLOPolicy
from jitserve.scheduler.slo_scheduler import SLOScheduler
from vllm.core.scheduler import PreemptionMode, SchedulerOutputs
from vllm.sequence import SequenceStatus
from native_chain import ChainCompletionPolicy, NativeChainScheduler
from joint_admission import select_joint

VERSION = 'v47_token_reward_deadline_drain_cached_forecast'
VARIANT = 'v47_joint_token_reward_v1'


def is_chain(group):
    return group.request_type == RequestType.COLLECTIVE


def foreground_view(group):
    seq = group.first_seq
    return {'request': group.request_id, 'prompt': seq.get_prompt_len(),
            'generated': seq.get_output_len(), 'max_tokens': group.sampling_params.max_tokens,
            'arrival': group.arrival_time}


class MixedPolicy(ChainCompletionPolicy):
    version = VERSION
    mixed_variant = VARIANT

    def __init__(self):
        super().__init__()
        self.joint_mode = False

    def add_seq_group(self, seq_group):
        group = seq_group
        if is_chain(group):
            return super().add_seq_group(group)
        return SLOPolicy.add_seq_group(self, group)

    def delete_seq_group(self, seq_group):
        group = seq_group
        if is_chain(group):
            return super().delete_seq_group(group)
        return SLOPolicy.delete_seq_group(self, group)

    def get_priority(self, seq_group):
        group = seq_group
        if is_chain(group):
            priority = super().get_priority(group)
            return (1,) + priority if self.joint_mode else priority
        priority = SLOPolicy.get_priority(self, group)
        return (0,) + tuple(priority) + (group.arrival_time,) if self.joint_mode else priority


class MixedScheduler(NativeChainScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = MixedPolicy()
        self.mixed_counts = Counter()
        self.last_joint_summary = None
        self._foreground_native = False

    def _release_foreground_plan(self):
        # Native bin queues are unexecuted selection plans, not physical KV.
        # Joint/chain ownership must not later execute an obsolete native plan.
        if self._foreground_native:
            self._pending_swap_out_queue.clear()
            self._current_bin_queue.clear()
            self._exec_order = []
            self._bins_by_index = []
            self._bin_bounds = []
            self._foreground_native = False

    def _schedule(self):
        groups = [g for g in list(self.running) + list(self.waiting) + list(self.swapped)
                  if not g.is_finished()]
        chains = [g for g in groups if is_chain(g)]
        foreground = [g for g in groups if not is_chain(g)]
        phase = 'main' if any(g.request_info.client_id != 900000 for g in groups) else 'warmup'
        if foreground:
            self._restore_drain_limits()
        if not foreground:
            self.policy.joint_mode = False
            self._release_foreground_plan()
            self.mixed_counts[phase + '_v45_chain_calls'] += 1
            return NativeChainScheduler._schedule(self)
        if not chains:
            self.policy.joint_mode = False
            self._foreground_native = True
            self.mixed_counts[phase + '_foreground_only_native_calls'] += 1
            return SLOScheduler._schedule(self)
        self.policy.joint_mode = True
        self._release_foreground_plan()
        self.mixed_counts[phase + '_joint_calls'] += 1
        return self._schedule_joint()

    def _schedule_joint(self):
        # This is the v45 loop with only request-type partitioning and joint
        # selection added. The original cadence, triggers, leases and drain stay.
        self.policy.update_schedule_count()
        self.free_finished_seq_groups()
        groups = list(self.running) + list(self.waiting) + list(self.swapped)
        chains = [g for g in groups if is_chain(g)]
        foreground = [g for g in groups if not is_chain(g)]
        live = {g.request_id for g in groups}
        now = time.perf_counter()
        for rid in list(self.leases):
            if rid not in live:
                self.leases.pop(rid, None)
        for g in self.running:
            if is_chain(g):
                self.leases.setdefault(g.request_id, g.first_seq.get_output_len())
        tick = self.policy.num_schedule_count
        sibling = False
        if self.selected:
            selected_chains = {g.collection_id for g in chains if g.request_id in self.selected}
            sibling = any(g.request_id not in (self.last_members or set())
                          and g.collection_id in selected_chains for g in chains)
        replan = (tick - self.last_plan >= 50
                  or (self.selected is not None and not (self.selected & live))
                  or (not self.last_members and bool(live)) or sibling)
        capacity = self.block_manager.num_total_gpu_blocks - self.block_manager.watermark_blocks
        protected = {g.request_id for g in self.running if is_chain(g)
                     and g.request_info.chain_deadline > now
                     and g.first_seq.get_output_len() - self.leases[g.request_id] < 50}
        if replan:
            goals = self.policy.goals(chains, now) if chains else []
            fg_sorted = sorted(foreground, key=lambda g: SLOPolicy.get_priority(self.policy, g))
            self.selected, self.rows, fg_selected = select_joint(
                goals, [foreground_view(g) for g in fg_sorted], capacity,
                self.scheduler_config.max_num_seqs, self.cache_config.block_size, protected)
            self.policy.draining = self.selected is None
            self.last_plan = tick
            self.mixed_counts['joint_plans'] += 1
            self.mixed_counts['joint_plans_with_positive_chain'] += any(g['probability'] > 0 for g in goals)
            chain_ids = {g.request_id for g in chains}
            self.mixed_counts['joint_plans_selecting_both_types'] += bool(
                self.selected is not None and fg_selected and self.selected & chain_ids)
            self.last_joint_summary = {'tick': tick, 'chain_goals': len(goals),
                                       'foreground_selected': len(fg_selected),
                                       'chain_selected': len((self.selected or set()) & chain_ids),
                                       'native_work_conserving_drain': self.policy.draining}
        if self.policy.draining:
            selected = set(live)
        else:
            selected = set(self.selected) & live
            for g in self.running:
                if (is_chain(g) and g.request_id not in selected
                        and g.request_info.chain_deadline > now
                        and g.first_seq.get_output_len() - self.leases[g.request_id] < 50):
                    selected.add(g.request_id)
        # Keep v45's update points without building per-tick decision snapshots.
        if replan or live != self.last_members or frozenset(selected) != self.last_recorded_selected:
            self.last_members = live
            self.last_recorded_selected = frozenset(selected)
        swaps = []
        preemptions = 0
        if not self.policy.draining:
            for g in sorted(list(self.running), key=self.policy.get_priority, reverse=True):
                if g.request_id in selected or preemptions >= self.max_swaps_per_iter:
                    continue
                mode = PreemptionMode.SWAP if self.block_manager.can_swap_out(g) else PreemptionMode.RECOMPUTE
                mode = self._preempt(g, swaps, mode)
                self.running.remove(g)
                self.leases.pop(g.request_id, None)
                (self.swapped if mode == PreemptionMode.SWAP else self.waiting).append(g)
                preemptions += 1
        if swaps or preemptions:
            return SchedulerOutputs([], 0, 0, [], swaps, [], [], 0, len(self.running), preemptions)
        hidden_waiting = [g for g in self.waiting if g.request_id not in selected]
        hidden_swapped = [g for g in self.swapped if g.request_id not in selected]
        self.waiting = deque(sorted((g for g in self.waiting if g.request_id in selected),
                                    key=self.policy.get_priority))
        self.swapped = deque(sorted((g for g in self.swapped if g.request_id in selected),
                                    key=self.policy.get_priority))
        self.running = deque(sorted(self.running, key=self.policy.get_priority))
        try:
            return self._schedule_chunked_prefill()
        finally:
            self.waiting.extend(hidden_waiting)
            self.swapped.extend(hidden_swapped)
