"""v43: chain-deadline admission on native vLLM paged KV/executor.

The original JITServe source is never edited. This module is installed only
inside C's process. It keeps native token execution, stopping and block APIs.
"""
from collections import deque
import time

from vllm.core.scheduler import PreemptionMode, SchedulerOutputs
from jitserve.scheduler.policy import SLOPolicy
from jitserve.scheduler.slo_scheduler import SLOScheduler
from deadline_reservation import Member, reserve


class ChainDeadlinePolicy(SLOPolicy):
    version = 'v43_native_chain_deadline_reservation'

    def get_priority(self, group):
        deadline = group.request_info.chain_deadline
        return (deadline < time.perf_counter(), deadline, group.arrival_time)


class NativeChainScheduler(SLOScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = ChainDeadlinePolicy(schedule_interval=50, penalty_factor=100)
        self.reservation_events = []
        self._last_selected = None

    def _schedule(self):
        self.policy.update_schedule_count()
        self.free_finished_seq_groups()
        groups = list(self.running) + list(self.waiting) + list(self.swapped)
        now = time.perf_counter()
        members = [Member(g.request_id, g.collection_id,
                          g.request_info.chain_deadline, g.arrival_time,
                          g.first_seq.get_prompt_len(),
                          g.first_seq.get_output_len(),
                          max(0, g.request_info.output_len),
                          g.sampling_params.max_tokens) for g in groups]
        capacity = self.block_manager.num_total_gpu_blocks - self.block_manager.watermark_blocks
        selected, rows, used = reserve(members, now, capacity,
                                      self.scheduler_config.max_num_seqs,
                                      self.cache_config.block_size)
        signature = frozenset(selected)
        if signature != self._last_selected or self.policy.num_schedule_count % 50 == 0:
            self.reservation_events.append({
                'kind': 'native_chain_reservation', 'monotonic': now,
                'tick': self.policy.num_schedule_count, 'rows': rows,
                'selected': sorted(selected), 'reserved_blocks': used,
                'capacity': capacity, 'free_gpu': self.block_manager.get_num_free_gpu_blocks(),
                'free_cpu': self.block_manager.get_num_free_cpu_blocks()})
            self._last_selected = signature

        # Give newly ready urgent chains room before scheduling native running
        # work. Finish physical swap-out before any swap-in can reuse blocks.
        swaps, preemptions = [], 0
        for group in sorted(list(self.running), key=self.policy.get_priority, reverse=True):
            if group.request_id in selected or preemptions >= self.max_swaps_per_iter:
                continue
            if not self.block_manager.can_swap_out(group):
                continue
            mode = self._preempt(group, swaps, PreemptionMode.SWAP)
            self.running.remove(group)
            (self.swapped if mode == PreemptionMode.SWAP else self.waiting).append(group)
            preemptions += 1
        if swaps or preemptions:
            return SchedulerOutputs([], 0, 0, [], swaps, [], [], 0,
                                    len(self.running), preemptions)

        # Hide deferred pending requests only for the base scheduler call;
        # they retain native ownership/status and count as unfinished work.
        hidden_waiting = [g for g in self.waiting if g.request_id not in selected]
        hidden_swapped = [g for g in self.swapped if g.request_id not in selected]
        self.waiting = deque(g for g in self.waiting if g.request_id in selected)
        self.swapped = deque(sorted((g for g in self.swapped if g.request_id in selected),
                                    key=self.policy.get_priority))
        self.running = deque(sorted(self.running, key=self.policy.get_priority))
        try:
            result = self._schedule_chunked_prefill()
        finally:
            self.waiting.extend(hidden_waiting)
            self.swapped.extend(hidden_swapped)
        return result


def install(server):
    import vllm.engine.llm_engine as llm_engine
    llm_engine.SLOScheduler = NativeChainScheduler
    original = server._handle_collective_request

    def collective(info, request_dict, prompt):
        # Served time includes all prior stages and client/tool time; the
        # input deadline remains untouched for audit and offline evaluation.
        stages = request_dict.get('num_stages', 6)
        served = request_dict.get('served_time', 0.0)
        info.chain_deadline = time.perf_counter() + info.slo_constraint[2] * stages - served
        return original(info, request_dict, prompt)
    server._handle_collective_request = collective
