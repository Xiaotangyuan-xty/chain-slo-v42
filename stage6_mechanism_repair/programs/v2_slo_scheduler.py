from collections import deque

from jitserve.scheduler.slo_scheduler import SLOScheduler
from first_step import plan_first_step
from scheduler_recompute import RecomputeInvalidatingScheduler
from vllm.core.scheduler import (PreemptionMode, SchedulingBudget, SchedulerOutputs,
                                 SchedulerPrefillOutputs, SchedulerSwappedInOutputs)
from vllm.core.interfaces import AllocStatus
from vllm.sequence import SequenceStatus


class V2SLOScheduler(RecomputeInvalidatingScheduler, SLOScheduler):
    def __init__(self, scheduler_config, cache_config, lora_config,
                 pipeline_parallel_size, output_proc_callback, v2_policy):
        super().__init__(scheduler_config, cache_config, lora_config,
                         pipeline_parallel_size, output_proc_callback)
        self.policy = v2_policy
        self.v2_ledger = None
        self.compound_events = []

    def _normalize_queues(self):
        normalized = {"running": [], "waiting": [], "swapped": []}
        seen = set()
        for group in list(self.running) + list(self.waiting) + list(self.swapped):
            if id(group) in seen:
                continue
            seen.add(id(group))
            status = group.first_seq.status
            if status == SequenceStatus.SWAPPED:
                normalized["swapped"].append(group)
            elif status == SequenceStatus.WAITING:
                normalized["waiting"].append(group)
            else:
                normalized["running"].append(group)
        self.running = deque(normalized["running"])
        self.waiting = deque(normalized["waiting"])
        self.swapped = deque(normalized["swapped"])

    def schedule(self):
        self._normalize_queues()
        self.policy.running_ids = {group.request_id for group in self.running}
        self.policy.swapped_ids = {group.request_id for group in self.swapped}
        groups = list(self.running) + list(self.waiting) + list(self.swapped)
        self.policy.before_schedule(groups, __import__("time").perf_counter(), self.v2_ledger)
        result = super().schedule()
        selected = [item.seq_group for item in result[1].scheduled_seq_groups]
        unique = []
        seen = set()
        for group in selected:
            if group.request_id not in seen:
                seen.add(group.request_id)
                unique.append(group)
        if len(unique) != len(selected):
            self.compound_events.append({
                "kind": "duplicate_scheduled_dropped",
                "request_ids": [group.request_id for group in selected[len(unique):]],
            })
            result[1].scheduled_seq_groups = unique
        self._normalize_queues()
        report = self.compound_events[-1] if self.compound_events else None
        self.policy.note_selection(unique, __import__("time").perf_counter(), report)
        return result

    def abort_seq_group(self, request_id):
        ids = {request_id} if isinstance(request_id, str) else set(request_id)
        for group in list(self.running) + list(self.waiting) + list(self.swapped):
            if group.request_id in ids:
                self.policy.abort_seq_group(group)
        super().abort_seq_group(request_id)

    def _preempt(self, seq_group, blocks_to_swap_out, preemption_mode=None):
        mode = super()._preempt(seq_group, blocks_to_swap_out, preemption_mode)
        self.policy.note_preemption(seq_group, mode)
        return mode

    def _schedule(self):
        if self.policy.custom_dispatch_active():
            self.policy.update_schedule_count()
            return self._schedule_v2_compound()
        return super()._schedule()

    def _adjust_batch_limits(self, process_list, min_tbt_required, budget):
        self.scheduler_config.max_num_seqs = self.policy.batch_slots
        budget.max_num_seqs = self.policy.batch_slots

    def _schedule_v2_compound(self):
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.policy.batch_slots,
        )
        self.free_finished_seq_groups()
        prefill_result, swapped_in_result, running_result = self._init_preemption_outputs(
            True, SchedulerPrefillOutputs, SchedulerSwappedInOutputs)
        groups = [group for group in list(self.running) + list(self.waiting) + list(self.swapped)
                  if not group.first_seq.is_finished()]
        process_list = self.policy.dispatch_order(groups)
        self._schedule_v2_process_list(process_list, running_result, prefill_result,
                                       swapped_in_result, budget, True)
        self.waiting.extendleft(running_result.preempted)
        self.running.clear()
        self.running.extend([item.seq_group for item in swapped_in_result.decode_seq_groups])
        self.running.extend([item.seq_group for item in swapped_in_result.prefill_seq_groups])
        self.running.extend([item.seq_group for item in running_result.decode_seq_groups])
        self.running.extend([item.seq_group for item in running_result.prefill_seq_groups])
        self.running.extend([item.seq_group for item in prefill_result.seq_groups])
        self.running.extend(running_result.leftover_running)
        abort_seqs = self._stop_dull_scheduling()
        self.swapped.extend(running_result.swapped_out)
        return SchedulerOutputs(
            scheduled_seq_groups=(prefill_result.seq_groups
                                  + running_result.prefill_seq_groups
                                  + swapped_in_result.prefill_seq_groups
                                  + running_result.decode_seq_groups
                                  + swapped_in_result.decode_seq_groups),
            num_prefill_groups=(len(prefill_result.seq_groups)
                                + len(swapped_in_result.prefill_seq_groups)
                                + len(running_result.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in_result.blocks_to_swap_in,
            blocks_to_swap_out=running_result.blocks_to_swap_out,
            blocks_to_copy=running_result.blocks_to_copy + swapped_in_result.blocks_to_copy,
            ignored_seq_groups=(prefill_result.ignored_seq_groups
                                + swapped_in_result.infeasible_seq_groups + abort_seqs),
            num_lookahead_slots=running_result.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=len(running_result.preempted) + len(running_result.swapped_out),
        )

    def _compound_units(self, process_list, include_missing=True):
        all_groups = list(self.running) + list(self.waiting) + list(self.swapped)
        grouped = {}
        for group in all_groups:
            if not group.first_seq.is_finished():
                grouped.setdefault((group.collection_id, group.request_info.stage_id), []).append(group)
        units = []
        seen = set()
        for group in process_list:
            key = (group.collection_id, group.request_info.stage_id)
            if key in seen or group.first_seq.is_finished():
                continue
            seen.add(key)
            if include_missing:
                members = [member for member in grouped.get(key, [])
                           if member in process_list or member in all_groups]
            else:
                members = [member for member in grouped.get(key, []) if member in process_list]
            members = [member for member in members if not member.first_seq.is_finished()]
            if members:
                units.append(members)
        return units

    def _append_blocks_required(self, group, lookahead):
        required = 0
        for seq in group.get_seqs(status=SequenceStatus.RUNNING):
            table = self.block_manager.block_tables[seq.seq_id]
            required += table.get_num_blocks_touched_by_append_slots(
                table.get_unseen_token_ids(seq.get_token_ids()),
                num_lookahead_slots=lookahead)
        return required

    def _resident_blocks(self, group):
        return self.block_manager.compute_running_seq_group_blocks([group])

    def _resource_snapshot(self, groups):
        units = {}
        ledger = self.v2_ledger.view()
        running_order = {group.request_id: index for index, group in enumerate(self.running)}
        for unit in self._compound_units(groups):
            key = f"{unit[0].collection_id}:stage:{unit[0].request_info.stage_id}"
            budget = SchedulingBudget(
                token_budget=self.scheduler_config.max_num_batched_tokens,
                max_num_seqs=self.policy.batch_slots)
            resources, reason = self._unit_resources(unit, budget, True)
            tokens, seqs, required = resources if resources is not None else (0, 0, None)
            running = [group for group in unit if group in self.running]
            statuses = {group.first_seq.status.name for group in unit}
            entries = [ledger.get(group.request_id) for group in unit]
            units[key] = {
                "tokens": tokens,
                "slots": seqs,
                "required_blocks": required,
                "resident_blocks": sum(self._resident_blocks(group) for group in running),
                "reason": reason,
                "queue_rank": min((running_order[group.request_id] for group in running),
                                  default=len(running_order) + len(units)),
                "status": statuses.pop() if len(statuses) == 1 else "MIXED",
                "protected": any(self.policy.residency_protected(group) for group in running),
                "decode_progress": min((self.policy.decode_progress.get(group.request_id, 0)
                                        for group in running), default=0),
                "gpu_tokens": sum(entry.gpu for entry in entries),
                "host_tokens": sum(entry.host for entry in entries),
                "pending_tokens": sum(entry.pending for entry in entries),
            }
        watermark = (self.block_manager.num_total_gpu_blocks
                     - self.block_manager.get_max_gpu_num_blocks_number())
        return {
            "units": units,
            "free_kv_blocks": self.block_manager.get_num_free_gpu_blocks(),
            "watermark_blocks": watermark,
            "block_size": self.cache_config.block_size,
            "max_swaps_per_iter": self.max_swaps_per_iter,
            "token_budget": self.scheduler_config.max_num_batched_tokens,
        }

    def _unit_resources(self, unit, budget, enable_chunking):
        tokens = 0
        seqs = 0
        blocks = 0
        for group in unit:
            if group in self.waiting:
                status = SequenceStatus.WAITING
            elif group in self.swapped:
                status = SequenceStatus.SWAPPED
            elif group in self.running:
                status = SequenceStatus.RUNNING
            else:
                return None, "missing_from_queues"
            num_tokens = self._get_num_new_tokens(group, status, enable_chunking, budget)
            if num_tokens <= 0:
                return None, "no_tokens"
            if status == SequenceStatus.WAITING and num_tokens > self._get_prompt_limit(group):
                return None, "prompt_limit"
            lookahead = self._get_num_lookahead_slots(group.is_prefill(), enable_chunking)
            if status == SequenceStatus.WAITING:
                if self.block_manager.can_allocate(group, lookahead) == AllocStatus.NEVER:
                    return None, "allocation_never"
                blocks += self.block_manager.get_num_blocks_number(group, status, lookahead)
            elif status == SequenceStatus.SWAPPED:
                if self.block_manager.can_swap_in(group, lookahead) == AllocStatus.NEVER:
                    return None, "swap_in_never"
                blocks += self.block_manager.get_num_blocks_number(group, status, lookahead)
            else:
                blocks += self._append_blocks_required(group, lookahead)
            tokens += num_tokens
            seqs += group.get_max_num_running_seqs()
        return (tokens, seqs, blocks), None

    def _schedule_v2_accepted(self, process_list, running_result,
                              prefill_result, swapped_in_result, budget,
                              enable_chunking):
        num_gpu_blocks_number = self.block_manager.get_max_gpu_num_blocks_number()
        waiting_scheduled = []
        swapped_scheduled = []
        running_scheduled = []
        stop_allocate = False
        stop_preempt = False
        force_preemption_count = 0

        for seq_group in process_list:
            seq_group_is_prefill = seq_group.is_prefill()
            if seq_group in self.waiting:
                seq_group_status = SequenceStatus.WAITING
            elif seq_group in self.swapped:
                seq_group_status = SequenceStatus.SWAPPED
            elif seq_group in self.running:
                seq_group_status = SequenceStatus.RUNNING
            else:
                continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(
                seq_group, seq_group_status, enable_chunking, budget)
            not_update = False
            if num_new_tokens == 0:
                if len(seq_group.seqs) == 0:
                    not_update = True
                elif seq_group.get_num_uncomputed_tokens() == 0:
                    not_update = True
                else:
                    stop_allocate = True

            if seq_group_status == SequenceStatus.WAITING:
                prompt_limit = self._get_prompt_limit(seq_group)
                if num_new_tokens > prompt_limit:
                    for seq in seq_group.get_seqs(SequenceStatus.WAITING):
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    self.waiting.remove(seq_group)
                    prefill_result.ignored_seq_groups.append(seq_group)
                    continue

            num_lookahead_slots = self._get_num_lookahead_slots(
                seq_group_is_prefill, enable_chunking)
            num_blocks_number = self.block_manager.get_num_blocks_number(
                seq_group, seq_group_status, num_lookahead_slots)

            if (not stop_allocate and num_new_tokens > 0
                    and budget.can_schedule(num_new_tokens=num_new_tokens,
                                            num_new_seqs=num_new_seqs)
                    and num_blocks_number <= num_gpu_blocks_number):
                if seq_group_status == SequenceStatus.WAITING and stop_preempt:
                    continue
                budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
                budget.add_num_seqs(seq_group.request_id, num_new_seqs)
                num_gpu_blocks_number -= num_blocks_number
                if seq_group_status == SequenceStatus.RUNNING:
                    running_scheduled.append((seq_group, num_new_tokens, num_lookahead_slots))
                elif seq_group_status == SequenceStatus.WAITING:
                    if not stop_preempt:
                        waiting_scheduled.append((seq_group, num_new_tokens, num_lookahead_slots))
                else:
                    swapped_scheduled.append((seq_group, num_new_tokens, num_lookahead_slots))
                continue
            elif not not_update:
                stop_allocate = True

            if seq_group_status == SequenceStatus.RUNNING and num_new_tokens > 0:
                if seq_group.is_finished():
                    self._free_finished_seq_group(seq_group)
                    continue
                preempt_mode = self._preempt(
                    seq_group, running_result.blocks_to_swap_out, PreemptionMode.SWAP)
                if preempt_mode == PreemptionMode.SWAP:
                    running_result.swapped_out.append(seq_group)
                elif preempt_mode == PreemptionMode.RECOMPUTE:
                    running_result.preempted.append(seq_group)
                force_preemption_count += 1

        for swapped_seq, num_new_tokens, num_lookahead_slots in swapped_scheduled:
            alloc_status = self.block_manager.can_swap_in(swapped_seq, num_lookahead_slots)
            if alloc_status != AllocStatus.OK:
                continue
            self.swapped.remove(swapped_seq)
            swapped_result = self._schedule_slo_swapped(
                swapped_seq, num_lookahead_slots, enable_chunking)
            if swapped_result.is_prefill:
                swapped_in_result.prefill_seq_groups.append(swapped_result.seq_group)
            else:
                swapped_in_result.decode_seq_groups.append(swapped_result.seq_group)
            swapped_in_result.blocks_to_swap_in.extend(swapped_result.blocks_to_swap_in)
            swapped_in_result.blocks_to_copy.extend(swapped_result.blocks_to_copy)

        for running_seq, num_new_tokens, num_lookahead_slots in running_scheduled:
            if not self._can_append_slots(running_seq, enable_chunking):
                running_result.leftover_running.append(running_seq)
                continue
            scheduled_seq_group = self._scheduled_seq_group_cache[self.cache_id].get_object()
            scheduled_seq_group.seq_group = running_seq
            if running_seq.is_prefill():
                scheduled_seq_group.token_chunk_size = num_new_tokens
                running_result.prefill_seq_groups.append(scheduled_seq_group)
                running_result.prefill_seq_groups_list.append(running_seq)
            else:
                scheduled_seq_group.token_chunk_size = 1
                running_result.decode_seq_groups.append(scheduled_seq_group)
                running_result.decode_seq_groups_list.append(running_seq)
            self._append_slots(running_seq, running_result.blocks_to_copy, enable_chunking)

        for waiting_seq, num_new_tokens, num_lookahead_slots in waiting_scheduled:
            remain_token_budget = budget.remaining_token_budget()
            if remain_token_budget > 0:
                more_new_tokens = 0
                for seq in waiting_seq.get_seqs(status=SequenceStatus.WAITING):
                    more_new_tokens += seq.get_num_new_tokens()
                more_new_tokens -= num_new_tokens
                add_tokens = min(more_new_tokens, remain_token_budget)
                num_new_tokens += add_tokens
                budget._num_batched_tokens += add_tokens

            can_allocate = self.block_manager.can_allocate(
                waiting_seq, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.NEVER:
                for seq in waiting_seq.get_seqs(SequenceStatus.WAITING):
                    seq.status = SequenceStatus.FINISHED_IGNORED
                prefill_result.ignored_seq_groups.append(waiting_seq)
                continue
            elif can_allocate == AllocStatus.LATER:
                continue

            self.waiting.remove(waiting_seq)
            waiting_result = self._schedule_slo_prefills(
                waiting_seq, num_new_tokens, num_lookahead_slots, enable_chunking)
            prefill_result.seq_groups.append(waiting_result.seq_group)

        return force_preemption_count

    def _schedule_v2_process_list(self, process_list, running_result,
                                  prefill_result, swapped_in_result, budget,
                                  enable_chunking):
        units = self._compound_units(process_list, include_missing=True)
        initial_free = self.block_manager.get_num_free_gpu_blocks()
        watermark = (self.block_manager.num_total_gpu_blocks
                     - self.block_manager.get_max_gpu_num_blocks_number())
        initial_available = initial_free - watermark
        policy_preemption = self.policy.policy_preemption_allowed()
        planner_units = []
        for unit in units:
            resources, reason = self._unit_resources(unit, budget, enable_chunking)
            tokens, seqs, required = resources if resources is not None else (0, 0, None)
            running = [group for group in unit if group in self.running]
            planner_units.append({
                "ids": [group.request_id for group in unit],
                "groups": unit,
                "tokens": tokens,
                "slots": seqs,
                "required_blocks": required,
                "reason": reason,
                "running": running,
                "resident_blocks": sum(self._resident_blocks(group) for group in running),
                "protected": any(self.policy.residency_protected(group) for group in running),
            })
        plan = plan_first_step(planner_units, initial_available,
                               self.policy.batch_slots, self.max_swaps_per_iter,
                               policy_preemption, budget.remaining_token_budget())
        accepted = [entry["groups"] for entry in plan["accepted"]]
        accepted_ids = {id(entry) for entry in plan["accepted"]}
        skipped = [(entry["groups"], entry["reason"] or "resource")
                   for entry in planner_units if id(entry) not in accepted_ids]
        evictions = plan["evictions"]
        deferred = plan["deferred"]
        unit_accounts = plan["unit_accounts"]
        running_result.leftover_running.extend(plan["leftover_running"])
        simulated_tokens = sum(entry["tokens"] for entry in plan["accepted"])
        simulated_seqs = plan["used_slots"]
        simulated_blocks = plan["used_blocks"]
        planned_freed = plan["planned_freed_blocks"]
        available = plan["remaining_available_blocks"]
        preemptions = []

        force_preemption_count = 0
        for unit, running, reason in evictions:
            for group in running:
                mode = self._preempt(group, running_result.blocks_to_swap_out,
                                     PreemptionMode.SWAP)
                if mode == PreemptionMode.SWAP:
                    running_result.swapped_out.append(group)
                elif mode == PreemptionMode.RECOMPUTE:
                    running_result.preempted.append(group)
                force_preemption_count += 1
                cause = ("model" if self.policy.model_action
                         else "forced" if self.policy.forced is not None else "resource")
                preemptions.append({"request_id": group.request_id, "cause": cause,
                                    "trigger": reason, "mode": str(mode)})

        if accepted:
            force_preemption_count += self._schedule_v2_accepted(
                [group for unit in accepted for group in unit],
                running_result, prefill_result, swapped_in_result,
                budget, enable_chunking)
            selected = {
                item.seq_group.request_id
                for item in (list(running_result.decode_seq_groups)
                             + list(running_result.prefill_seq_groups)
                             + list(prefill_result.seq_groups)
                             + list(swapped_in_result.decode_seq_groups)
                             + list(swapped_in_result.prefill_seq_groups))
            }
            for unit in accepted:
                expected = {group.request_id for group in unit}
                executed = expected & selected
                if executed != expected:
                    raise RuntimeError(
                        f"Atomic admission/execution mismatch: executed={sorted(executed)} "
                        f"missing={sorted(expected - executed)}")
        if available != initial_available + planned_freed - simulated_blocks:
            raise RuntimeError("KV planning conservation failed")
        self.compound_events.append({
            "units": [[group.request_id for group in unit] for unit in units],
            "accepted": [[group.request_id for group in unit] for unit in accepted],
            "skipped": [([group.request_id for group in unit], reason) for unit, reason in skipped],
            "simulated": [simulated_tokens, simulated_seqs, simulated_blocks],
            "initial_free_blocks": initial_free,
            "watermark_blocks": watermark,
            "initial_available_blocks": initial_available,
            "planned_freed_blocks": planned_freed,
            "remaining_available_blocks": available,
            "unit_accounts": unit_accounts,
            "policy_preemption": policy_preemption,
            "preemptions": preemptions,
            "deferred": deferred,
        })
        return force_preemption_count


def install_v2_slo(engine, perf, pred, stage_counts, candidate_selector=None):
    old = engine.engine.scheduler[0]
    policy = V2CompoundPolicy(perf, pred, stage_counts, candidate_selector,
                              old.scheduler_config.max_num_seqs)
    scheduler = V2SLOScheduler(
        old.scheduler_config,
        old.cache_config,
        old.lora_config,
        1,
        None,
        v2_policy=policy,
    )
    scheduler.v2_ledger = engine.engine.model_executor.backend.ledger
    scheduler.recompute_invalidate = engine.engine.model_executor.invalidate_request
    policy.resource_provider = scheduler._resource_snapshot
    engine.engine.scheduler[0] = scheduler
    return scheduler


from v2_compound_policy import V2CompoundPolicy
