from collections import deque

from jitserve.scheduler.slo_scheduler import SLOScheduler
from first_step import plan_first_step
from scheduler_recompute import HostCapacityDefer, RecomputeInvalidatingScheduler
from vllm.core.scheduler import (PreemptionMode, SchedulingBudget, SchedulerOutputs,
                                 SchedulerRunningOutputs, SchedulerPrefillOutputs,
                                 SchedulerSwappedInOutputs)
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
        self._dispatch_event = None

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
        self._dispatch_event = None
        self.policy.running_ids = {group.request_id for group in self.running}
        self.policy.swapped_ids = {group.request_id for group in self.swapped}
        try:
            result = super().schedule()
        except HostCapacityDefer as error:
            return self._host_defer_outputs(error)
        selected = [item.seq_group for item in result[1].scheduled_seq_groups]
        details = [{"request_id": item.seq_group.request_id,
                    "token_chunk_size": item.token_chunk_size,
                    "prefill": item.seq_group.is_prefill()}
                   for item in result[1].scheduled_seq_groups]
        unique = []
        unique_details = []
        unique_items = []
        seen = set()
        for item, group, detail in zip(result[1].scheduled_seq_groups, selected, details):
            if group.request_id not in seen:
                seen.add(group.request_id)
                unique.append(group)
                unique_details.append(detail)
                unique_items.append(item)
        if len(unique) != len(selected):
            self.compound_events.append({
                "kind": "duplicate_scheduled_dropped",
                "request_ids": [group.request_id for group in selected],
            })
            result[1].scheduled_seq_groups = unique_items
        self._normalize_queues()
        self.policy.note_selection(unique, __import__("time").perf_counter(),
                                   self._dispatch_event, unique_details)
        return result

    def _host_defer_outputs(self, error):
        self.policy.model_action = False
        self.policy.pending_action = None
        self.policy.resident_set = set()
        self.policy.resident_order = []
        self.policy.recovery_set = set()
        self.policy.deadline_deferred = set()
        self.policy.admission_paused = set()
        self.policy.anchor = None
        self.policy.remaining = 0.0
        self.policy.frame_pending = False
        self.policy.forced = None
        self.policy.forced_action = None
        self.policy.effective_controller = "official"
        self.policy.record("v2_host_capacity_deferred", request_id=error.request_id,
                           free_blocks=error.free_blocks, free_tokens=error.free_tokens)
        self.policy.note_selection([], __import__("time").perf_counter(), scheduled_details=[])
        outputs = SchedulerOutputs(
            scheduled_seq_groups=[], num_prefill_groups=0, num_batched_tokens=0,
            blocks_to_swap_in=[], blocks_to_swap_out=[], blocks_to_copy=[],
            ignored_seq_groups=[], num_lookahead_slots=0,
            running_queue_size=len(self.running), preempted=0)
        self.compound_events.append({"kind": "host_capacity_deferred",
                                     "request_id": error.request_id})
        return [], outputs, False

    def abort_seq_group(self, request_id):
        ids = {request_id} if isinstance(request_id, str) else set(request_id)
        for group in list(self.running) + list(self.waiting) + list(self.swapped):
            if group.request_id in ids:
                self.policy.abort_seq_group(group)
        super().abort_seq_group(request_id)

    def _schedule(self):
        official = self.policy.update_schedule_count()
        self.policy.frame_pending = bool(official and self._test_enough_swap_space())
        groups = list(self.running) + list(self.waiting) + list(self.swapped)
        self.policy.before_schedule(groups, __import__("time").perf_counter(), self.v2_ledger)
        if self.policy.frame_pending:
            return self._schedule_slo()
        return self._schedule_chunked_prefill()

    def _schedule_slo(self):
        self.policy.frame_pending = True
        return super()._schedule_slo()

    def _schedule_chunked_prefill(self):
        self.policy.frame_pending = False
        return super()._schedule_chunked_prefill()

    def _preempt(self, seq_group, blocks_to_swap_out, preemption_mode=None):
        mode = super()._preempt(seq_group, blocks_to_swap_out, preemption_mode)
        self.policy.note_preemption(seq_group, mode)
        return mode

    def _has_compound(self, groups=None):
        counts = {}
        source = list(self.running) + list(self.waiting) + list(self.swapped)
        if groups is not None:
            source.extend(groups)
        seen = set()
        for group in source:
            if id(group) in seen:
                continue
            seen.add(id(group))
            if group.first_seq.is_finished():
                continue
            key = (group.collection_id, group.request_info.stage_id)
            counts[key] = counts.get(key, 0) + 1
        declared = getattr(self.policy, "compound_members", {})
        return (any(count > 1 for count in counts.values())
                or any(len(members) > 1 for members in declared.values()))

    def _preempt_pending(self, running_result):
        if not self.policy.policy_preemption_allowed():
            return super()._preempt_pending(running_result)
        self._pending_swap_out_queue.clear()
        return 0, 0, max(0, int(self.max_swaps_per_iter))

    def _build_process_list(self, running_result, swapped_this_iter):
        if not self.policy.policy_preemption_allowed():
            return super()._build_process_list(running_result, swapped_this_iter)
        all_groups = list(self.running) + list(self.waiting) + list(self.swapped)
        unique = [group for group in all_groups if not group.first_seq.is_finished()]
        process_list = self.policy.dispatch_order(unique)
        running_in_bin = [group for group in process_list if group in self.running]
        min_tbt_required = min([group.TBT_constraint for group in running_in_bin],
                               default=0.001)
        to_activate = [group for group in process_list if group not in self.running]
        return process_list, min_tbt_required, to_activate, running_in_bin

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
                members = list(grouped.get(key, []))
            else:
                members = [member for member in grouped.get(key, []) if member in process_list]
            members = [member for member in members if not member.first_seq.is_finished()]
            if members:
                units.append(members)
        return units

    def _mixed_queue_unit(self, unit):
        key = f"{unit[0].collection_id}:stage:{unit[0].request_info.stage_id}"
        declared = getattr(self.policy, "compound_members", {}).get(key)
        if declared is not None and set(declared) != {group.request_id for group in unit}:
            return True
        full = self._compound_units([unit[0]], include_missing=True)
        return not full or len(full[0]) != len(unit)

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

    def _group_status(self, group):
        if group in self.waiting:
            return SequenceStatus.WAITING
        if group in self.swapped:
            return SequenceStatus.SWAPPED
        if group in self.running:
            return SequenceStatus.RUNNING
        return None

    def _token_demands(self, unit, enable_chunking):
        demands = {}
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=max(self.max_batch_size, len(unit)))
        for group in unit:
            status = self._group_status(group)
            if status is None:
                return None
            demands[group.request_id] = self._get_num_new_tokens(
                group, status, enable_chunking, budget)
        return demands

    def _unit_resources(self, unit, budget, enable_chunking):
        tokens = 0
        seqs = 0
        blocks = 0
        demands = self._token_demands(unit, enable_chunking)
        if demands is None:
            return None, "missing_from_queues"
        for group in unit:
            status = self._group_status(group)
            num_tokens = demands[group.request_id]
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

    def _planning_batch_slots(self, target_key=None):
        if getattr(self.policy, "frame_pending", False):
            return max(int(self.max_batch_size),
                       len(self.running) + len(self.waiting) + len(self.swapped))
        pending = len(self.waiting) + len(self.swapped)
        activated = min(int(self.max_swaps_per_iter), pending)
        if target_key is not None:
            members = self.policy.compound_members.get(target_key, [])
            target_incoming = sum(member not in getattr(self.policy, "running_ids", set())
                                  for member in members)
            activated = max(activated, target_incoming)
        return max(int(self.max_batch_size), len(self.running) + activated)

    def _planner_units(self, units, enable_chunking=True):
        planner_units = []
        for unit in units:
            resources, reason = self._unit_resources(unit, None, enable_chunking)
            key = f"{unit[0].collection_id}:stage:{unit[0].request_info.stage_id}"
            if key in getattr(self.policy, "admission_paused", set()) and reason is None:
                reason = "chain_deadline_deferred"
            tokens, seqs, required = resources if resources is not None else (0, 0, None)
            running = [group for group in unit if group in self.running]
            demands = self._token_demands(unit, enable_chunking) or {}
            planner_units.append({
                "ids": [group.request_id for group in unit],
                "groups": unit,
                "tokens": tokens,
                "min_tokens": sum(value > 0 for value in demands.values()),
                "slots": seqs,
                "required_blocks": required,
                "reason": reason,
                "running": running,
                "resident_blocks": sum(self._resident_blocks(group) for group in running),
                "protected": any(self.policy.residency_protected(group) for group in running),
            })
        return planner_units

    def _resource_snapshot(self, groups):
        units = {}
        ledger = self.v2_ledger.view()
        running_order = {group.request_id: index for index, group in enumerate(self.running)}
        batch_slots = self._planning_batch_slots(self.policy.forced or self.policy.anchor)
        self.policy.batch_slots = batch_slots
        for unit in self._compound_units(groups):
            key = f"{unit[0].collection_id}:stage:{unit[0].request_info.stage_id}"
            resources, reason = self._unit_resources(unit, None, True)
            tokens, seqs, required = resources if resources is not None else (0, 0, None)
            running = [group for group in unit if group in self.running]
            statuses = {group.first_seq.status.name for group in unit}
            entries = [ledger.get(group.request_id) for group in unit]
            units[key] = {
                "tokens": tokens,
                "min_tokens": len(unit) if resources is not None else 0,
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
            "host_free_kv_blocks": self.block_manager.get_num_free_cpu_blocks(),
            "host_kv_capacity_blocks": self.block_manager.num_total_cpu_blocks,
            "batch_slots": batch_slots,
        }

    def _action_feasibility(self, target_key):
        groups = list(self.running) + list(self.waiting) + list(self.swapped)
        target_keys = set(getattr(self.policy, "resident_set", None) or {target_key})
        target_members = {request_id for key in target_keys
                          for request_id in self.policy.compound_members.get(key, [])}
        if not target_members:
            return {"feasible": False, "reason": "missing_target"}
        resident_order = getattr(self.policy, "resident_order", [])
        if resident_order:
            order = {key: index for index, key in enumerate(resident_order)}
            ordered = sorted(groups, key=lambda group: (
                order.get(f"{group.collection_id}:stage:{group.request_info.stage_id}",
                          len(order)),
                f"{group.collection_id}:stage:{group.request_info.stage_id}"))
        else:
            protected = [group for group in groups if self.policy.residency_protected(group)]
            target = [group for group in groups if group.request_id in target_members
                      and group not in protected]
            remaining = [group for group in groups if group not in protected and group not in target]

            def projected_priority(group):
                key = f"{group.collection_id}:stage:{group.request_info.stage_id}"
                members = self.policy.compound_members.get(key, [group.request_id])
                level = min((self.policy.current[member]["level"] for member in members
                             if member in self.policy.current), default=0)
                ready_since = self.policy.current[group.request_id]["ready_since"]
                return level, ready_since, key

            target.sort(key=lambda group: (
                self.policy.current[group.request_id]["ready_since"],
                f"{group.collection_id}:stage:{group.request_info.stage_id}"))
            ordered = protected + target + sorted(remaining, key=projected_priority)
        units = self._compound_units(ordered)
        planner_units = self._planner_units(units)
        watermark = (self.block_manager.num_total_gpu_blocks
                     - self.block_manager.get_max_gpu_num_blocks_number())
        available = self.block_manager.get_num_free_gpu_blocks() - watermark
        batch_slots = self._planning_batch_slots(target_key)
        plan = plan_first_step(planner_units, available, batch_slots,
                               self.max_swaps_per_iter, True,
                               self.scheduler_config.max_num_batched_tokens)
        accepted_ids = {request_id for entry in plan["accepted"] for request_id in entry["ids"]}
        feasible = target_members <= accepted_ids
        target_deferred = [item for item in plan["deferred"]
                           if target_members & set(item[0])]
        return {
            "feasible": feasible,
            "reason": None if feasible else (target_deferred[0][1] if target_deferred else "not_admitted"),
            "target": sorted(target_members),
            "accepted": [entry["ids"] for entry in plan["accepted"]],
            "deferred": plan["deferred"],
            "batch_slots": batch_slots,
            "available_blocks": available,
            "swaps_remaining": plan["swaps_remaining"],
            "unit_accounts": plan["unit_accounts"],
        }

    def _allocate_unit_tokens(self, unit, planned_tokens, enable_chunking):
        demands = self._token_demands(unit, enable_chunking)
        if demands is None or any(value <= 0 for value in demands.values()):
            raise RuntimeError("accepted unit lost executable token demand")
        allocation = {group.request_id: 0 for group in unit}
        remaining = int(planned_tokens)
        for group in unit:
            allocation[group.request_id] = 1
            remaining -= 1
        if remaining < 0:
            raise RuntimeError("atomic token allocation below minimum progress")
        while remaining:
            progressed = False
            for group in unit:
                request_id = group.request_id
                if allocation[request_id] >= demands[request_id]:
                    continue
                allocation[request_id] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
            if not progressed:
                break
        if remaining or sum(allocation.values()) != planned_tokens:
            raise RuntimeError("planned and executable prefill tokens differ")
        return allocation

    def _accepted_preflight(self, accepted, budget, enable_chunking):
        simulated = SchedulingBudget(token_budget=budget.remaining_token_budget(),
                                     max_num_seqs=budget.max_num_seqs - budget.num_curr_seqs)
        allocations = {}
        for entry in accepted:
            unit = entry["groups"]
            allocation = self._allocate_unit_tokens(unit, entry["tokens"], enable_chunking)
            allocations[tuple(entry["ids"])] = allocation
            num_seqs = sum(group.get_max_num_running_seqs() for group in unit)
            if not simulated.can_schedule(num_new_tokens=entry["tokens"],
                                          num_new_seqs=num_seqs):
                return None, "budget_recheck"
            simulated.add_num_batched_tokens(unit[0].request_id, entry["tokens"])
            simulated.add_num_seqs(unit[0].request_id, num_seqs)
            for group in unit:
                status = self._group_status(group)
                if status is None or allocation[group.request_id] <= 0:
                    return None, "missing_or_no_tokens"
                lookahead = self._get_num_lookahead_slots(group.is_prefill(), enable_chunking)
                if status == SequenceStatus.WAITING:
                    if allocation[group.request_id] > self._get_prompt_limit(group):
                        return None, "prompt_limit"
                    if self.block_manager.can_allocate(group, lookahead) != AllocStatus.OK:
                        return None, "allocation_recheck"
                elif status == SequenceStatus.SWAPPED:
                    if self.block_manager.can_swap_in(group, lookahead) != AllocStatus.OK:
                        return None, "swap_in_recheck"
                elif not self._can_append_slots(group, enable_chunking):
                    return None, "append_recheck"
        return allocations, None

    def _schedule_v2_accepted(self, accepted, running_result,
                              prefill_result, swapped_in_result, budget,
                              enable_chunking):
        allocations, reason = self._accepted_preflight(accepted, budget, enable_chunking)
        if allocations is None:
            for entry in accepted:
                running_result.leftover_running.extend(
                    group for group in entry["groups"] if group in self.running)
            return reason, []
        executed = []
        for entry in accepted:
            unit = entry["groups"]
            allocation = allocations[tuple(entry["ids"])]
            for group in unit:
                status = self._group_status(group)
                num_new_tokens = allocation[group.request_id]
                lookahead = self._get_num_lookahead_slots(group.is_prefill(), enable_chunking)
                if status == SequenceStatus.SWAPPED:
                    self.swapped.remove(group)
                    swapped_result = self._schedule_slo_swapped(
                        group, num_new_tokens, lookahead, enable_chunking)
                    if swapped_result.is_prefill:
                        swapped_in_result.prefill_seq_groups.append(swapped_result.seq_group)
                    else:
                        swapped_in_result.decode_seq_groups.append(swapped_result.seq_group)
                    swapped_in_result.blocks_to_swap_in.extend(swapped_result.blocks_to_swap_in)
                    swapped_in_result.blocks_to_copy.extend(swapped_result.blocks_to_copy)
                elif status == SequenceStatus.RUNNING:
                    scheduled = self._scheduled_seq_group_cache[self.cache_id].get_object()
                    scheduled.seq_group = group
                    if group.is_prefill():
                        scheduled.token_chunk_size = num_new_tokens
                        running_result.prefill_seq_groups.append(scheduled)
                        running_result.prefill_seq_groups_list.append(group)
                    else:
                        scheduled.token_chunk_size = 1
                        running_result.decode_seq_groups.append(scheduled)
                        running_result.decode_seq_groups_list.append(group)
                    self._append_slots(group, running_result.blocks_to_copy, enable_chunking)
                elif status == SequenceStatus.WAITING:
                    self.waiting.remove(group)
                    waiting_result = self._schedule_slo_prefills(
                        group, num_new_tokens, lookahead, enable_chunking)
                    prefill_result.seq_groups.append(waiting_result.seq_group)
                else:
                    raise RuntimeError("accepted unit disappeared after preflight")
                budget.add_num_batched_tokens(group.request_id, num_new_tokens)
                budget.add_num_seqs(group.request_id, group.get_max_num_running_seqs())
                executed.append((group.request_id, num_new_tokens))
        return None, executed

    def _schedule_process_list(self, process_list, running_result,
                               prefill_result, swapped_in_result, budget,
                               enable_chunking):
        if not self.policy.policy_preemption_allowed():
            return super()._schedule_process_list(
                process_list, running_result, prefill_result,
                swapped_in_result, budget, enable_chunking)
        units = self._compound_units(process_list, include_missing=True)
        action = getattr(self.policy, "pending_action", None)
        if (self.policy.model_action and action is not None
                and not action.get("scheduled") and action.get("expected_first")):
            # This is the execution witness for the first dispatch, not a
            # permanent resident order for every subsequent lease renewal.
            order = {key: index for index, key in enumerate(action["expected_first"])}
            units.sort(key=lambda unit: (
                order.get(f"{unit[0].collection_id}:stage:{unit[0].request_info.stage_id}",
                          len(order)),
                f"{unit[0].collection_id}:stage:{unit[0].request_info.stage_id}"))
        initial_free = self.block_manager.get_num_free_gpu_blocks()
        watermark = (self.block_manager.num_total_gpu_blocks
                     - self.block_manager.get_max_gpu_num_blocks_number())
        initial_available = initial_free - watermark
        planner_units = self._planner_units(units, enable_chunking)
        self.policy.batch_slots = budget.max_num_seqs
        prior_swaps = len(running_result.swapped_out) + len(running_result.preempted)
        swaps_remaining = max(0, int(self.max_swaps_per_iter) - prior_swaps)
        plan = plan_first_step(planner_units, initial_available,
                               budget.max_num_seqs, swaps_remaining,
                               self.policy.policy_preemption_allowed(),
                               budget.remaining_token_budget())
        accepted = list(plan["accepted"])
        running_result.leftover_running.extend(plan["leftover_running"])
        force_preemption_count = 0
        preemptions = []
        for unit, running, reason in plan["evictions"]:
            for group in running:
                mode = self._preempt(group, running_result.blocks_to_swap_out,
                                     PreemptionMode.SWAP)
                if mode == PreemptionMode.SWAP:
                    running_result.swapped_out.append(group)
                elif mode == PreemptionMode.RECOMPUTE:
                    running_result.preempted.append(group)
                force_preemption_count += 1
                cause = ("forced" if self.policy.forced is not None
                         else "model" if self.policy.model_action else "resource")
                preemptions.append({"request_id": group.request_id, "cause": cause,
                                    "trigger": reason, "mode": str(mode)})

        admission_error, executed = self._schedule_v2_accepted(
            accepted, running_result, prefill_result, swapped_in_result,
            budget, enable_chunking)
        if admission_error is not None:
            accepted = []
        accepted_ids = {request_id for entry in accepted for request_id in entry["ids"]}
        simulated_tokens = sum(entry["tokens"] for entry in accepted)
        simulated_seqs = sum(entry["slots"] for entry in accepted)
        simulated_blocks = sum(entry["required_blocks"] for entry in accepted)
        available = initial_available + plan["planned_freed_blocks"] - simulated_blocks
        if available < 0:
            raise RuntimeError("KV planning conservation failed")
        if admission_error is None and budget.num_batched_tokens != simulated_tokens:
            raise RuntimeError("planned token budget and scheduled tokens differ")
        skipped = [(entry["ids"], "evicted" if any(entry is unit for unit, _, _ in plan["evictions"])
                    else next((reason for ids, reason, _ in plan["deferred"] if ids == entry["ids"]),
                              admission_error or "resource"))
                   for entry in planner_units
                   if not set(entry["ids"]) <= accepted_ids]
        self._dispatch_event = {
            "kind": "official_frame_atomic_admission",
            "units": [[group.request_id for group in unit] for unit in units],
            "accepted": [entry["ids"] for entry in accepted],
            "skipped": skipped,
            "simulated": [simulated_tokens, simulated_seqs, simulated_blocks],
            "executed_tokens": executed,
            "admission_error": admission_error,
            "initial_free_blocks": initial_free,
            "watermark_blocks": watermark,
            "initial_available_blocks": initial_available,
            "planned_freed_blocks": plan["planned_freed_blocks"],
            "remaining_available_blocks": available,
            "unit_accounts": plan["unit_accounts"],
            "policy_preemption": self.policy.policy_preemption_allowed(),
            "preemptions": preemptions,
            "deferred": plan["deferred"],
            "swaps_remaining": plan["swaps_remaining"],
            "batch_slots": budget.max_num_seqs,
        }
        self.compound_events.append(self._dispatch_event)
        return force_preemption_count

    def _schedule_running(self, budget, curr_loras, enable_chunking=False):
        if self.lora_enabled or not self.policy.policy_preemption_allowed():
            return super()._schedule_running(budget, curr_loras, enable_chunking)
        result = self._scheduler_running_outputs_cache[self.cache_id].get_object()
        result.blocks_to_swap_out.clear()
        result.blocks_to_copy.clear()
        result.decode_seq_groups.clear()
        result.prefill_seq_groups.clear()
        result.leftover_running.clear()
        result.preempted.clear()
        result.swapped_out.clear()
        result.decode_seq_groups_list.clear()
        result.prefill_seq_groups_list.clear()
        result.num_lookahead_slots = self._get_num_lookahead_slots(False, enable_chunking)
        assert len(self._async_stopped) == 0
        decode = [group for group in self.running if not group.is_prefill()]
        prefill = [group for group in self.running if group.is_prefill()]
        units = self._compound_units(
            self.policy.dispatch_order(decode + prefill), include_missing=False)
        leftover = []
        prefill_result = SchedulerPrefillOutputs.create_empty()
        swapped_in_result = SchedulerSwappedInOutputs.create_empty()
        swaps_used = 0
        index = 0
        while index < len(units):
            unit = units[index]
            key = f"{unit[0].collection_id}:stage:{unit[0].request_info.stage_id}"
            if key in getattr(self.policy, "admission_paused", set()):
                leftover.extend(unit)
                index += 1
                continue
            if self._mixed_queue_unit(unit):
                if swaps_used + len(unit) > self.max_swaps_per_iter:
                    leftover.extend(unit)
                else:
                    swaps_used += len(unit)
                    for group in unit:
                        mode = self._preempt(group, result.blocks_to_swap_out,
                                             PreemptionMode.SWAP)
                        if mode == PreemptionMode.RECOMPUTE:
                            result.preempted.append(group)
                        else:
                            result.swapped_out.append(group)
                        if group in self.running:
                            self.running.remove(group)
                index += 1
                continue
            demands = self._token_demands(unit, enable_chunking)
            if demands is None or any(value <= 0 for value in demands.values()):
                leftover.extend(unit)
                index += 1
                continue
            planned_tokens = min(sum(demands.values()), budget.remaining_token_budget())
            num_seqs = sum(group.get_max_num_running_seqs() for group in unit)
            if planned_tokens < len(unit) or not budget.can_schedule(
                    num_new_tokens=planned_tokens, num_new_seqs=num_seqs):
                leftover.extend(unit)
                index += 1
                continue
            preempt_current = False
            while any(not self._can_append_slots(group, enable_chunking) for group in unit):
                victim = units[-1] if len(units) > index + 1 else unit
                if swaps_used + len(victim) > self.max_swaps_per_iter:
                    leftover.extend(unit)
                    break
                if victim is unit:
                    preempt_current = True
                else:
                    units.pop()
                swaps_used += len(victim)
                for group in victim:
                    mode = self._preempt(group, result.blocks_to_swap_out,
                                         PreemptionMode.SWAP)
                    if mode == PreemptionMode.RECOMPUTE:
                        result.preempted.append(group)
                    else:
                        result.swapped_out.append(group)
                    if group in self.running:
                        self.running.remove(group)
                if preempt_current:
                    break
            if preempt_current or unit in leftover:
                index += 1
                continue
            if any(group in leftover for group in unit):
                index += 1
                continue
            entry = {"ids": [group.request_id for group in unit],
                     "groups": unit, "tokens": planned_tokens}
            error, _ = self._schedule_v2_accepted(
                [entry], result, prefill_result, swapped_in_result,
                budget, enable_chunking)
            if error is not None:
                leftover.extend(unit)
            index += 1
        self.running = deque(leftover)
        result.leftover_running.extend(leftover)
        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()
        return result

    def _schedule_prefills(self, budget, curr_loras, enable_chunking=False):
        if self.lora_enabled or not self.policy.policy_preemption_allowed():
            return super()._schedule_prefills(budget, curr_loras, enable_chunking)
        prefill_result = SchedulerPrefillOutputs.create_empty()
        prefill_result.num_lookahead_slots = self._get_num_lookahead_slots(True, enable_chunking)
        if not self._passed_delay(__import__("time").time()):
            return prefill_result
        ordered = sorted(self.waiting, key=self.policy.get_priority)
        units = self._compound_units(ordered, include_missing=False)
        planner_units = self._planner_units(units, enable_chunking)
        for entry, unit in zip(planner_units, units):
            if self._mixed_queue_unit(unit):
                entry["reason"] = "compound_mixed_queue"
        watermark = (self.block_manager.num_total_gpu_blocks
                     - self.block_manager.get_max_gpu_num_blocks_number())
        available = self.block_manager.get_num_free_gpu_blocks() - watermark
        plan = plan_first_step(planner_units, available,
                               budget.max_num_seqs - budget.num_curr_seqs, 0, False,
                               budget.remaining_token_budget())
        for entry in planner_units:
            if entry["reason"] not in ("prompt_limit", "allocation_never"):
                continue
            for group in entry["groups"]:
                for seq in group.get_seqs(status=SequenceStatus.WAITING):
                    seq.status = SequenceStatus.FINISHED_IGNORED
                if group in self.waiting:
                    self.waiting.remove(group)
                prefill_result.ignored_seq_groups.append(group)
        running_result = SchedulerRunningOutputs.create_empty()
        swapped_in_result = SchedulerSwappedInOutputs.create_empty()
        error, executed = self._schedule_v2_accepted(
            plan["accepted"], running_result, prefill_result,
            swapped_in_result, budget, enable_chunking)
        self._dispatch_event = {
            "kind": "official_chunked_waiting_atomic_admission",
            "accepted": [entry["ids"] for entry in plan["accepted"]] if error is None else [],
            "deferred": plan["deferred"], "admission_error": error,
            "executed_tokens": executed, "batch_slots": budget.max_num_seqs,
        }
        self.compound_events.append(self._dispatch_event)
        return prefill_result

    def _schedule_swapped(self, budget, curr_loras, enable_chunking=False):
        if self.lora_enabled or not self.policy.policy_preemption_allowed():
            return super()._schedule_swapped(budget, curr_loras, enable_chunking)
        swapped_in_result = SchedulerSwappedInOutputs.create_empty()
        swapped_in_result.num_lookahead_slots = self._get_num_lookahead_slots(False, enable_chunking)
        units = self._compound_units(list(self.swapped), include_missing=False)
        planner_units = self._planner_units(units, enable_chunking)
        for entry, unit in zip(planner_units, units):
            if self._mixed_queue_unit(unit):
                entry["reason"] = "compound_mixed_queue"
        watermark = (self.block_manager.num_total_gpu_blocks
                     - self.block_manager.get_max_gpu_num_blocks_number())
        available = self.block_manager.get_num_free_gpu_blocks() - watermark
        plan = plan_first_step(planner_units, available,
                               budget.max_num_seqs - budget.num_curr_seqs, 0, False,
                               budget.remaining_token_budget())
        for entry in planner_units:
            if entry["reason"] != "swap_in_never":
                continue
            for group in entry["groups"]:
                for seq in group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)
                if group in self.swapped:
                    self.swapped.remove(group)
                swapped_in_result.infeasible_seq_groups.append(group)
        prefill_result = SchedulerPrefillOutputs.create_empty()
        running_result = SchedulerRunningOutputs.create_empty()
        error, executed = self._schedule_v2_accepted(
            plan["accepted"], running_result, prefill_result,
            swapped_in_result, budget, enable_chunking)
        self._dispatch_event = {
            "kind": "official_chunked_swapped_atomic_admission",
            "accepted": [entry["ids"] for entry in plan["accepted"]] if error is None else [],
            "deferred": plan["deferred"], "admission_error": error,
            "executed_tokens": executed, "batch_slots": budget.max_num_seqs,
        }
        self.compound_events.append(self._dispatch_event)
        return swapped_in_result


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
    policy.feasibility_provider = scheduler._action_feasibility
    engine.engine.scheduler[0] = scheduler
    return scheduler


from v2_compound_policy import V2CompoundPolicy
