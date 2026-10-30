import time

from contracts import CacheEntry, CallView
from pool_engine import PoolCache
from support_state import SupportState
from execution_view import ExecutionView
from v2_contracts import GoalView
from v2_state import ChainState
from jitserve.scheduler.policy import SLOPolicy


class V2CompoundPolicy(SLOPolicy):
    version = "v2_slo_execution_v5_action_contracts"

    def __init__(self, perf, pred, stage_counts, candidate_selector=None, batch_slots=4):
        super().__init__(schedule_interval=50, penalty_factor=100)
        self.perf = perf
        self.pred = pred
        self.state = ChainState(stage_counts)
        self.current = self.state.current
        self.chain_meta = self.state.meta
        self.support = SupportState(pred)
        self.candidate_selector = candidate_selector
        self.batch_slots = batch_slots
        self.compound_members = {}
        self.priorities = {}
        self.anchor = None
        self.remaining = 0.0
        self.forced = None
        self.last_selected = []
        self.selection_ready = {}
        self.selection_time = None
        self.audit_events = []
        self.model_action = False
        self.action_serial = 0
        self.pending_action = None
        self.last_view = None
        self.decode_progress = {}
        self.resource_snapshot = {}
        self.resource_provider = None

    def record(self, kind, **fields):
        self.audit_events.append({"n": len(self.audit_events), "kind": kind,
                                  "monotonic": time.perf_counter(), **fields})

    def add_seq_group(self, seq_group):
        reset, previous_history = self.state.add(
            seq_group, time.perf_counter(), set(self.seq_group_dict))
        if reset:
            self.record("v2_chain_reset", collection_id=seq_group.collection_id,
                        previous_history=previous_history)
        super().add_seq_group(seq_group)
        self.record("v2_admit", request_id=seq_group.request_id,
                    collection_id=seq_group.collection_id,
                    stage_id=seq_group.request_info.stage_id)

    def delete_seq_group(self, seq_group):
        super().delete_seq_group(seq_group)
        self.decode_progress.pop(seq_group.request_id, None)
        remaining = [group for group in self.seq_group_dict.get(seq_group.collection_id, [])
                     if group.request_info.stage_id == seq_group.request_info.stage_id]
        state = self.state.finish(seq_group, time.perf_counter(), remaining)
        if state is not None:
            self.record("v2_finish", request_id=seq_group.request_id,
                        collection_id=seq_group.collection_id,
                        stage_id=seq_group.request_info.stage_id,
                        service=state["service"], wait=state["wait"])

    def abort_seq_group(self, seq_group):
        self.state.abort(seq_group)
        self.decode_progress.pop(seq_group.request_id, None)
        super().delete_seq_group(seq_group)
        self.record("v2_abort", request_id=seq_group.request_id,
                    collection_id=seq_group.collection_id)

    def _view(self, groups, now, ledger):
        grouped = {}
        for group in groups:
            grouped.setdefault((group.collection_id, group.request_info.stage_id), []).append(group)
        collections = {collection for collection, _ in grouped}
        if len(collections) != len({group.collection_id for group in groups}):
            raise RuntimeError("ambiguous collection grouping")
        per_collection = {}
        for collection, stage in grouped:
            if collection in per_collection:
                raise RuntimeError("same collection has concurrent stages; goal identity is ambiguous")
            per_collection[collection] = stage

        goals = []
        entries = []
        owners = {}
        self.compound_members = {}
        native_priorities = {}
        for (collection, stage), members in sorted(grouped.items()):
            members = sorted(members, key=lambda group: (group.arrival_time, group.request_id))
            key = f"{collection}:stage:{stage}"
            self.compound_members[key] = [group.request_id for group in members]
            native_priorities[key] = min(SLOPolicy.get_priority(self, group) for group in members)
            states = [self.current[group.request_id] for group in members]
            meta = self.chain_meta[collection]
            ledger_entries = [ledger.view().get(group.request_id) for group in members]
            cache_entry = CacheEntry(
                key=f"0:{collection}",
                gpu=sum(entry.gpu for entry in ledger_entries),
                host=sum(entry.host for entry in ledger_entries),
                gpu_touch=max(entry.gpu_touch for entry in ledger_entries),
                host_touch=max(entry.host_touch for entry in ledger_entries),
                pending=sum(entry.pending for entry in ledger_entries),
            )
            call = CallView(
                key=key,
                program=collection,
                ordinal=stage,
                prompt_tokens=sum(group.first_seq.get_prompt_len() for group in members),
                emitted=sum(group.first_seq.get_output_len() for group in members),
                produced_prefix=cache_entry.gpu,
                level=min(state["level"] for state in states),
                quantum_used=sum(state["quantum"] for state in states),
                ready_wait=max(now - state["ready_since"] for state in states),
                service=sum(state["service"] for state in states),
                virtual_service=meta["virtual"],
                running=any(group.request_id in getattr(self, "running_ids", set())
                            for group in members),
                historical_wait=sum(state["wait"] for state in states),
            )
            goals.append(GoalView(
                program=collection,
                arrival=meta["arrival"],
                deadline=meta["deadline"],
                state="READY",
                history=tuple(meta["history"]),
                ready=call,
                produced_prefix=cache_entry.gpu,
            ))
            entries.append(cache_entry)
            owners[collection] = 0
        goals.extend(self.state.away_goals(now, set(per_collection)))
        owners.update({goal.program: 0 for goal in goals if goal.state == "AWAY"})
        cache = PoolCache(ledger.gpu_capacity, ledger.host_capacity, tuple(entries),
                          tuple(sorted(owners.items())))
        virtual = tuple(sorted((collection, meta["virtual"])
                               for collection, meta in self.chain_meta.items()))
        declared = {
            f"{collection}:stage:{stage}": sum(group.request_info.input_len for group in members)
            for (collection, stage), members in grouped.items()
        }
        for goal in goals:
            if goal.state == "AWAY":
                last = goal.history[-1]
                declared[f"away:{goal.program}"] = self.chain_meta[goal.program]["declared_history"].get(
                    last.ordinal, last.prompt_tokens)
        snapshot = self.resource_snapshot
        return ExecutionView(
            now, tuple(goals), cache, virtual, tuple(declared.items()),
            tuple(self.compound_members.items()),
            resource_units=tuple(snapshot.get("units", {}).items()),
            free_kv_blocks=snapshot.get("free_kv_blocks", 0),
            watermark_blocks=snapshot.get("watermark_blocks", 0),
            block_size=snapshot.get("block_size", 16),
            max_swaps_per_iter=snapshot.get("max_swaps_per_iter", 3),
            token_budget=snapshot.get("token_budget", 2048),
            native_priorities=tuple(native_priorities.items()))

    def before_schedule(self, groups, now, ledger):
        self.priorities = {}
        self.compound_members = {}
        groups = [group for group in groups if not group.first_seq.is_finished()]
        if not groups:
            self.anchor = None
            self.remaining = 0.0
            self.forced = None
            self.model_action = False
            self.last_view = None
            return
        view = self._view(groups, now, ledger)
        self.last_view = view
        decode_seconds = self.perf.decode_seconds
        quantum = 64 * decode_seconds
        floor = min(self.chain_meta[group.collection_id]["virtual"] for group in groups)
        eligible_groups = [group for group in groups
                           if self.chain_meta[group.collection_id]["virtual"] <= floor + 2 * quantum]
        eligible_ids = {group.request_id for group in eligible_groups}
        eligible = {goal.ready.key for goal in view.goals if goal.ready is not None
                    and all(member in eligible_ids for member in self.compound_members[goal.ready.key])}
        oldest = max(groups, key=lambda group: now - self.current[group.request_id]["ready_since"])
        self.forced = None
        if now - self.current[oldest.request_id]["ready_since"] > 8 * quantum:
            self.forced = next((goal.ready.key for goal in view.goals if goal.ready is not None
                                and oldest.request_id in self.compound_members[goal.ready.key]), None)

        audit = None
        report = None
        if self.anchor not in eligible or self.remaining <= 0:
            if self.forced is not None:
                self.anchor = self.forced
                self.remaining = 4 * decode_seconds
                audit = {"accepted": False, "fallback": "forced",
                         "fallback_kind": "forced", "actions": [], "seconds": 0.0}
            else:
                if self.resource_provider is not None:
                    self.resource_snapshot = self.resource_provider(groups)
                    view = self._view(groups, now, ledger)
                    self.last_view = view
                report = self.support.check_goals(
                    view.goals, decode_seconds, self.compound_members, dict(view.declared_inputs))
                if report.supported and self.candidate_selector is not None:
                    try:
                        (self.anchor, self.remaining), audit = self.candidate_selector(
                            view, 0, eligible, self.pred, self.perf,
                            {goal.program: 0 for goal in view.goals},
                            batch_slots=self.batch_slots, compound_members=self.compound_members)
                    except (ValueError, RuntimeError, TimeoutError) as error:
                        (self.anchor, self.remaining), audit = self.support.fallback_action(
                            view.calls, eligible, decode_seconds)
                        audit.update(fallback=str(error), fallback_kind="evaluation_error")
                else:
                    (self.anchor, self.remaining), audit = self.support.fallback_action(
                        view.calls, eligible, decode_seconds)
                    audit["unsupported_goals"] = report.unsupported_keys
                    audit["goal_denominator"] = report.denominator
            if self.forced is None:
                disposition = self.support.observe_audit(
                    audit, view.goals, decode_seconds, self.compound_members, dict(view.declared_inputs))
                audit["fallback_kind"] = disposition
            self.model_action = bool(audit.get("accepted"))
            if self.model_action:
                self.action_serial += 1
                chosen = next((item for item in audit.get("actions", [])
                               if item["key"] == self.anchor and item["h"] == self.remaining), None)
                trace = chosen.get("batch_trace", []) if chosen else []
                self.pending_action = {
                    "id": self.action_serial, "anchor": self.anchor,
                    "expected_first": trace[0]["keys"] if trace else None,
                    "scheduled": False, "executed": False,
                }
                self.record("v2_action_accepted", **self.pending_action)
            self.record("v2_candidate", anchor=self.anchor, remaining=self.remaining,
                        eligible=sorted(eligible), forced=self.forced,
                        compound_members=self.compound_members, audit=audit,
                        support_results=[] if report is None else [result.__dict__ for result in report.results],
                        view=view.to_dict())

        for goal in view.goals:
            if goal.ready is None:
                continue
            if goal.ready.key == self.forced:
                rank = 0
            elif goal.ready.key == self.anchor:
                rank = 1
            else:
                rank = 2 + min(self.current[request_id]["level"]
                               for request_id in self.compound_members[goal.ready.key])
            for request_id in self.compound_members[goal.ready.key]:
                self.priorities[request_id] = (
                    rank,
                    -(now - self.current[request_id]["ready_since"]),
                    goal.ready.key,
                )

    def policy_preemption_allowed(self):
        return self.model_action or self.forced is not None

    def custom_dispatch_active(self):
        return (self.policy_preemption_allowed()
                or any(len(members) > 1 for members in self.compound_members.values()))

    def note_preemption(self, group, mode):
        request_id = group.request_id
        previous = self.decode_progress.get(request_id, 0)
        self.decode_progress[request_id] = 0
        self.record("v2_preemption_progress", request_id=request_id,
                    previous_progress=previous, mode=str(mode))

    def residency_protected(self, group):
        if group.request_id not in getattr(self, "running_ids", set()):
            return False
        progress = self.decode_progress.get(group.request_id)
        return progress is not None and (group.is_prefill() or progress < 4)

    def dispatch_order(self, groups):
        protected = [group for group in groups if self.residency_protected(group)]
        remaining = [group for group in groups if not self.residency_protected(group)]
        if self.policy_preemption_allowed():
            return protected + sorted(remaining, key=self.get_priority)
        running = [group for group in remaining if group.request_id in getattr(self, "running_ids", set())]
        stationary = [group for group in remaining if group.request_id not in getattr(self, "running_ids", set())]
        return protected + running + sorted(stationary, key=self.get_priority)

    def get_priority(self, seq_group):
        if not self.policy_preemption_allowed():
            return super().get_priority(seq_group)
        return self.priorities.get(seq_group.request_id, (99, seq_group.arrival_time))

    def note_selection(self, selected, now, dispatch_report=None):
        self.last_selected = list(selected)
        self.selection_time = now
        self.selection_ready = {group.request_id: self.current[group.request_id]["ready_since"]
                                for group in self.last_selected if group.request_id in self.current}
        selected_ids = {group.request_id for group in selected}
        self.record("v2_selected", selected=[group.request_id for group in selected])
        action = self.pending_action
        if self.model_action and action is not None and not action["scheduled"]:
            members = set(self.compound_members.get(action["anchor"], []))
            if members and members <= selected_ids:
                action["scheduled"] = True
                actual_keys = sorted({group_key for group_key, request_ids in self.compound_members.items()
                                      if selected_ids & set(request_ids)})
                action["matches_projection"] = (None if action["expected_first"] is None else
                                                sorted(action["expected_first"]) == actual_keys)
                self.record("v2_action_scheduled", id=action["id"], anchor=action["anchor"],
                            expected_first=action["expected_first"], actual_first=actual_keys,
                            matches_projection=action["matches_projection"],
                            dispatch_report=dispatch_report)

    def after_model_step(self, span, now):
        selected = self.last_selected
        if not selected:
            return
        for group in selected:
            if not group.is_prefill() and group.request_id in self.decode_progress:
                self.decode_progress[group.request_id] += 1
            self.state.after_step(group, span, now, self.selection_time,
                                  self.selection_ready, self.perf.decode_seconds)
        anchor_members = set(self.compound_members.get(self.anchor, []))
        if any(group.request_id in anchor_members for group in selected):
            self.remaining -= self.perf.decode_seconds
        action = self.pending_action
        if (self.model_action and action is not None and action["scheduled"]
                and not action["executed"] and action.get("matches_projection") is not False):
            if any(group.request_id in anchor_members for group in selected):
                action["executed"] = True
                self.record("v2_action_executed", id=action["id"], anchor=action["anchor"],
                            selected=[group.request_id for group in selected])
        self.record("v2_after_step", selected=[group.request_id for group in selected],
                    span=span, anchor=self.anchor, remaining=self.remaining)

    def snapshot(self):
        return {
            "mechanism_version": self.version,
            "anchor": self.anchor,
            "remaining": self.remaining,
            "forced": self.forced,
            "decode_progress": dict(self.decode_progress),
            "support_cache": self.support.snapshot(),
            "chain_state": self.state.snapshot(),
            "events": self.audit_events,
        }
