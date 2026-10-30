import time

from contracts import CacheEntry, CallView
from pool_engine import PoolCache
from support_state import SupportState
from execution_view import ExecutionView
from execution_reference import forecasts
from deadline_admission import admission_pauses
from v2_contracts import GoalView
from v2_state import ChainState
from jitserve.scheduler.policy import SLOPolicy


class V2CompoundPolicy(SLOPolicy):
    version = "v3_chain_dynamic_v42"
    lease_yield_enabled = False
    background_hold_enabled = True
    defer_safe_later_enabled = True

    def __init__(self, perf, pred, stage_counts, candidate_selector=None, batch_slots=4):
        super().__init__(schedule_interval=50, penalty_factor=100)
        self.perf = perf
        self.pred = pred
        self.state = ChainState(stage_counts)
        self.stage_counts = stage_counts
        self.current = self.state.current
        self.chain_meta = self.state.meta
        self.support = SupportState(pred)
        self.candidate_selector = candidate_selector
        self.batch_slots = batch_slots
        self.compound_members = {}
        self.priorities = {}
        self.resident_set = set()
        self.resident_order = []
        self.recovery_set = set()
        self.deadline_deferred = set()
        self.admission_paused = set()
        self.anchor = None
        self.remaining = 0.0
        self.forced = None
        self.forced_action = None
        self.forced_status = None
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
        self.feasibility_provider = None
        self.custom_forced_enabled = False
        self.frame_pending = False
        self.effective_controller = "official"
        self.last_schedule_report = None

    def record(self, kind, **fields):
        self.audit_events.append({"n": len(self.audit_events), "kind": kind,
                                  "monotonic": time.perf_counter(), **fields})

    def allocation_control(self, seq_group):
        unit_time = self._get_interval_time()
        return sum((unit_time * (peer.predict_output_length - peer.seqs[0].get_decode_len())
                    for peer in self.seq_group_dict[seq_group.collection_id]), 0.0)

    def add_seq_group(self, seq_group):
        reset, previous_history = self.state.add(
            seq_group, time.perf_counter(), set(self.seq_group_dict))
        if reset:
            self.record("v2_chain_reset", collection_id=seq_group.collection_id,
                        previous_history=previous_history)
        super().add_seq_group(seq_group)
        self.decode_progress.setdefault(seq_group.request_id, 0)
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
        member_progress = {}
        native_priorities = {}
        for (collection, stage), members in sorted(grouped.items()):
            members = sorted(members, key=lambda group: (group.arrival_time, group.request_id))
            key = f"{collection}:stage:{stage}"
            self.compound_members[key] = [group.request_id for group in members]
            member_progress[key] = tuple((group.request_id, group.first_seq.get_output_len())
                                         for group in members)
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
            native_priorities=tuple(native_priorities.items()),
            compound_progress=tuple(member_progress.items()),
            schedule_count=self.num_schedule_count + 1,
            schedule_interval=self.schedule_interval,
            official_frame=self.frame_pending,
            incumbent_action=((self.anchor, self.remaining)
                              if self.model_action and self.anchor is not None else None),
            host_free_kv_blocks=snapshot.get("host_free_kv_blocks"),
            host_kv_capacity_blocks=snapshot.get("host_kv_capacity_blocks"),
            custom_forced_enabled=self.custom_forced_enabled,
            defer_hopeless=self.background_hold_enabled,
            defer_safe_later=self.defer_safe_later_enabled,
            admission_frame_reserve=self.schedule_interval * self.perf.decode_seconds)

    def _forced_candidate(self, groups, view, now):
        if not self.custom_forced_enabled:
            return None
        quantum = 64 * self.perf.decode_seconds
        oldest = max(groups, key=lambda group: now - self.current[group.request_id]["ready_since"])
        if now - self.current[oldest.request_id]["ready_since"] <= 8 * quantum:
            return None
        return next((goal.ready.key for goal in view.goals if goal.ready is not None
                     and oldest.request_id in self.compound_members[goal.ready.key]), None)

    def _audit_forced_status(self, key, feasible, report):
        status = (key, feasible, report.get("reason") if report else "no_feasibility_provider")
        if status == self.forced_status:
            return
        self.forced_status = status
        self.record("v2_forced_feasibility", anchor=key, feasible=feasible,
                    frame_pending=self.frame_pending, report=report)

    def _interrupt_model_lease(self, forced):
        if not (self.model_action and self.remaining > 0):
            return
        action = self.pending_action
        if action is not None:
            action["lease_interrupted"] = True
            action["interrupted_by"] = forced
        self.record("v2_forced_lease_override", previous_anchor=self.anchor,
                    previous_remaining=self.remaining,
                    previous_action_id=None if action is None else action["id"],
                    forced=forced, reason="starvation")
        self.pending_action = None
        self.model_action = False

    def before_schedule(self, groups, now, ledger):
        self.priorities = {}
        self.compound_members = {}
        groups = [group for group in groups if not group.first_seq.is_finished()]
        if not groups:
            self.anchor = None
            self.remaining = 0.0
            self.forced = None
            self.forced_action = None
            self.model_action = False
            self.pending_action = None
            self.resident_set = set()
            self.recovery_set = set()
            self.admission_paused = set()
            self.last_view = None
            self.effective_controller = "official"
            return
        if not self.frame_pending and not self.model_action and self.forced is None:
            self.last_view = None
            self.effective_controller = "official"
            return
        if self.resource_provider is not None:
            self.resource_snapshot = self.resource_provider(groups)
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
        attainable = set()
        risk_by_goal = {}
        slack_by_goal = {}
        if self.pred is not None:
            estimated_jobs = {}
            if hasattr(view.cache, "get"):
                try:
                    estimated_jobs = {job["key"]: job for job in forecasts(
                        view, self.pred, self.perf, conformal=False)}
                except ValueError:
                    pass
            for goal in view.goals:
                if goal.ready is None:
                    continue
                if goal.ready.key in estimated_jobs:
                    work = estimated_jobs[goal.ready.key]["work"]
                else:
                    try:
                        estimate = self.pred.estimate(
                            goal, self.perf, False,
                            slots=len(self.compound_members[goal.ready.key]))
                    except ValueError:
                        continue
                    work = estimate["work"]
                slack = goal.deadline - now
                risk_work = work + max(0.0, getattr(self.pred, "risk_q", 0.0)) * max(
                    16.0, 0.25 * work / decode_seconds) * decode_seconds
                risk_by_goal[goal.ready.key] = risk_work
                slack_by_goal[goal.ready.key] = slack
                if risk_work <= slack:
                    attainable.add(goal.ready.key)
                    eligible.add(goal.ready.key)
        self.deadline_deferred = (set() if self.pred is None else {
            goal.ready.key for goal in view.goals
            if goal.ready is not None and goal.ready.key not in attainable
        })
        ready_goals = {goal.ready.key: goal for goal in view.goals if goal.ready is not None}
        paused = admission_pauses(
            risk_by_goal, {key: goal.deadline for key, goal in ready_goals.items()},
            now, ready_goals, defer_hopeless=self.background_hold_enabled,
            defer_safe_later=self.defer_safe_later_enabled,
            frame_reserve=self.schedule_interval * decode_seconds,
            recovery_costs={key: 2 * self.resource_snapshot.get("units", {}).get(
                key, {}).get("gpu_tokens", 0) * getattr(self.perf, "copy_seconds_per_token", 0.)
                for key in ready_goals})
        # Finish the current bounded lease before pausing an anchor. Its
        # execution witness and lease accounting must remain closed.
        if self.model_action and self.remaining > 0:
            paused.discard(self.anchor)
        if paused != self.admission_paused:
            self.record("v2_admission_pause", paused=sorted(paused),
                        attainable=sorted(attainable),
                        reason="reserve_compute_for_urgent_cohort" if paused else "drain_all_calls")
        self.admission_paused = paused

        if (self.frame_pending and self.model_action and self.remaining <= 0
                and self.anchor in eligible):
            can_renew = ((self.pred is None or self.anchor in attainable)
                         and self.anchor not in self.admission_paused)
            anchor_goal = next(goal for goal in view.goals
                               if goal.ready is not None and goal.ready.key == self.anchor)
            final_stage = (anchor_goal.ready.ordinal + 1
                           >= self.stage_counts[anchor_goal.program])
            yield_to = None
            if can_renew and final_stage and self.lease_yield_enabled and self.pred is not None:
                current_ratio = (slack_by_goal[self.anchor]
                                 / max(1e-9, risk_by_goal[self.anchor]))
                alternatives = [(slack_by_goal[key] / max(1e-9, risk_by_goal[key]), key)
                                for key in attainable if key != self.anchor]
                if alternatives:
                    ratio, key = min(alternatives)
                    if ratio < 0.8 * current_ratio:
                        yield_to = (key, ratio, current_ratio)
            if can_renew and yield_to is None:
                lease_steps = max(1, self.schedule_interval
                                  - self.num_schedule_count % self.schedule_interval)
                self.remaining = lease_steps * decode_seconds
                self.record("v2_lease_renewed",
                            id=None if self.pending_action is None else self.pending_action["id"],
                            anchor=self.anchor, lease_steps=lease_steps,
                            remaining=self.remaining)
            elif yield_to is not None:
                self.record("v2_lease_yield",
                            id=None if self.pending_action is None else self.pending_action["id"],
                            anchor=self.anchor, next_anchor=yield_to[0],
                            next_ratio=yield_to[1], current_ratio=yield_to[2])

        forced_candidate = self._forced_candidate(groups, view, now)
        self.forced = None
        forced_report = None
        if forced_candidate is not None:
            feasible = False
            if self.frame_pending and self.feasibility_provider is not None:
                forced_report = self.feasibility_provider(forced_candidate)
                feasible = bool(forced_report.get("feasible"))
            self._audit_forced_status(forced_candidate, feasible, forced_report)
            if feasible:
                self.forced = forced_candidate

        audit = None
        report = None
        if self.forced is not None:
            self._interrupt_model_lease(self.forced)
            self.anchor = self.forced
            self.remaining = 4 * decode_seconds
            self.model_action = False
            self.pending_action = None
            self.resident_set = set(self.compound_members.get(self.forced, []))
            self.recovery_set = set()
            self.action_serial += 1
            self.forced_action = {
                "id": self.action_serial, "action_kind": "forced", "anchor": self.forced,
                "scheduled": False, "executed": False,
                "feasibility": forced_report,
            }
            audit = {"accepted": False, "fallback": "forced",
                     "fallback_kind": "forced", "actions": [], "seconds": 0.0,
                     "feasibility": forced_report}
            self.record("v2_forced_accepted", **self.forced_action)
        elif self.frame_pending and (not self.model_action or self.anchor not in eligible
                                     or self.remaining <= 0):
            incumbent_action = ((self.anchor, self.remaining)
                                if self.model_action and self.anchor is not None else None)
            self.pending_action = None
            self.model_action = False
            self.resident_set = set()
            self.recovery_set = set()
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
                        batch_slots=self.batch_slots, compound_members=self.compound_members,
                        incumbent=incumbent_action)
                except (ValueError, RuntimeError, TimeoutError) as error:
                    (self.anchor, self.remaining), audit = self.support.fallback_action(
                        view.calls, eligible, decode_seconds)
                    audit.update(fallback=str(error), fallback_kind="evaluation_error")
            else:
                (self.anchor, self.remaining), audit = self.support.fallback_action(
                    view.calls, eligible, decode_seconds)
                audit["unsupported_goals"] = report.unsupported_keys
                audit["goal_denominator"] = report.denominator
            disposition = self.support.observe_audit(
                audit, view.goals, decode_seconds, self.compound_members, dict(view.declared_inputs))
            audit["fallback_kind"] = disposition
            self.model_action = bool(audit.get("accepted"))
            if self.model_action:
                chosen = next((item for item in audit.get("actions", [])
                               if item["key"] == self.anchor and item["h"] == self.remaining), None)
                trace = chosen.get("batch_trace", []) if chosen else []
                first = trace[0] if trace else None
                self.resident_set = ({self.anchor} if first is None else set(first["keys"]))
                self.resident_order = ([self.anchor] if first is None else list(first["keys"]))
                self.recovery_set = set(audit.get("evicted_attainable", {}))
                if self.feasibility_provider is not None:
                    feasible = self.feasibility_provider(self.anchor)
                    if not feasible.get("feasible"):
                        audit.update(accepted=False, fallback="resource_infeasible",
                                     resource_infeasible=feasible)
                        self.model_action = False
                        self.pending_action = None
                        self.resident_set = set()
                        self.resident_order = []
                        self.recovery_set = set()
            if self.model_action:
                self.action_serial += 1
                self.pending_action = {
                    "id": self.action_serial, "action_kind": "model", "anchor": self.anchor,
                    "expected_first": None if first is None else list(first["keys"]),
                    "expected_first_slots": None if first is None else first.get("slots"),
                    "expected_lease_after": None if first is None else first.get("lease_after"),
                    "expected_first_trace": first,
                    "lease_total": self.remaining,
                    "decode_steps": 0,
                    "lease_completed": False,
                    "scheduled": False, "executed": False,
                }
                self.record("v2_action_accepted", **self.pending_action)

        if audit is not None:
            self.record("v2_candidate", anchor=self.anchor, remaining=self.remaining,
                        eligible=sorted(eligible), forced=self.forced,
                        compound_members=self.compound_members, audit=audit,
                        support_results=[] if report is None else [result.__dict__ for result in report.results],
                        view=view.to_dict(), batch_slots=self.batch_slots,
                        frame_pending=self.frame_pending)

        for goal in view.goals:
            if goal.ready is None:
                continue
            if goal.ready.key == self.forced:
                rank = 0
            elif goal.ready.key == self.anchor and self.model_action:
                rank = float("-inf")
            elif self.model_action and goal.ready.key in self.resident_set:
                rank = 2
            elif self.model_action and goal.ready.key in self.recovery_set:
                for request_id in self.compound_members[goal.ready.key]:
                    self.priorities[request_id] = (2, goal.deadline, goal.ready.key)
                continue
            else:
                rank = 3 + min(self.current[request_id]["level"]
                               for request_id in self.compound_members[goal.ready.key])
            for request_id in self.compound_members[goal.ready.key]:
                self.priorities[request_id] = (
                    rank,
                    -(now - self.current[request_id]["ready_since"]),
                    goal.ready.key,
                )

    def policy_preemption_allowed(self):
        return (self.model_action or bool(self.deadline_deferred or self.admission_paused)
                or (self.frame_pending and self.forced is not None))

    def custom_dispatch_active(self):
        return self.policy_preemption_allowed()

    def note_preemption(self, group, mode):
        request_id = group.request_id
        previous = self.decode_progress.get(request_id, 0)
        self.decode_progress[request_id] = 0
        self.record("v2_preemption_progress", request_id=request_id,
                    previous_progress=previous, mode=str(mode))

    def residency_protected(self, group):
        if group.request_id not in getattr(self, "running_ids", set()):
            return False
        key = f"{group.collection_id}:stage:{group.request_info.stage_id}"
        if key in self.deadline_deferred or key in self.admission_paused:
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
        key = f"{seq_group.collection_id}:stage:{seq_group.request_info.stage_id}"
        if self.forced is not None and key == self.forced:
            return self.priorities.get(seq_group.request_id, (float("-inf"), key))
        if self.model_action and key == self.anchor:
            return self.priorities.get(seq_group.request_id, (float("-inf"), key))
        if key in self.deadline_deferred:
            return (float("inf"), self.chain_meta[seq_group.collection_id]["deadline"], key)
        # Deadline classification must not silently replace the official order
        # with MLFQ on frames, or with FCFS after priorities are cleared off-frame.
        return super().get_priority(seq_group)

    def _set_effective_controller(self, controller, selected, scheduled_details):
        self.effective_controller = controller
        if self.frame_pending or controller != "official":
            self.record("v2_effective_controller", controller=controller,
                        schedule_count=self.num_schedule_count, frame_pending=self.frame_pending,
                        selected=[group.request_id for group in selected],
                        scheduled=scheduled_details)

    def note_selection(self, selected, now, dispatch_report=None, scheduled_details=None):
        self.last_selected = list(selected)
        self.selection_time = now
        self.selection_ready = {group.request_id: self.current[group.request_id]["ready_since"]
                                for group in self.last_selected if group.request_id in self.current}
        if not self.frame_pending and not self.model_action and self.forced is None:
            self._set_effective_controller(
                "deadline_class" if self.deadline_deferred or self.admission_paused else "official",
                selected, scheduled_details or [])
            return
        selected_ids = {group.request_id for group in selected}
        actual_keys = sorted({key for key, request_ids in self.compound_members.items()
                              if selected_ids & set(request_ids)})
        scheduled_details = scheduled_details or []
        self.last_schedule_report = dispatch_report
        if self.frame_pending or self.model_action or self.forced is not None:
            self.record("v2_selected", selected=[group.request_id for group in selected],
                        actual_first=actual_keys, scheduled=scheduled_details,
                        dispatch_report=dispatch_report)
        forced_members = set(self.compound_members.get(self.forced, []))
        anchor_members = set(self.compound_members.get(self.anchor, []))
        if (self.frame_pending and self.forced is not None and forced_members
                and forced_members <= selected_ids):
            controller = "forced"
        elif (self.model_action and anchor_members
              and anchor_members <= selected_ids):
            controller = "model"
        else:
            controller = "deadline_class" if self.deadline_deferred or self.admission_paused else "official"
        self._set_effective_controller(controller, selected, scheduled_details)

        action = self.pending_action
        if (self.frame_pending and self.model_action and action is not None
                and not action["scheduled"] and anchor_members <= selected_ids):
            action["scheduled"] = True
            action["actual_first"] = actual_keys
            action["actual_request_ids"] = [group.request_id for group in selected]
            action["actual_chunks"] = scheduled_details
            action["matches_projection"] = (None if action["expected_first"] is None else
                                            sorted(action["expected_first"]) == actual_keys)
            action["dispatch_report"] = dispatch_report
            self.record("v2_action_scheduled", **action)

        forced = self.forced_action
        if (self.frame_pending and self.forced is not None and forced is not None
                and not forced["scheduled"]):
            forced["actual_first"] = actual_keys
            forced["actual_request_ids"] = [group.request_id for group in selected]
            forced["actual_chunks"] = scheduled_details
            forced["dispatch_report"] = dispatch_report
            if forced_members and forced_members <= selected_ids:
                forced["scheduled"] = True
                self.record("v2_forced_scheduled", **forced)
            else:
                forced["infeasible_at_admission"] = True
                self.record("v2_forced_not_scheduled", **forced)

    def after_model_step(self, span, now):
        selected = self.last_selected
        if not selected:
            return
        for group in selected:
            if not group.is_prefill() and group.request_id in self.decode_progress:
                self.decode_progress[group.request_id] += 1
            self.state.after_step(group, span, now, self.selection_time,
                                  self.selection_ready, self.perf.decode_seconds)
        if not self.frame_pending and not self.model_action and self.forced is None:
            return
        anchor_members = set(self.compound_members.get(self.anchor, []))
        selected_anchor = any(group.request_id in anchor_members for group in selected)
        action = self.pending_action
        if self.effective_controller == "model" and self.model_action and selected_anchor:
            self.remaining -= self.perf.decode_seconds
            if action is not None:
                action["decode_steps"] += 1
                if action["decode_steps"] == 1:
                    action["actual_lease_after_first"] = self.remaining
                    expected = action.get("expected_lease_after")
                    action["lease_first_matches"] = (
                        None if expected is None else abs(expected - self.remaining) <= 1e-9)
                if self.remaining <= 1e-12 and not action["lease_completed"]:
                    action["lease_completed"] = True
                    self.record("v2_lease_completed", **action)
        if (self.effective_controller == "model" and self.model_action
                and action is not None and action["scheduled"]
                and not action["executed"] and action.get("matches_projection") is not False):
            if selected_anchor:
                action["executed"] = True
                action["execution_controller"] = self.effective_controller
                self.record("v2_action_executed", **action)
        forced = self.forced_action
        if (self.effective_controller == "forced" and forced is not None
                and forced.get("scheduled") and not forced["executed"]):
            forced_members = set(self.compound_members.get(self.forced, []))
            if forced_members and all(any(group.request_id == member for group in selected)
                                      for member in forced_members):
                forced["executed"] = True
                forced["execution_controller"] = self.effective_controller
                self.record("v2_forced_executed", **forced)
        if self.frame_pending or self.effective_controller != "official":
            self.record("v2_after_step", selected=[group.request_id for group in selected],
                        span=span, anchor=self.anchor, remaining=self.remaining,
                        effective_controller=self.effective_controller)

    def snapshot(self):
        return {
            "mechanism_version": self.version,
            "anchor": self.anchor,
            "remaining": self.remaining,
            "forced": self.forced,
            "forced_action": self.forced_action,
            "pending_action": self.pending_action,
            "effective_controller": self.effective_controller,
            "frame_pending": self.frame_pending,
            "batch_slots": self.batch_slots,
            "recovery_set": sorted(self.recovery_set),
            "deadline_deferred": sorted(self.deadline_deferred),
            "admission_paused": sorted(self.admission_paused),
            "decode_progress": dict(self.decode_progress),
            "support_cache": self.support.snapshot(),
            "chain_state": self.state.snapshot(),
            "events": self.audit_events,
        }
