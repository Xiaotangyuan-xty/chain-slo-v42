from dataclasses import replace
import time

from candidate_evaluator import candidate_action
from contracts import CacheEntry, CallView
from pool_engine import PoolCache
from v2_contracts import GoalView, View
from v2_policy import V2Policy, V2Scheduler
from vllm.core.scheduler import Scheduler


class V2CompoundPolicy(V2Policy):
    version = "v2_compound_stage_eval_v1"

    def __init__(self, *args, batch_slots=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_slots = batch_slots
        self.support_cache = {}
        self.compound_members = {}

    def _view(self, groups, now, ledger):
        grouped = {}
        for group in groups:
            key = (group.collection_id, group.request_info.stage_id)
            grouped.setdefault(key, []).append(group)
        goals = []
        self.compound_members = {}
        for (collection, stage), members in sorted(grouped.items()):
            members = sorted(members, key=lambda group: (group.arrival_time, group.request_id))
            composite_key = f"{collection}:stage:{stage}"
            self.compound_members[composite_key] = [group.request_id for group in members]
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
                key=composite_key,
                program=collection,
                ordinal=stage,
                prompt_tokens=sum(group.first_seq.get_prompt_len() for group in members),
                emitted=sum(group.first_seq.get_output_len() for group in members),
                produced_prefix=cache_entry.gpu,
                level=min(state["level"] for state in states),
                quantum_used=sum(state["quantum"] for state in states),
                ready_wait=max(now - state["ready_since"] for state in states),
                service=sum(state["service"] for state in states),
                virtual_service=sum(state["virtual"] for state in states),
                running=False,
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
        entries = []
        for goal in goals:
            collection = goal.program
            members = grouped[(collection, goal.ready.ordinal)]
            ledger_entries = [ledger.view().get(group.request_id) for group in members]
            entries.append(CacheEntry(
                key=f"0:{collection}",
                gpu=sum(entry.gpu for entry in ledger_entries),
                host=sum(entry.host for entry in ledger_entries),
                gpu_touch=max(entry.gpu_touch for entry in ledger_entries),
                host_touch=max(entry.host_touch for entry in ledger_entries),
                pending=sum(entry.pending for entry in ledger_entries),
            ))
        owners = {goal.program: 0 for goal in goals}
        cache = PoolCache(ledger.gpu_capacity, ledger.host_capacity, tuple(entries),
                          tuple(sorted(owners.items())))
        return View(now, tuple(goals), cache)

    def before_schedule(self, groups, now, ledger):
        self.priorities = {}
        self.compound_members = {}
        if not groups:
            self.anchor = None
            self.remaining = 0.0
            self.forced = None
            return
        view = self._view(groups, now, ledger)
        q = 64 * self.perf.decode_seconds
        floor = min(self.current[group.request_id]["virtual"] for group in groups)
        eligible_groups = [group for group in groups
                           if self.current[group.request_id]["virtual"] <= floor + 2 * q]
        eligible_composites = {group.ready.key for group in view.goals
                               if all(member in {item.request_id for item in eligible_groups}
                                      for member in self.compound_members[group.ready.key])}
        oldest = max(groups, key=lambda group: now - self.current[group.request_id]["ready_since"])
        self.forced = None
        if now - self.current[oldest.request_id]["ready_since"] > 8 * q:
            self.forced = next((goal.ready.key for goal in view.goals
                                if oldest.request_id in self.compound_members[goal.ready.key]), None)
        if self.anchor not in eligible_composites or self.remaining <= 0:
            support = True
            for goal in view.goals:
                prompt = goal.ready.prompt_tokens
                ordinal = goal.ready.ordinal
                key = (prompt, ordinal)
                if key not in self.support_cache:
                    try:
                        self.pred.ready_base(prompt, ordinal, 0)
                        self.support_cache[key] = True
                        self.record("v2_support_gate", supported=True, prompt=prompt, ordinal=ordinal)
                    except ValueError as error:
                        self.support_cache[key] = False
                        self.record("v2_support_gate", supported=False, prompt=prompt,
                                    ordinal=ordinal, error=str(error))
                support = support and self.support_cache[key]
            owners = {goal.program: 0 for goal in view.goals}
            if support:
                (self.anchor, self.remaining), audit = candidate_action(
                    view, 0, eligible_composites, self.pred, self.perf, owners,
                    batch_slots=self.batch_slots, compound_members=self.compound_members)
                fallback = str(audit.get("fallback"))
                if fallback != "no_model_gain":
                    for goal in view.goals:
                        self.support_cache[(goal.ready.prompt_tokens, goal.ready.ordinal)] = False
                    self.record("v2_support_gate", supported=False,
                                reason=fallback)
            else:
                default = min(view.calls, key=lambda call: (call.level, -call.ready_wait, call.key))
                self.anchor = default.key
                self.remaining = 4 * self.perf.decode_seconds
                audit = {"pi0": default.key, "actions": [], "accepted": False,
                         "fallback": "support_gate", "seconds": 0.0}
            if self.first_view is None:
                self.first_view = view
                self.first_legal = set(eligible_composites)
            self.record("v2_candidate", anchor=self.anchor, remaining=self.remaining,
                        eligible=sorted(eligible_composites), forced=self.forced,
                        compound_members=self.compound_members,
                        audit=audit, view=view.to_dict())
        for goal in view.goals:
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
                    request_id,
                )

    def after_model_step(self, span, now):
        selected = self.last_selected
        if not selected:
            return
        for group in selected:
            if group.request_id not in self.current:
                continue
            state = self.current[group.request_id]
            if group.request_id in self.selection_ready:
                state["wait"] += max(0.0, self.selection_time - self.selection_ready[group.request_id])
            state["service"] += span
            state["virtual"] += span / len(selected)
            state["ready_since"] = now
            state["quantum"] += self.perf.decode_seconds
            q = 64 * self.perf.decode_seconds
            if state["quantum"] >= q * (2 ** state["level"]):
                state["level"] = min(3, state["level"] + 1)
                state["quantum"] = 0.0
            composite = next((key for key, members in self.compound_members.items()
                              if group.request_id in members), None)
            if composite == self.anchor:
                self.remaining -= self.perf.decode_seconds
        self.record("v2_after_step", selected=[group.request_id for group in selected],
                    span=span, anchor=self.anchor, remaining=self.remaining)

    def snapshot(self):
        result = super().snapshot()
        result.update({"mechanism_version": self.version,
                       "support_cache": {str(key): value for key, value in self.support_cache.items()}})
        return result


class V2CompoundScheduler(V2Scheduler):
    def add_seq_group(self, seq_group):
        Scheduler.add_seq_group(self, seq_group)

    def note_selection(self, selected, now):
        selected_ids = {group.request_id for group in selected}
        for composite, members in self.policy.compound_members.items():
            selected_members = [request_id for request_id in members if request_id in selected_ids]
            if selected_members and len(selected_members) != len(members):
                raise RuntimeError(f"Partial compound stage group {composite}: {selected_members} of {members}")
        super().note_selection(selected, now)


def install_v2_compound(engine, perf, pred, stage_counts=None):
    old = engine.engine.scheduler[0]
    policy = V2CompoundPolicy(perf, pred, stage_counts=stage_counts,
                              batch_slots=old.scheduler_config.max_num_seqs)
    scheduler = V2CompoundScheduler(
        old.scheduler_config,
        old.cache_config,
        old.lora_config,
        1,
        None,
        v2_policy=policy,
    )
    scheduler.v2_ledger = engine.engine.model_executor.backend.ledger
    engine.engine.scheduler[0] = scheduler
    return scheduler
