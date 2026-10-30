from collections import deque
from dataclasses import replace
import time

from contracts import CacheEntry
from pool_engine import PoolCache, candidate_action
from v2_contracts import GoalView, ObservedCall, View
from vllm.core.scheduler import Scheduler
from vllm.sequence import SequenceStatus
from jitserve.scheduler.policy import BasePolicy


class V2Policy(BasePolicy):
    def __init__(self, perf, pred, penalty_factor=100, stage_counts=None):
        super().__init__(schedule_interval=50, penalty_factor=penalty_factor)
        self.perf = perf
        self.pred = pred
        self.stage_counts = stage_counts or {}
        self.current = {}
        self.chain_meta = {}
        self.priorities = {}
        self.anchor = None
        self.remaining = 0.0
        self.forced = None
        self.audit_events = []
        self.first_view = None
        self.first_legal = None
        self.selection_ready = {}
        self.selection_time = None
        self.last_selected = []

    def add_seq_group(self, seq_group):
        collection = seq_group.collection_id
        if seq_group.request_info.stage_id == 0 and not self.seq_group_dict.get(collection):
            if collection in self.chain_meta and self.chain_meta[collection]["history"]:
                self.record("v2_chain_reset", collection_id=collection,
                            previous_history=len(self.chain_meta[collection]["history"]))
            arrival = time.perf_counter() - (time.time() - seq_group.arrival_time)
            self.chain_meta[collection] = {
                "arrival": arrival,
                "deadline": arrival + seq_group.request_info.slo_constraint[2] * self.stage_counts.get(collection, 6),
                "history": [],
            }
            self.anchor = None
            self.remaining = 0.0
            self.forced = None
        super().add_seq_group(seq_group)
        now = time.perf_counter()
        if collection not in self.chain_meta:
            arrival = now - (time.time() - seq_group.arrival_time)
            self.chain_meta[collection] = {
                "arrival": arrival,
                "deadline": arrival + seq_group.request_info.slo_constraint[2] * self.stage_counts.get(collection, 6),
                "history": [],
            }
        self.current[seq_group.request_id] = {
            "level": 0,
            "quantum": 0.0,
            "virtual": 0.0,
            "ready_since": now,
            "service": 0.0,
            "wait": 0.0,
        }
        self.record("v2_admit", request_id=seq_group.request_id, collection_id=collection,
                    stage_id=seq_group.request_info.stage_id)

    def delete_seq_group(self, seq_group):
        state = self.current.pop(seq_group.request_id, None)
        if state is not None and seq_group.first_seq.status != SequenceStatus.FINISHED_ABORTED:
            history = self.chain_meta[seq_group.collection_id]["history"]
            history.append(ObservedCall(
                ordinal=seq_group.request_info.stage_id,
                prompt_tokens=seq_group.first_seq.get_prompt_len(),
                outputs=seq_group.first_seq.get_output_len(),
                finished=time.perf_counter(),
                service=state["service"],
                wait=state["wait"],
            ))
            self.record("v2_finish", request_id=seq_group.request_id,
                        collection_id=seq_group.collection_id,
                        stage_id=seq_group.request_info.stage_id,
                        outputs=seq_group.first_seq.get_output_len(),
                        service=state["service"], wait=state["wait"])
        super().delete_seq_group(seq_group)

    def record(self, kind, **fields):
        self.audit_events.append({"n": len(self.audit_events), "kind": kind,
                                  "monotonic": time.perf_counter(), **fields})

    def _view(self, groups, now, ledger):
        goals = []
        entries = []
        owners = {}
        for group in groups:
            state = self.current[group.request_id]
            meta = self.chain_meta[group.collection_id]
            entry = ledger.view().get(group.request_id)
            entries.append(replace(entry, key=f"0:{group.collection_id}"))
            owners[group.collection_id] = 0
            emitted = group.first_seq.get_output_len()
            call = __import__("contracts").CallView(
                key=group.request_id,
                program=group.collection_id,
                ordinal=group.request_info.stage_id,
                prompt_tokens=group.first_seq.get_prompt_len(),
                emitted=emitted,
                produced_prefix=entry.gpu,
                level=state["level"],
                quantum_used=state["quantum"],
                ready_wait=max(0.0, now - state["ready_since"]),
                service=state["service"],
                virtual_service=state["virtual"],
                running=group in getattr(self, "running", ()),
                historical_wait=state["wait"],
            )
            goals.append(GoalView(
                program=group.collection_id,
                arrival=meta["arrival"],
                deadline=meta["deadline"],
                state="READY",
                history=tuple(meta["history"]),
                ready=call,
                produced_prefix=entry.gpu,
            ))
        cache = PoolCache(
            ledger.gpu_capacity,
            ledger.host_capacity,
            tuple(entries),
            tuple(sorted(owners.items())),
        )
        return View(now, tuple(goals), cache)

    def before_schedule(self, groups, now, ledger):
        self.priorities = {}
        if not groups:
            self.anchor = None
            self.remaining = 0.0
            self.forced = None
            return
        view = self._view(groups, now, ledger)
        q = 64 * self.perf.decode_seconds
        floor = min(self.current[group.request_id]["virtual"] for group in groups)
        eligible_groups = [
            group for group in groups
            if self.current[group.request_id]["virtual"] <= floor + 2 * q
        ]
        eligible = {group.request_id for group in eligible_groups}
        oldest = max(groups, key=lambda group: now - self.current[group.request_id]["ready_since"])
        self.forced = oldest.request_id if now - self.current[oldest.request_id]["ready_since"] > 8 * q else None
        if self.anchor not in eligible or self.remaining <= 0:
            owners = {group.collection_id: 0 for group in groups}
            (self.anchor, self.remaining), audit = candidate_action(
                view, 0, eligible, self.pred, self.perf, owners)
            if self.first_view is None:
                self.first_view = view
                self.first_legal = set(eligible)
            self.record("v2_candidate", anchor=self.anchor, remaining=self.remaining,
                        eligible=sorted(eligible), forced=self.forced, audit=audit,
                        view=view.to_dict())
        for group in groups:
            if group.request_id == self.forced:
                rank = 0
            elif group.request_id == self.anchor:
                rank = 1
            else:
                rank = 2 + self.current[group.request_id]["level"]
            self.priorities[group.request_id] = (
                rank,
                -(now - self.current[group.request_id]["ready_since"]),
                group.request_id,
            )

    def get_priority(self, seq_group):
        return self.priorities.get(seq_group.request_id, (99, seq_group.arrival_time))

    def note_selection(self, selected, now):
        self.last_selected = list(selected)
        self.selection_time = now
        self.selection_ready = {
            group.request_id: self.current[group.request_id]["ready_since"]
            for group in self.last_selected if group.request_id in self.current
        }

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
            if group.request_id == self.anchor:
                self.remaining -= self.perf.decode_seconds
        self.record("v2_after_step", selected=[group.request_id for group in selected],
                    span=span, anchor=self.anchor, remaining=self.remaining)

    def snapshot(self):
        return {
            "anchor": self.anchor,
            "remaining": self.remaining,
            "forced": self.forced,
            "decode_seconds": self.perf.decode_seconds,
            "predictor_version": self.pred.version,
            "predictor_q_cal": self.pred.q_cal,
            "events": self.audit_events,
        }


class V2Scheduler(Scheduler):
    def __init__(self, *args, v2_policy, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = v2_policy
        self.pending = deque()

    def add_seq_group(self, seq_group):
        collection = seq_group.collection_id
        active = {group.collection_id for group in list(self.waiting) + list(self.running) + list(self.swapped) + list(self.pending)}
        if collection in active:
            self.pending.append(seq_group)
            self.policy.record("v2_hold_same_collection", request_id=seq_group.request_id,
                               collection_id=collection, stage_id=seq_group.request_info.stage_id)
        else:
            super().add_seq_group(seq_group)

    def _release_pending(self):
        active = {group.collection_id for group in list(self.waiting) + list(self.running) + list(self.swapped)}
        admitted = set()
        for group in list(self.pending):
            if group.collection_id not in active and group.collection_id not in admitted:
                self.pending.remove(group)
                super().add_seq_group(group)
                admitted.add(group.collection_id)
                active.add(group.collection_id)
                self.policy.record("v2_release_same_collection", request_id=group.request_id,
                                   collection_id=group.collection_id, stage_id=group.request_info.stage_id)

    def schedule(self):
        self._release_pending()
        groups = list(self.running) + list(self.waiting) + list(self.swapped)
        self.policy.before_schedule(groups, time.perf_counter(),
                                    self.v2_ledger)
        self.policy.num_schedule_count += 1
        result = super().schedule()
        selected = [item.seq_group for item in result[1].scheduled_seq_groups]
        self.policy.note_selection(selected, time.perf_counter())
        return result

    def has_unfinished_seqs(self):
        return bool(self.pending) or super().has_unfinished_seqs()

    def get_num_unfinished_seq_groups(self):
        return len(self.pending) + super().get_num_unfinished_seq_groups()

    def abort_seq_group(self, request_id):
        for group in list(self.pending):
            if group.request_id == request_id:
                self.pending.remove(group)
                for seq in group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)
                self.finished.append(group)
                self.policy.record("v2_pending_abort", request_id=request_id)
                return
        super().abort_seq_group(request_id)


def install_v2(engine, perf, pred, stage_counts=None):
    old = engine.engine.scheduler[0]
    policy = V2Policy(perf, pred, stage_counts=stage_counts)
    scheduler = V2Scheduler(
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
