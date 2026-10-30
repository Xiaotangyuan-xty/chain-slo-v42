from v2_contracts import GoalView, ObservedCall
from vllm.sequence import SequenceStatus


class ChainState:
    def __init__(self, stage_counts):
        self.stage_counts = stage_counts
        self.current = {}
        self.meta = {}

    def add(self, group, now, active_collections):
        collection = group.collection_id
        stage = group.request_info.stage_id
        reset = False
        previous_history = 0
        if stage == 0 and collection not in active_collections:
            if collection in self.meta and self.meta[collection]["history"]:
                reset = True
                previous_history = len(self.meta[collection]["history"])
            arrival = now - (group.arrival_time - __import__("time").time())
            self.meta[collection] = {
                "arrival": arrival,
                "deadline": arrival + group.request_info.slo_constraint[2] * self.stage_counts[collection],
                "history": [],
                "virtual": 0.0,
                "completed_stages": set(),
                "declared_history": {},
                "tool_entered": None,
            }
        if collection not in self.meta:
            arrival = now - (group.arrival_time - __import__("time").time())
            self.meta[collection] = {
                "arrival": arrival,
                "deadline": arrival + group.request_info.slo_constraint[2] * self.stage_counts[collection],
                "history": [],
                "virtual": 0.0,
                "completed_stages": set(),
                "declared_history": {},
                "tool_entered": None,
            }
        self.current[group.request_id] = {
            "level": 0,
            "quantum": 0.0,
            "ready_since": now,
            "service": 0.0,
            "wait": 0.0,
        }
        return reset, previous_history

    def finish(self, group, now, remaining_stage_groups):
        state = self.current.pop(group.request_id, None)
        if state is None:
            return None
        collection = group.collection_id
        stage = group.request_info.stage_id
        meta = self.meta[collection]
        aborted = group.first_seq.status == SequenceStatus.FINISHED_ABORTED
        if not aborted:
            meta["history"].append(ObservedCall(
                ordinal=stage,
                prompt_tokens=group.first_seq.get_prompt_len(),
                outputs=group.first_seq.get_output_len(),
                finished=now,
                service=state["service"],
                wait=state["wait"],
            ))
            meta["declared_history"][stage] = (meta["declared_history"].get(stage, 0)
                                                + group.request_info.input_len)
            if not remaining_stage_groups:
                meta["completed_stages"].add(stage)
                meta["tool_entered"] = now if stage + 1 < self.stage_counts[collection] else None
        return state

    def abort(self, group):
        return self.current.pop(group.request_id, None)

    def after_step(self, group, span, now, selection_time, selection_ready, decode_seconds):
        if group.request_id not in self.current:
            return
        state = self.current[group.request_id]
        if group.request_id in selection_ready:
            state["wait"] += max(0.0, selection_time - selection_ready[group.request_id])
        state["service"] += span
        state["ready_since"] = now
        state["quantum"] += decode_seconds
        self.meta[group.collection_id]["virtual"] += span / max(1, len(selection_ready))
        quantum = 64 * decode_seconds
        if state["quantum"] >= quantum * (2 ** state["level"]):
            state["level"] = min(3, state["level"] + 1)
            state["quantum"] = 0.0

    def away_goals(self, now, ready_collections):
        goals = []
        for collection, meta in self.meta.items():
            if collection in ready_collections:
                continue
            if len(meta["completed_stages"]) >= self.stage_counts[collection]:
                continue
            entered = meta["tool_entered"]
            if entered is None or not meta["history"]:
                continue
            goals.append(GoalView(
                program=collection,
                arrival=meta["arrival"],
                deadline=meta["deadline"],
                state="AWAY",
                history=tuple(meta["history"]),
                tool_entered=entered,
                tool_elapsed=max(0.0, now - entered),
                produced_prefix=0,
            ))
        return goals

    def snapshot(self):
        return {
            collection: {
                "arrival": meta["arrival"],
                "deadline": meta["deadline"],
                "virtual": meta["virtual"],
                "completed_stages": sorted(meta["completed_stages"]),
                "tool_entered": meta["tool_entered"],
                "history_length": len(meta["history"]),
            }
            for collection, meta in self.meta.items()
        }
