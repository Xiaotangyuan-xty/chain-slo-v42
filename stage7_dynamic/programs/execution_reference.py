from dataclasses import replace
import inspect

from conditional_predictor import safe_prefill


def safe_recovery(profile, entry, needed):
    try:
        return profile.recovery(entry, needed)
    except ValueError:
        if entry.gpu >= needed:
            return 0.0
        available = max(entry.gpu, min(entry.host, needed))
        copied = max(0, available - entry.gpu) * profile.copy_seconds_per_token
        return copied + safe_prefill(profile, needed - available, available)


def recovery_parts(profile, entry, needed):
    needed = max(0, int(needed))
    if entry.gpu >= needed:
        return {"needed": needed, "available": needed, "copied_tokens": 0,
                "new_tokens": 0, "copy_seconds": 0.0, "prefill_seconds": 0.0,
                "seconds": 0.0}
    available = max(entry.gpu, min(entry.host, needed))
    copied = max(0, available - entry.gpu)
    new = max(0, needed - available)
    prefill = safe_prefill(profile, new, available)
    total = safe_recovery(profile, entry, needed)
    prefill = min(total, prefill)
    return {"needed": needed, "available": available, "copied_tokens": copied,
            "new_tokens": new, "copy_seconds": max(0.0, total - prefill),
            "prefill_seconds": prefill, "seconds": total}


def _cache_entry(view, goal):
    entry = view.cache.get(goal.program)
    alternate = view.cache.get(f"0:{goal.program}")
    if alternate.gpu + alternate.host + alternate.pending > entry.gpu + entry.host + entry.pending:
        entry = alternate
    produced = max(goal.produced_prefix,
                   goal.ready.produced_prefix if goal.ready is not None else 0)
    if produced > entry.gpu:
        entry = replace(entry, gpu=produced)
    return entry


def forecasts(view, predictor, profile, conformal=True):
    jobs = []
    compound_members = dict(getattr(view, "compound_members", ()))
    parameters = inspect.signature(predictor.estimate).parameters
    for goal in view.goals:
        original = goal.ready
        key = original.key if original is not None else f"away:{goal.program}"
        declared = view.declared_for(key) if hasattr(view, "declared_for") else None
        slots = len(compound_members.get(key, (key,)))
        member_progress = (view.member_progress_for(key)
                           if hasattr(view, "member_progress_for") else None)
        prediction_goal = goal
        if original is not None:
            if declared is not None:
                prediction_goal = replace(goal, ready=replace(original, prompt_tokens=declared))
            current = original
        else:
            last = goal.history[-1]
            if declared is not None:
                history = list(goal.history)
                history[-1] = replace(last, prompt_tokens=declared)
                prediction_goal = replace(goal, history=tuple(history))
            current = None
        kwargs = {}
        if "slots" in parameters:
            kwargs["slots"] = slots
        if "member_progress" in parameters:
            kwargs["member_progress"] = member_progress
        estimate = predictor.estimate(prediction_goal, profile, conformal, **kwargs)
        phases = list(estimate["phases"])
        metadata = list(estimate.get("phase_metadata", ()))
        if len(metadata) != len(phases):
            metadata = [{"kind": kind, "ordinal": (original.ordinal if original else 0),
                         "width": slots, "prefill_tokens": 0}
                        for kind, _ in phases]
        parts = {"seconds": 0.0, "copy_seconds": 0.0, "prefill_seconds": 0.0,
                 "new_tokens": 0, "needed": 0, "copied_tokens": 0}
        if current is not None:
            prefix = (current.prompt_tokens + max(0, current.emitted - slots)
                      if current.emitted else current.prompt_tokens)
            parts = recovery_parts(profile, _cache_entry(view, goal), prefix)
            recovery = parts["seconds"]
            index = next(i for i, phase in enumerate(phases) if phase[0] == "compute")
            if recovery:
                phases[index] = ("compute", phases[index][1] + recovery)
            metadata[index] = dict(metadata[index], prefill_tokens=parts["new_tokens"],
                                   prefill_seconds=parts["prefill_seconds"],
                                   recovery_seconds=parts["copy_seconds"])
        for index, (kind, _) in enumerate(phases):
            if kind == "compute" and metadata[index].get("prefill_tokens", 0):
                metadata[index] = dict(metadata[index])
                metadata[index].setdefault(
                    "prefill_seconds", safe_prefill(profile, metadata[index]["prefill_tokens"]))
        recovery = parts["seconds"]
        support = dict(estimate.get("support", {}))
        support["cache_prefix_tokens"] = parts["needed"]
        support["recovery_new_tokens"] = parts["new_tokens"]
        jobs.append({
            "key": key,
            "program": goal.program,
            "state": goal.state,
            "deadline": goal.deadline,
            **estimate,
            "work": estimate["work"] + recovery,
            "point": estimate["point"] + recovery,
            "current": estimate["current"] + recovery if current else 0.0,
            "recovery": recovery,
            "recovery_copy": parts["copy_seconds"],
            "prefill_seconds": parts["prefill_seconds"],
            "prefill_tokens": parts["new_tokens"],
            "member_progress": member_progress,
            "phases": tuple(phases),
            "phase_metadata": tuple(metadata),
            "support": support,
        })
    return jobs
