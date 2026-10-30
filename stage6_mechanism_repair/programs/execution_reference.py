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


def forecasts(view, predictor, profile, conformal=True):
    jobs = []
    compound_members = dict(getattr(view, "compound_members", ()))
    for goal in view.goals:
        original = goal.ready
        key = original.key if original is not None else f"away:{goal.program}"
        declared = view.declared_for(key) if hasattr(view, "declared_for") else None
        slots = len(compound_members.get(key, (key,)))
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
        if "slots" in inspect.signature(predictor.estimate).parameters:
            estimate = predictor.estimate(prediction_goal, profile, conformal, slots=slots)
        else:
            estimate = predictor.estimate(prediction_goal, profile, conformal)
        call = current
        prefix = ((call.prompt_tokens + max(0, call.emitted - 1)
                   if call.emitted else call.produced_prefix) if call else goal.produced_prefix)
        recovery = safe_recovery(profile, view.cache.get(goal.program), prefix)
        phases = list(estimate["phases"])
        if recovery:
            index = next(i for i, phase in enumerate(phases) if phase[0] == "compute")
            phases[index] = ("compute", phases[index][1] + recovery)
        jobs.append({
            "key": key,
            "program": goal.program,
            "state": goal.state,
            "deadline": goal.deadline,
            **estimate,
            "work": estimate["work"] + recovery,
            "point": estimate["point"] + recovery,
            "current": estimate["current"] + recovery if call else 0.0,
            "recovery": recovery,
            "phases": tuple(phases),
        })
    return jobs
