"""Frozen candidate generation with lease-aware, compound-slot evaluation."""
import math
import time

from lease_reference import VERSION as EVALUATION_VERSION, project
from v2_reference import forecasts


def _different(baseline, candidate):
    return (baseline["good"] != candidate["good"]
            or not math.isclose(baseline["mean"], candidate["mean"],
                                rel_tol=1e-12, abs_tol=1e-12))


def candidate_action(view, instance, eligible, pred, perf, owners,
                     batch_slots=4, compound_members=None):
    start = time.perf_counter()
    quantum = 64 * perf.decode_seconds
    calls = [call for call in view.calls if call.key in eligible]
    if not calls:
        raise ValueError("no eligible candidate")
    baseline_key = min(calls, key=lambda call: (
        call.level, -call.ready_wait, call.key)).key
    baseline_horizon = 4 * perf.decode_seconds
    compound_members = compound_members or {}
    audit = {
        "pi0": baseline_key,
        "actions": [],
        "accepted": False,
        "fallback": None,
        "evaluation_version": EVALUATION_VERSION,
        "batch_slots": batch_slots,
        "candidate_discriminability": "not_evaluated",
    }
    try:
        jobs = forecasts(view, pred, perf)
        goals = {goal.program: goal for goal in view.goals}
        for job in jobs:
            goal = goals[job["program"]]
            job["instance"] = owners[goal.program]
            call = goal.ready
            if call:
                job["context"] = call.prompt_tokens + call.emitted
                job["level"] = call.level
                job["ready_wait"] = call.ready_wait
                job["slots"] = len(compound_members.get(call.key, (call.key,)))
            else:
                last = goal.history[-1]
                job["context"] = last.prompt_tokens + last.outputs
                job["slots"] = 1
        by_key = {job["key"]: job for job in jobs}
        actions = [(baseline_key, baseline_horizon)]
        for call in calls:
            for horizon in (baseline_horizon, max(
                    perf.decode_seconds, min(quantum, by_key[call.key]["current"]))):
                if (call.key, horizon) not in actions:
                    actions.append((call.key, horizon))
        for key, horizon in actions:
            score = project(jobs, (key, horizon), view.now, perf, instance,
                            start + 0.02, batch_slots=batch_slots)
            audit["actions"].append({"key": key, "h": horizon, **score})
        baseline = audit["actions"][0]
        for candidate in audit["actions"][1:]:
            candidate["score_differs_from_baseline"] = _different(baseline, candidate)
            candidate["finish_differs_from_baseline"] = any(
                not math.isclose(baseline["finishes"][key], finish,
                                 rel_tol=1e-12, abs_tol=1e-12)
                for key, finish in candidate["finishes"].items())
        discriminating = any(candidate["score_differs_from_baseline"]
                             for candidate in audit["actions"][1:])
        audit["candidate_discriminability"] = (
            "distinguished" if discriminating else "all_scores_tied")
        winner = max(range(len(actions)), key=lambda index: (
            (audit["actions"][index]["good"], -audit["actions"][index]["mean"]),
            -index,
        ))
        if winner == 0:
            audit["fallback"] = "no_model_gain"
            audit["fallback_diagnosis"] = (
                "baseline_best_among_distinguished_actions" if discriminating
                else "evaluation_has_no_distinguishing_action")
        audit["accepted"] = winner != 0
        audit["seconds"] = time.perf_counter() - start
        return actions[winner], audit
    except (ValueError, RuntimeError, TimeoutError) as error:
        audit.update(fallback=str(error), seconds=time.perf_counter() - start,
                     candidate_discriminability="evaluation_failed")
        return (baseline_key, baseline_horizon), audit
