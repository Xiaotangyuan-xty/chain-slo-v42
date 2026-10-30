import math
import time

from lease_reference import (VERSION as EVALUATION_VERSION, compare_scores,
                             first_plan_for_action, project)
from execution_reference import forecasts


def _score_differs(baseline, candidate):
    return compare_scores(baseline, candidate) != 0


def _finish_differs(baseline, candidate):
    return any(abs(baseline["finishes"][key] - finish) > 1e-9
               for key, finish in candidate["finishes"].items())


def candidate_action(view, instance, eligible, pred, perf, owners,
                     batch_slots=4, compound_members=None, project_fn=None,
                     evaluation_seconds=0.75, decision_conformal=False):
    start = time.perf_counter()
    decode_seconds = perf.decode_seconds
    quantum = 64 * decode_seconds
    calls = [call for call in view.calls if call.key in eligible]
    if not calls:
        raise ValueError("no eligible candidate")
    if all(view.native_priority_for(call.key) is not None for call in calls):
        baseline_key = min(calls, key=lambda call: view.native_priority_for(call.key)).key
    else:
        baseline_key = min(calls, key=lambda call: (
            call.level, -call.ready_wait, call.key)).key
    baseline_horizon = 4 * decode_seconds
    compound_members = compound_members or {}
    project_fn = project_fn or project
    resource = view.resource_contract() if hasattr(view, "resource_contract") else None
    audit = {
        "pi0": baseline_key,
        "actions": [],
        "accepted": False,
        "fallback": None,
        "evaluation_version": EVALUATION_VERSION,
        "batch_slots": batch_slots,
        "candidate_discriminability": "not_evaluated",
        "timeouts": 0,
        "project_calls": 0,
        "resource_skips": [],
        "completed_scan": False,
        "preflight_reason": "potential_contention",
        "decision_conformal": decision_conformal,
    }
    ready = [goal.ready for goal in view.goals if goal.ready is not None]
    ready_slots = sum(len(compound_members.get(call.key, (call.key,))) for call in ready)
    preflight = None
    if len(view.goals) == 1:
        preflight = "single_goal"
    elif len(ready) == len(view.goals) and ready_slots <= batch_slots and resource is None:
        preflight = "all_ready_units_fit_fixed_batch"
    if preflight is not None:
        audit.update(fallback="degenerate", candidate_discriminability="degenerate",
                     preflight_reason=preflight, seconds=time.perf_counter() - start)
        return (baseline_key, baseline_horizon), audit
    try:
        jobs = forecasts(view, pred, perf, conformal=decision_conformal)
        goals = {}
        for goal in view.goals:
            key = goal.ready.key if goal.ready is not None else f"away:{goal.program}"
            if key in goals:
                raise ValueError("duplicate goal identity")
            goals[key] = goal
        for job in jobs:
            goal = goals[job["key"]]
            job["instance"] = owners[goal.program]
            job["slots"] = len(compound_members.get(job["key"], (job["key"],)))
            job["virtual"] = view.virtual_for(goal.program) if hasattr(view, "virtual_for") else (
                goal.ready.virtual_service if goal.ready is not None else 0.0)
            if goal.ready is not None:
                call = goal.ready
                job["context"] = call.prompt_tokens + call.emitted
                job["level"] = call.level
                job["quantum"] = call.quantum_used
                job["ready_wait"] = call.ready_wait
                job["running"] = call.running
            else:
                last = goal.history[-1]
                job["context"] = last.prompt_tokens + last.outputs
                job["level"] = 0
                job["quantum"] = 0.0
                job["ready_wait"] = 0.0
            info = view.resource_for(job["key"]) or {} if hasattr(view, "resource_for") else {}
            job["tokens"] = info.get("tokens", 0)
            job["kv_required"] = info.get("required_blocks", 0 if goal.ready is None else None)
            job["resident_blocks"] = info.get("resident_blocks", 0)
            job["protected"] = info.get("protected", False)
            job["resource_reason"] = info.get("reason")
            job["queue_rank"] = info.get("queue_rank", len(jobs))
            job["dispatch_status"] = info.get("status", "AWAY" if goal.ready is None else None)
            job["decode_progress"] = info.get("decode_progress", 0)
            job["gpu_tokens"] = info.get("gpu_tokens", 0)
            job["host_tokens"] = info.get("host_tokens", 0)
            if hasattr(view, "native_priority_for"):
                job["native_priority"] = view.native_priority_for(job["key"])
        by_key = {job["key"]: job for job in jobs}
        if len(ready) == len(view.goals) and ready_slots <= batch_slots and resource is not None:
            baseline_plan = first_plan_for_action(
                jobs, (baseline_key, baseline_horizon), view.now, perf, instance,
                batch_slots, resource)
            admitted = {unit["ids"][0] for unit in baseline_plan["plan"]["accepted"]}
            if len(admitted) == len(jobs):
                audit.update(fallback="degenerate", candidate_discriminability="degenerate",
                             preflight_reason="all_ready_units_fit_real_batch",
                             seconds=time.perf_counter() - start)
                return (baseline_key, baseline_horizon), audit
        signatures = {
            (job["deadline"], job["instance"], job["slots"], job["context"],
             job["virtual"], job["level"], job["ready_wait"], tuple(job["phases"]))
            for job in jobs
        }
        if resource is None and len(signatures) == 1 and sum(job["slots"] for job in jobs) <= batch_slots:
            audit.update(fallback="degenerate", candidate_discriminability="degenerate",
                         seconds=time.perf_counter() - start)
            return (baseline_key, baseline_horizon), audit

        actions = [(baseline_key, baseline_horizon)]
        for call in calls:
            for horizon in (baseline_horizon, max(
                    decode_seconds, min(quantum, by_key[call.key]["current"]))):
                if (call.key, horizon) not in actions:
                    actions.append((call.key, horizon))
        audit["planned_actions"] = len(actions)
        for key, horizon in actions:
            if resource is not None and key != baseline_key:
                feasible = first_plan_for_action(
                    jobs, (key, horizon), view.now, perf, instance, batch_slots, resource)
                admitted = {unit["ids"][0] for unit in feasible["plan"]["accepted"]}
                if key not in admitted:
                    audit["resource_skips"].append({
                        "key": key, "h": horizon, "reason": "resource_infeasible",
                        "deferred": feasible["plan"]["deferred"],
                    })
                    continue
            audit["project_calls"] += 1
            try:
                kwargs = {"batch_slots": batch_slots}
                if resource is not None:
                    kwargs["resource"] = resource
                score = project_fn(jobs, (key, horizon), view.now, perf, instance,
                                   start + evaluation_seconds, **kwargs)
            except TimeoutError:
                audit["timeouts"] += 1
                break
            audit["actions"].append({"key": key, "h": horizon, **score})
        else:
            audit["completed_scan"] = True
        audit["unevaluated_actions"] = 0 if audit["completed_scan"] else len(actions) - len(audit["actions"])
        if not audit["actions"]:
            audit.update(fallback="execution-aligned reference budget",
                         candidate_discriminability="evaluation_failed",
                         seconds=time.perf_counter() - start)
            return (baseline_key, baseline_horizon), audit
        baseline = audit["actions"][0]
        for candidate in audit["actions"][1:]:
            candidate["score_differs_from_baseline"] = _score_differs(baseline, candidate)
            candidate["finish_differs_from_baseline"] = _finish_differs(baseline, candidate)
        discriminating = any(candidate["score_differs_from_baseline"]
                             and candidate["finish_differs_from_baseline"]
                             for candidate in audit["actions"][1:])
        finish_distinct = any(candidate["finish_differs_from_baseline"]
                              for candidate in audit["actions"][1:])
        audit["candidate_discriminability"] = (
            "distinguished" if discriminating
            else "invariant" if not finish_distinct else "all_scores_tied")
        winner = 0
        for index in range(1, len(audit["actions"])):
            if compare_scores(audit["actions"][index], audit["actions"][winner]) > 0:
                winner = index
        if winner != 0 and not (
                compare_scores(audit["actions"][winner], baseline) > 0
                and audit["actions"][winner]["finish_differs_from_baseline"]):
            winner = 0
        reject_reason = None
        if not audit["completed_scan"]:
            winner = 0
            reject_reason = "incomplete_evaluation"
        if winner != 0:
            chosen = audit["actions"][winner]
            incremental = chosen.get("switch_cost", 0.0) - baseline.get("switch_cost", 0.0)
            mean_gain = baseline["mean"] - chosen["mean"]
            net_gain = mean_gain - incremental
            audit["incremental_switch_cost"] = incremental
            audit["net_mean_gain"] = net_gain
            if chosen["tardiness"] < baseline["tardiness"]:
                objective_gain = baseline["tardiness"] - chosen["tardiness"] - incremental
                objective_dimension = "tardiness"
            else:
                objective_gain = net_gain
                objective_dimension = "mean"
            audit["net_objective_gain"] = objective_gain
            audit["net_objective_dimension"] = objective_dimension
            if chosen["good"] <= baseline["good"] and mean_gain < 0:
                audit["rejected_action"] = {"key": chosen["key"], "h": chosen["h"],
                                            "mean_gain": mean_gain,
                                            "objective_dimension": objective_dimension,
                                            "incremental_switch_cost": incremental}
                winner = 0
                reject_reason = "negative_mean_gain"
            elif chosen["good"] <= baseline["good"] and incremental > 0 and objective_gain <= 0:
                audit["rejected_action"] = {"key": chosen["key"], "h": chosen["h"],
                                            "mean_gain": mean_gain,
                                            "objective_dimension": objective_dimension,
                                            "incremental_switch_cost": incremental}
                winner = 0
                reject_reason = "insufficient_net_gain"
        if winner == 0:
            audit["fallback"] = "no_model_gain"
            audit["fallback_diagnosis"] = reject_reason or (
                "baseline_best_among_distinguished_actions" if discriminating
                else "evaluation_has_no_distinguishing_action")
        audit["accepted"] = winner != 0
        audit["seconds"] = time.perf_counter() - start
        chosen = audit["actions"][winner]
        return (chosen["key"], chosen["h"]), audit
    except (ValueError, RuntimeError) as error:
        audit.update(fallback=str(error), seconds=time.perf_counter() - start,
                     candidate_discriminability="evaluation_failed")
        return (baseline_key, baseline_horizon), audit
