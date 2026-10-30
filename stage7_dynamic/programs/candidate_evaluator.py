from collections import OrderedDict
import copy
import hashlib
import json
import math
import time

from lease_reference import (VERSION as EVALUATION_VERSION, compare_scores,
                             first_plan_for_action, project, resolve_batch_slots)
from execution_reference import forecasts
from deadline_admission import admission_pauses


_SCORE_CACHE = OrderedDict()
_CACHE_LIMIT = 256
_TIMEOUT = {"timeout": True}


def _score_differs(baseline, candidate):
    return compare_scores(baseline, candidate) != 0


def _finish_differs(baseline, candidate):
    return any(abs(baseline["finishes"][key] - finish) > 1e-9
               for key, finish in candidate["finishes"].items())


def _freeze(value):
    if isinstance(value, dict):
        return {str(key): _freeze(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_freeze(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def _signature(jobs, action, now, instance, batch_slots, resource, perf, project_fn, eligible):
    value = {
        "jobs": sorted(jobs, key=lambda job: job["key"]), "action": action,
        "now": now, "instance": instance, "batch_slots": batch_slots,
        "resource": resource, "eligible": sorted(eligible, key=str),
        "performance": {"type": f"{type(perf).__module__}.{type(perf).__qualname__}",
                        "state": getattr(perf, "__dict__", {}),
                        "decode_seconds": perf.decode_seconds,
                        "copy_seconds": getattr(perf, "copy_seconds_per_token", 0.0)},
        "project_fn": id(project_fn), "evaluation_version": EVALUATION_VERSION,
    }
    encoded = json.dumps(_freeze(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cache_get(cache, key):
    if key not in cache:
        return None
    value = cache[key]
    if hasattr(cache, "move_to_end"):
        cache.move_to_end(key)
    return value


def _cache_put(cache, key, value):
    cache[key] = value
    if hasattr(cache, "move_to_end"):
        cache.move_to_end(key)
    while len(cache) > _CACHE_LIMIT:
        if hasattr(cache, "popitem") and hasattr(cache, "move_to_end"):
            cache.popitem(last=False)
        else:
            cache.pop(next(iter(cache)))


def _with_costs(raw, jobs, now, evaluation_cost):
    score = copy.deepcopy(raw)
    legacy_cost = (0.0 if score.get("costs_in_finishes") else
                   score.get("switch_cost", 0.0) + score.get("resume_cost", 0.0))
    shift = evaluation_cost + legacy_cost
    if shift:
        score["finishes"] = {key: value + shift for key, value in score["finishes"].items()}
        if "elapsed" in score:
            score["elapsed"] += evaluation_cost
    deadlines = {job["key"]: job["deadline"] for job in jobs}
    score["good"] = sum(score["finishes"][key] <= deadline + 1e-12
                        for key, deadline in deadlines.items())
    score["tardiness"] = math.fsum(max(0.0, score["finishes"][key] - deadline)
                                   for key, deadline in deadlines.items())
    score["mean"] = math.fsum(score["finishes"].values()) / len(deadlines) - now
    score["evaluation_cost"] = evaluation_cost + score.get("evaluation_cost", 0.0)
    score["costs_in_finishes"] = True
    score["net_objective"] = {"good": score["good"],
                              "negative_tardiness": -score["tardiness"],
                              "negative_mean": -score["mean"]}
    return score


def candidate_action(view, instance, eligible, pred, perf, owners,
                     batch_slots=4, compound_members=None, project_fn=None,
                     evaluation_seconds=0.75, decision_conformal=False,
                     evaluation_cache=None, incumbent=None, official_frame=None,
                     synchronous_evaluation=True, deadline_rescue=False,
                     deadline_rescue_only=False):
    start = time.perf_counter()
    decode_seconds = perf.decode_seconds
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
    frame_allowed = (view.is_official_frame if official_frame is None and hasattr(
        view, "is_official_frame") else True if official_frame is None else official_frame)
    audit = {
        "pi0": baseline_key, "actions": [], "accepted": False, "fallback": None,
        "evaluation_version": EVALUATION_VERSION,
        "batch_slots": batch_slots if isinstance(batch_slots, int) else repr(batch_slots),
        "candidate_discriminability": "not_evaluated", "timeouts": 0,
        "timeout_cache_hits": 0, "cache_hits": 0, "project_calls": 0,
        "resource_skips": [], "completed_scan": False,
        "preflight_reason": "full_horizon_evaluation",
        "decision_conformal": decision_conformal,
        "schedule_count": getattr(view, "schedule_count", None),
        "frame_index": getattr(view, "frame_index", None),
        "official_frame": frame_allowed,
        "baseline_on_time_protection": True,
    }
    if not frame_allowed:
        audit.update(fallback="non_official_frame", candidate_discriminability="frame_boundary",
                     seconds=time.perf_counter() - start)
        return (baseline_key, baseline_horizon), audit
    if len(view.goals) == 1:
        audit.update(fallback="degenerate", candidate_discriminability="degenerate",
                     preflight_reason="single_goal", seconds=time.perf_counter() - start)
        return (baseline_key, baseline_horizon), audit
    cache = _SCORE_CACHE if evaluation_cache is None else evaluation_cache
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
            info = (view.resource_for(job["key"]) or {}) if hasattr(view, "resource_for") else {}
            cache_entry = view.cache.get(goal.program)
            alternate = view.cache.get(f"0:{goal.program}")
            if (alternate.gpu + alternate.host + alternate.pending
                    > cache_entry.gpu + cache_entry.host + cache_entry.pending):
                cache_entry = alternate
            job["tokens"] = info.get("tokens", 0)
            job["kv_required"] = info.get("required_blocks", 0 if goal.ready is None else None)
            job["resident_blocks"] = info.get(
                "resident_blocks", math.ceil(cache_entry.gpu / (resource or {}).get("block_size", 16)))
            job["protected"] = info.get("protected", False)
            job["resource_reason"] = info.get("reason")
            job["queue_rank"] = info.get("queue_rank", len(jobs))
            job["dispatch_status"] = info.get("status", "AWAY" if goal.ready is None else None)
            job["decode_progress"] = info.get("decode_progress", 0)
            job["gpu_tokens"] = info.get("gpu_tokens", cache_entry.gpu)
            job["host_tokens"] = info.get("host_tokens", cache_entry.host)
            job["custom_forced_enabled"] = getattr(view, "custom_forced_enabled", True)
            if hasattr(view, "native_priority_for"):
                job["native_priority"] = view.native_priority_for(job["key"])
        by_key = {job["key"]: job for job in jobs}
        if deadline_rescue and resource is not None:
            slack = {job["key"]: by_key[job["key"]]["deadline"] - view.now for job in jobs}
            risk_q = max(0.0, getattr(pred, "risk_q", 0.0))
            risk_work = {
                job["key"]: job["work"] + risk_q * max(
                    16.0, 0.25 * job["work"] / decode_seconds) * decode_seconds
                for job in jobs
            }
            audit["risk_q"] = risk_q
            audit["risk_work"] = risk_work
            for job in jobs:
                job["deadline_attainable"] = risk_work[job["key"]] <= slack[job["key"]]
            paused = admission_pauses(
                risk_work, {job["key"]: job["deadline"] for job in jobs}, view.now,
                {job["key"] for job in jobs if job["state"] != "AWAY"},
                defer_hopeless=getattr(view, "defer_hopeless", False),
                defer_safe_later=getattr(view, "defer_safe_later", False),
                frame_reserve=getattr(view, "admission_frame_reserve", 0.0),
                recovery_costs={job["key"]: 2 * job.get("gpu_tokens", 0)
                                * getattr(perf, "copy_seconds_per_token", 0.) for job in jobs})
            for job in jobs:
                job["admission_paused"] = job["key"] in paused
            protected = {job["key"] for job in jobs if job["deadline_attainable"]}
            attainable = [job for job in jobs if job["key"] in protected
                          and job["key"] in eligible and not job.get("admission_paused")]
            hopeless = any(risk_work[job["key"]] > slack[job["key"]] for job in jobs)
            if hopeless and attainable:
                if incumbent is not None or getattr(view, "incumbent_action", None) is not None:
                    selection_rule = "incumbent_least_laxity"
                    rescue = min(attainable, key=lambda job: (
                        slack[job["key"]] / risk_work[job["key"]],
                        by_key[job["key"]]["deadline"], job["key"]))
                else:
                    selection_rule = "earliest_deadline_first"
                    rescue = min(attainable, key=lambda job: (
                        by_key[job["key"]]["deadline"], -risk_work[job["key"]], job["key"]))
                if rescue["key"] != baseline_key:
                    schedule_count = getattr(view, "schedule_count", None)
                    schedule_interval = getattr(view, "schedule_interval", 50)
                    lease_steps = (max(1, schedule_interval - schedule_count % schedule_interval)
                                   if schedule_count is not None else 1)
                    lease = lease_steps * decode_seconds
                    plan = first_plan_for_action(
                        jobs, (rescue["key"], lease), view.now, perf, instance,
                        resolve_batch_slots(batch_slots, 1, instance), resource)
                    accepted = plan["plan"]["accepted"]
                    accepted_keys = {unit["ids"][0] for unit in accepted}
                    evicted = [unit["ids"] for unit, _, _ in plan["plan"]["evictions"]]
                    safe_evictions = {}
                    for key in (key for unit in evicted for key in unit):
                        if key not in protected:
                            continue
                        recovery = (2 * by_key[key].get("gpu_tokens", 0)
                                    * getattr(perf, "copy_seconds_per_token", 0.0))
                        if risk_work[key] + lease + recovery <= slack[key]:
                            safe_evictions[key] = {
                                "deadline": by_key[key]["deadline"], "slack": slack[key],
                                "risk_work": risk_work[key], "recovery_seconds": recovery,
                            }
                    protected_evictions = sorted({key for unit in evicted for key in unit
                                                  if key in protected and key not in safe_evictions})
                    if protected_evictions:
                        audit["resource_skips"].append({
                            "key": rescue["key"], "reason": "rescue_evicts_attainable_chain",
                            "evicted": evicted, "protected_evictions": protected_evictions,
                            "safe_evictions": safe_evictions, "switch_cost": plan["switch_cost"],
                        })
                    elif rescue["key"] in accepted_keys:
                        duration = perf.iteration(
                            plan["used_slots"], math.fsum(job["context"] for job in plan["selected"]))
                        trace = {
                            "step": 1, "instance": instance,
                            "keys": [unit["ids"][0] for unit in accepted],
                            "slots": plan["used_slots"], "duration": duration,
                            "anchor": rescue["key"],
                            "lease_after": (lease_steps - 1) * decode_seconds, "forced": None,
                            "tokens": sum(unit["tokens"] for unit in accepted),
                            "token_budget": resource["token_budget"],
                            "slot_limit": resolve_batch_slots(batch_slots, 1, instance),
                            "switch_cost": plan["switch_cost"], "resume_cost": 0.0,
                            "evaluation_cost": 0.0,
                            "free_blocks": plan["plan"]["remaining_available_blocks"]
                                           + resource["watermark_blocks"],
                            "first_deferred": plan["plan"]["deferred"],
                        }
                        score = {
                            "key": rescue["key"], "h": lease,
                            "good": len(attainable),
                            "tardiness": math.fsum(max(0.0, risk_work[job["key"]] - slack[job["key"]])
                                                   for job in jobs),
                            "mean": math.fsum(risk_work.values()) / len(jobs),
                            "denominator": len(jobs), "switch_cost": plan["switch_cost"],
                            "resume_cost": 0.0, "recompute_seconds": 0.0,
                            "finishes": {rescue["key"]: view.now + duration},
                            "batch_trace": [trace], "baseline_protection_violation": [],
                            "score_differs_from_baseline": True,
                            "finish_differs_from_baseline": True,
                        }
                        audit.update(
                            accepted=True, fallback=None,
                            candidate_discriminability="distinguished",
                            completed_scan=True, evaluation_version="deadline-rescue-v5",
                            heuristic="deadline_rescue", selection_rule=selection_rule,
                            evicted_hopeless=[key for unit in evicted for key in unit],
                            evicted_attainable=safe_evictions,
                            risk_q=risk_q, risk_work=risk_work, lease_steps=lease_steps,
                            rescue_ratio=slack[rescue["key"]] / risk_work[rescue["key"]],
                            actions=[score], seconds=time.perf_counter() - start)
                        return (rescue["key"], lease), audit
        if deadline_rescue_only:
            audit.update(fallback="conservative_evaluation_cap",
                         candidate_discriminability="no_rescue",
                         selection_scope="deadline_rescue_only",
                         evaluation_version="deadline-rescue-v1",
                         heuristic="deadline_rescue", seconds=time.perf_counter() - start)
            return (baseline_key, baseline_horizon), audit
        baseline_action = (baseline_key, baseline_horizon)
        actions = [baseline_action]
        incumbent = incumbent or getattr(view, "incumbent_action", None)
        if incumbent is not None:
            incumbent = tuple(incumbent)
            if (incumbent[0] in by_key and incumbent != baseline_action
                    and math.isfinite(incumbent[1]) and incumbent[1] > 0):
                actions.append(incumbent)
        frame_lease_cap = decode_seconds
        audit["frame_lease_cap"] = frame_lease_cap
        for call in sorted(calls, key=lambda item: item.key):
            for horizon in (baseline_horizon, max(
                    decode_seconds, min(frame_lease_cap, by_key[call.key]["current"]))):
                if (call.key, horizon) not in actions:
                    actions.append((call.key, horizon))
        audit["planned_actions"] = len(actions)
        audit["incumbent"] = incumbent
        unresolved = 0
        scan_deadline = start + evaluation_seconds
        for action_index, (key, horizon) in enumerate(actions):
            if resource is not None and key != baseline_key:
                feasible = first_plan_for_action(
                    jobs, (key, horizon), view.now, perf, instance,
                    resolve_batch_slots(batch_slots, 1, instance), resource)
                admitted = {unit["ids"][0] for unit in feasible["plan"]["accepted"]}
                if key not in admitted:
                    audit["resource_skips"].append({
                        "key": key, "h": horizon, "reason": "resource_infeasible",
                        "deferred": feasible["plan"]["deferred"],
                    })
                    continue
            signature = _signature(jobs, (key, horizon), view.now, instance,
                                   batch_slots, resource, perf, project_fn, eligible)
            cached = _cache_get(cache, signature)
            if cached == _TIMEOUT:
                audit["timeout_cache_hits"] += 1
                if action_index == 0:
                    audit.update(fallback="baseline_timeout_cache",
                                 candidate_discriminability="evaluation_failed",
                                 seconds=time.perf_counter() - start)
                    return (baseline_key, baseline_horizon), audit
                unresolved += 1
                continue
            if cached is not None:
                audit["cache_hits"] += 1
                raw = copy.deepcopy(cached)
            else:
                if action_index and time.perf_counter() >= scan_deadline:
                    break
                audit["project_calls"] += 1
                try:
                    kwargs = {"batch_slots": batch_slots}
                    if resource is not None:
                        kwargs["resource"] = resource
                    raw = project_fn(jobs, (key, horizon), view.now, perf, instance,
                                     scan_deadline, **kwargs)
                except TimeoutError:
                    audit["timeouts"] += 1
                    _cache_put(cache, signature, _TIMEOUT)
                    if action_index == 0:
                        audit.update(fallback="baseline_evaluation_timeout",
                                     candidate_discriminability="evaluation_failed",
                                     seconds=time.perf_counter() - start)
                        return (baseline_key, baseline_horizon), audit
                    unresolved += 1
                    if time.perf_counter() >= scan_deadline:
                        break
                    continue
                _cache_put(cache, signature, copy.deepcopy(raw))
            audit["actions"].append({"key": key, "h": horizon, **raw})
        unresolved = len(actions) - len(audit["actions"]) - len(audit["resource_skips"])
        audit["completed_scan"] = unresolved == 0
        audit["unevaluated_actions"] = unresolved
        audit["selection_scope"] = ("complete_action_set" if not unresolved
                                    else "evaluated_actions_partial")
        if not audit["actions"]:
            audit.update(fallback="execution-aligned reference budget",
                         candidate_discriminability="evaluation_failed",
                         seconds=time.perf_counter() - start)
            return (baseline_key, baseline_horizon), audit
        evaluation_cost = ((time.perf_counter() - start) if synchronous_evaluation else 0.0)
        evaluation_cost += getattr(view, "decision_overhead_seconds", 0.0)
        audit["actions"] = [
            {"key": action["key"], "h": action["h"],
             **_with_costs({key: value for key, value in action.items() if key not in ("key", "h")},
                           jobs, view.now, evaluation_cost)}
            for action in audit["actions"]
        ]
        baseline = audit["actions"][0]
        attainable = {job["key"] for job in jobs
                      if job["work"] <= by_key[job["key"]]["deadline"] - view.now + 1e-12}
        baseline_on_time = attainable | {key for key, finish in baseline["finishes"].items()
                                         if finish <= by_key[key]["deadline"] + 1e-12}
        audit["attainable"] = sorted(attainable, key=str)
        audit["baseline_on_time"] = sorted(baseline_on_time, key=str)
        for candidate in audit["actions"][1:]:
            candidate["score_differs_from_baseline"] = _score_differs(baseline, candidate)
            candidate["finish_differs_from_baseline"] = _finish_differs(baseline, candidate)
            candidate["baseline_protection_violation"] = sorted(
                (key for key in baseline_on_time
                 if candidate["finishes"][key] > by_key[key]["deadline"] + 1e-12), key=str)
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
            candidate = audit["actions"][index]
            if candidate["baseline_protection_violation"]:
                continue
            if compare_scores(candidate, audit["actions"][winner]) > 0:
                winner = index
        if winner != 0 and not (
                compare_scores(audit["actions"][winner], baseline) > 0
                and audit["actions"][winner]["finish_differs_from_baseline"]):
            winner = 0
        if winner != 0:
            chosen = audit["actions"][winner]
            incremental = chosen.get("switch_cost", 0.0) - baseline.get("switch_cost", 0.0)
            resume_increment = chosen.get("resume_cost", 0.0) - baseline.get("resume_cost", 0.0)
            mean_gain = baseline["mean"] - chosen["mean"]
            audit["incremental_switch_cost"] = incremental
            audit["incremental_resume_cost"] = resume_increment
            audit["net_mean_gain"] = mean_gain
            if chosen["good"] > baseline["good"]:
                objective_dimension = "good"
                objective_gain = chosen["good"] - baseline["good"]
            elif not math.isclose(chosen["tardiness"], baseline["tardiness"], abs_tol=1e-9):
                objective_dimension = "tardiness"
                objective_gain = baseline["tardiness"] - chosen["tardiness"]
            else:
                objective_dimension = "mean"
                objective_gain = mean_gain
            audit["net_objective_gain"] = objective_gain
            audit["net_objective_dimension"] = objective_dimension
        else:
            audit["fallback"] = "no_model_gain"
            protected_rejections = sum(bool(candidate["baseline_protection_violation"])
                                       for candidate in audit["actions"][1:])
            audit["protected_rejections"] = protected_rejections
            audit["fallback_diagnosis"] = (
                "baseline_on_time_protection" if protected_rejections and discriminating
                else "baseline_best_among_distinguished_actions" if discriminating
                else "evaluation_has_no_distinguishing_action")
        audit["accepted"] = winner != 0
        audit["seconds"] = time.perf_counter() - start
        chosen = audit["actions"][winner]
        return (chosen["key"], chosen["h"]), audit
    except (ValueError, RuntimeError) as error:
        audit.update(fallback=str(error), seconds=time.perf_counter() - start,
                     candidate_discriminability="evaluation_failed")
        return (baseline_key, baseline_horizon), audit
