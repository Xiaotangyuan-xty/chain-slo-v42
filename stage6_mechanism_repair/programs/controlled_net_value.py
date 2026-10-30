from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import traceback

from candidate_evaluator import candidate_action
from controlled_evaluation import direct_view
from controlled_scheduler import ToyPerformance, predictor
from lease_reference import compare_scores, first_plan_for_action, project


class CostPerformance(ToyPerformance):
    copy_seconds_per_token = 0.01


def direct_resource_view():
    view, members = direct_view(5)
    resources = {
        "0:stage:0": {"slots": 1, "tokens": 2, "required_blocks": 184,
                      "resident_blocks": 0, "status": "SWAPPED", "host_tokens": 2933},
        "1:stage:0": {"slots": 1, "tokens": 2, "required_blocks": 170,
                      "resident_blocks": 0, "status": "SWAPPED", "host_tokens": 2709},
        "2:stage:0": {"slots": 1, "tokens": 1, "required_blocks": 1,
                      "resident_blocks": 157, "status": "RUNNING", "gpu_tokens": 2500},
        "3:stage:0": {"slots": 1, "tokens": 0, "required_blocks": 0,
                      "resident_blocks": 16, "status": "RUNNING", "gpu_tokens": 256},
        "4:stage:0": {"slots": 1, "tokens": 0, "required_blocks": 0,
                      "resident_blocks": 16, "status": "RUNNING", "gpu_tokens": 256},
    }
    goals = []
    for goal in view.goals:
        if goal.program >= 2:
            goals.append(replace(goal, ready=replace(goal.ready, running=True)))
        else:
            goals.append(goal)
    priorities = tuple((key, (0, index)) for index, key in enumerate(sorted(resources)))
    return replace(view, goals=tuple(goals), resource_units=tuple(resources.items()),
                   free_kv_blocks=339, watermark_blocks=5, max_swaps_per_iter=3,
                   token_budget=2048, native_priorities=priorities), members


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only net-value gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        perf = CostPerformance()
        view, members = direct_resource_view()
        jobs = [{
            "key": goal.ready.key, "state": "READY", "instance": 0,
            "deadline": goal.deadline, "slots": 1, "level": 0, "ready_wait": 0.0,
            "virtual": 0.0, "context": goal.ready.prompt_tokens,
            "running": goal.ready.running,
            "native_priority": view.native_priority_for(goal.ready.key),
            "kv_required": view.resource_for(goal.ready.key)["required_blocks"],
            "resource_reason": view.resource_for(goal.ready.key).get("reason"),
            **view.resource_for(goal.ready.key),
        } for goal in view.goals]
        resource = view.resource_contract()
        feasible = first_plan_for_action(jobs, ("0:stage:0", 0.04), 0.0, perf, 0, 4, resource)
        accepted = [entry["key"] for entry in feasible["selected"]]
        limited_resource = dict(resource, free_blocks=100)
        infeasible = first_plan_for_action(jobs, ("0:stage:0", 0.04), 0.0, perf, 0, 4,
                                           limited_resource)
        infeasible_accepted = [entry["key"] for entry in infeasible["selected"]]
        details["first_plans"] = {"feasible": accepted, "infeasible": infeasible_accepted,
                                  "deferred": infeasible["plan"]["deferred"]}
        check("first_plan_uses_real_kv_and_retained_state",
              accepted == ["0:stage:0", "2:stage:0", "3:stage:0", "4:stage:0"]
              and "0:stage:0" not in infeasible_accepted
              and any(item[0] == ["0:stage:0"] and item[1] == "kv"
                      for item in infeasible["plan"]["deferred"]),
              details["first_plans"],
              "long-a is admitted, long-b and retained owners cannot be silently over-admitted")

        calls = []

        def counted(*args, **kwargs):
            calls.append(args[1][0])
            return project(*args, **kwargs)

        limited_view = replace(view, free_kv_blocks=100)
        action, audit = candidate_action(limited_view, 0, set(members), predictor(), perf,
                                         {index: 0 for index in range(5)},
                                         compound_members=members, project_fn=counted)
        details["resource_evaluation"] = audit
        check("infeasible_anchor_skips_full_projection",
              any(item["key"] == "1:stage:0" and item["reason"] == "resource_infeasible"
                  for item in audit["resource_skips"])
              and "1:stage:0" not in calls and audit["completed_scan"],
              {"resource_skips": audit["resource_skips"], "project_calls": calls},
              "the infeasible anchor is rejected by the shared planner before rollout")
        check("lower_mean_is_better",
              compare_scores({"good": 1, "tardiness": 0.0, "mean": 80.0},
                             {"good": 1, "tardiness": 0.0, "mean": 100.0}) > 0,
              "mean 80 versus mean 100", "the lower mean wins when good and tardiness tie")

        cost_jobs = [
            {"key": "a", "state": "READY", "deadline": 1.0, "instance": 0,
             "context": 100, "current": 0.02, "phases": (("compute", 0.02),),
             "slots": 1, "level": 0, "ready_wait": 0.0, "virtual": 0.0,
             "running": True, "kv_required": 1, "resident_blocks": 16,
             "gpu_tokens": 256, "host_tokens": 0},
            {"key": "b", "state": "READY", "deadline": 1.0, "instance": 0,
             "context": 100, "current": 0.02, "phases": (("compute", 0.02),),
             "slots": 1, "level": 0, "ready_wait": 0.0, "virtual": 0.0,
             "running": True, "kv_required": 0, "resident_blocks": 16,
             "gpu_tokens": 256, "host_tokens": 0},
        ]
        cost_resource = {"free_blocks": 6, "watermark_blocks": 0,
                         "block_size": 16, "max_swaps": 3, "token_budget": 2048}
        score = project(cost_jobs, ("b", 0.04), 0.0, perf, 0,
                        batch_slots=1, resource=cost_resource)
        details["switch_cost"] = score["switch_cost"]
        check("switch_cost_uses_actual_gpu_tokens",
              abs(score["switch_cost"] - 2.56) < 1e-9,
              score["switch_cost"], "256 GPU tokens at 0.01 seconds/token cost 2.56 seconds")

        free_view, free_members = direct_view(5)
        free_resources = {
            goal.ready.key: {"slots": 1, "tokens": 1, "required_blocks": 0,
                             "resident_blocks": 0, "status": "RUNNING", "gpu_tokens": 0}
            for goal in free_view.goals
        }
        free_view = replace(free_view, resource_units=tuple(free_resources.items()),
                            free_kv_blocks=100, watermark_blocks=0,
                            native_priorities=tuple((key, (0, index))
                                                    for index, key in enumerate(sorted(free_resources))))
        owners = {index: 0 for index in range(5)}

        def synthetic(mean, cost=10.0, timeout=False):
            calls = {"n": 0}

            def score(jobs_value, action, *args, **kwargs):
                calls["n"] += 1
                if timeout and calls["n"] == 2:
                    raise TimeoutError("injected partial scan")
                key, _ = action
                return {"good": 1, "tardiness": 0.0,
                        "mean": 100.0 if key == "0:stage:0" else mean,
                        "switch_cost": 0.0 if key == "0:stage:0" else cost,
                        "finishes": {job["key"]: (2.0 if key == "0:stage:0" else 1.0)
                                     for job in jobs_value},
                        "batch_trace": [{"keys": [key], "slots": 1, "lease_after": 0.03}]}
            return score

        _, low = candidate_action(free_view, 0, set(free_members), predictor(), perf,
                                  owners, compound_members=free_members,
                                  project_fn=synthetic(99.0))
        check("small_mean_gain_cannot_cover_switch_cost",
              not low["accepted"] and low["fallback_diagnosis"] == "insufficient_net_gain"
              and low["net_mean_gain"] < 0,
              {"accepted": low["accepted"], "diagnosis": low["fallback_diagnosis"],
               "net_mean_gain": low.get("net_mean_gain")},
              "one second of mean gain is rejected against ten seconds of switching cost")

        action, high = candidate_action(free_view, 0, set(free_members), predictor(), perf,
                                        owners, compound_members=free_members,
                                        project_fn=synthetic(80.0))
        check("positive_net_gain_can_be_accepted",
              action == ("1:stage:0", 0.04) and high["accepted"]
              and high["net_mean_gain"] > 0,
              {"action": action, "accepted": high["accepted"],
               "net_mean_gain": high["net_mean_gain"]},
              "twenty seconds of mean gain covers ten seconds of switching cost")

        _, partial = candidate_action(free_view, 0, set(free_members), predictor(), perf,
                                      owners, compound_members=free_members,
                                      project_fn=synthetic(0.0, cost=0.0, timeout=True))
        check("partial_scan_cannot_accept_nondefault",
              not partial["accepted"] and partial["fallback_diagnosis"] == "incomplete_evaluation",
              {"accepted": partial["accepted"], "diagnosis": partial["fallback_diagnosis"]},
              "a winning action from an incomplete scan is not accepted")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "net-value gate completed")

    passed = all(item["pass"] for item in checks.values())
    result = {"schema": "stage10-net-value-gate/v1", "attempt": attempt,
              "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
              "gpu_used": False}
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_NET_VALUE_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
