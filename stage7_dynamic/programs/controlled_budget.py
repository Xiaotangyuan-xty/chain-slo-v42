from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import traceback

from candidate_evaluator import candidate_action
from controlled_evaluation import direct_view
from controlled_net_value import CostPerformance, direct_resource_view
from controlled_scheduler import predictor


def tied(jobs, action, *args, **kwargs):
    return {"good": 0, "tardiness": 0.0, "mean": 1.0,
            "finishes": {job["key"]: 1.0 for job in jobs},
            "switch_cost": 0.0,
            "batch_trace": [{"keys": [action[0]], "slots": 1, "lease_after": 0.03}]}


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only evaluation budget gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        perf = CostPerformance()
        view, members = direct_resource_view()
        view = replace(view, free_kv_blocks=100)
        calls = []

        def counted(jobs, action, *args, **kwargs):
            calls.append(action[0])
            return tied(jobs, action, *args, **kwargs)

        _, audit = candidate_action(view, 0, set(members), predictor(), perf,
                                    {index: 0 for index in range(5)},
                                    compound_members=members, project_fn=counted)
        details["budget"] = audit
        check("resource_preflight_reduces_full_rollouts",
              audit["planned_actions"] > audit["project_calls"]
              and audit["resource_skips"] and audit["completed_scan"]
              and "2:stage:0" in calls,
              {"planned": audit["planned_actions"], "project_calls": audit["project_calls"],
               "resource_skips": audit["resource_skips"], "calls": calls},
              "infeasible actions are skipped while feasible candidates remain evaluated")

        small, small_members = direct_view(2)
        small_resources = {
            "0:stage:0": {"slots": 1, "tokens": 10, "required_blocks": 10,
                          "resident_blocks": 0, "status": "SWAPPED", "host_tokens": 1000},
            "1:stage:0": {"slots": 1, "tokens": 1, "required_blocks": 0,
                          "resident_blocks": 16, "status": "RUNNING", "gpu_tokens": 256},
        }
        small = replace(small, resource_units=tuple(small_resources.items()),
                        free_kv_blocks=5, watermark_blocks=0,
                        native_priorities=(("0:stage:0", (0, 0)), ("1:stage:0", (0, 1))))
        _, small_audit = candidate_action(small, 0, set(small_members), predictor(), perf,
                                          {0: 0, 1: 0}, compound_members=small_members,
                                          project_fn=tied)
        details["slots_fit_kv_not_fit"] = small_audit
        check("slots_fit_does_not_hide_kv_contention",
              small_audit["fallback"] != "degenerate"
              and small_audit["preflight_reason"] != "all_ready_units_fit_fixed_batch",
              {"fallback": small_audit["fallback"],
               "preflight_reason": small_audit["preflight_reason"],
               "discriminability": small_audit["candidate_discriminability"]},
              "two slots do not produce a false degenerate result when KV cannot fit")

        native, native_members = direct_view(2)
        native_resources = {
            key: {"slots": 1, "tokens": 1, "required_blocks": 0,
                  "resident_blocks": 0, "status": "RUNNING", "gpu_tokens": 0}
            for key in ("0:stage:0", "1:stage:0")
        }
        native = replace(native, resource_units=tuple(native_resources.items()),
                         free_kv_blocks=100, watermark_blocks=0,
                         native_priorities=(("0:stage:0", (0, 1)), ("1:stage:0", (0, 0))))
        _, native_audit = candidate_action(native, 0, set(native_members), predictor(), perf,
                                           {0: 0, 1: 0}, compound_members=native_members,
                                           project_fn=tied)
        check("baseline_uses_native_priority_not_v2_level",
              native_audit["pi0"] == "1:stage:0",
              native_audit["pi0"], "native priority selects 1:stage:0 despite V2 key ordering")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "evaluation budget gate completed")

    passed = all(item["pass"] for item in checks.values())
    result = {"schema": "stage10-evaluation-budget-gate/v1", "attempt": attempt,
              "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
              "gpu_used": False}
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_BUDGET_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
