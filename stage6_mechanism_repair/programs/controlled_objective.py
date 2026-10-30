import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

from candidate_evaluator import candidate_action
from controlled_evaluation import direct_view
from controlled_scheduler import ToyPerformance, predictor
from lease_reference import compare_scores, project


def jobs(deadlines):
    return [{
        "key": str(index), "state": "READY", "deadline": deadline,
        "instance": 0, "context": 100, "current": 0.02,
        "phases": (("compute", 0.02),), "slots": 1, "level": 0,
        "ready_wait": 0.0, "virtual": 0.0, "running": False,
    } for index, deadline in enumerate(deadlines)]


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only objective gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        perf = ToyPerformance()
        primary = [project(jobs([0.025, 0.1]), (str(index), 0.04), 0.0, perf, 0,
                          batch_slots=1) for index in range(2)]
        details["good_priority"] = [{key: score[key] for key in
                                     ("good", "tardiness", "mean", "finishes")}
                                    for score in primary]
        check("good_is_strictly_primary",
              primary[0]["good"] == 2 and primary[1]["good"] == 1
              and compare_scores(primary[0], primary[1]) > 0,
              details["good_priority"], "the action attaining both deadlines beats the one attaining one")

        gradient = [project(jobs([0.015, 0.1]), (str(index), 0.04), 0.0, perf, 0,
                           batch_slots=1) for index in range(2)]
        details["tardiness_gradient"] = [{key: score[key] for key in
                                          ("good", "tardiness", "mean", "finishes")}
                                         for score in gradient]
        check("same_good_and_mean_have_tardiness_gradient",
              gradient[0]["good"] == gradient[1]["good"] == 1
              and math.isclose(gradient[0]["mean"], gradient[1]["mean"], abs_tol=1e-12)
              and math.isclose(gradient[0]["tardiness"], 0.005, abs_tol=1e-12)
              and math.isclose(gradient[1]["tardiness"], 0.025, abs_tol=1e-12)
              and compare_scores(gradient[0], gradient[1]) > 0,
              details["tardiness_gradient"],
              "equal good and mean prefer the order with 0.005 rather than 0.025 seconds of deadline excess")
        check("good_cannot_be_traded_for_mean",
              compare_scores({"good": 1, "tardiness": 100.0, "mean": 100.0},
                             {"good": 0, "tardiness": 0.0, "mean": -100.0}) > 0,
              "one attained deadline versus zero", "good dominates every soft term")
        check("comparison_tolerance_is_stable",
              compare_scores({"good": 1, "tardiness": 1.0, "mean": 2.0},
                             {"good": 1, "tardiness": 1.0 + 1e-12, "mean": 2.0 - 1e-12}) == 0,
              "1e-12 differences", "differences below 1e-9 seconds are ties")

        view, members = direct_view(5)
        calls = []

        def synthetic(jobs_value, action, *args, **kwargs):
            calls.append(action)
            key, horizon = action
            if key == "0:stage:0":
                return {"good": 1, "tardiness": 1.0, "mean": 2.0,
                        "finishes": {job["key"]: 1.0 if job["key"] != "0:stage:0" else 3.0
                                     for job in jobs_value},
                        "batch_trace": [{"keys": ["0:stage:0"], "slots": 1, "lease_after": 0.03}]}
            return {"good": 1, "tardiness": 0.0, "mean": 2.0,
                    "finishes": {job["key"]: 2.0 for job in jobs_value},
                    "batch_trace": [{"keys": ["1:stage:0"], "slots": 1, "lease_after": 0.03}]}

        action, audit = candidate_action(view, 0, set(members), predictor(), perf,
                                         {index: 0 for index in range(5)}, compound_members=members,
                                         project_fn=synthetic)
        details["selector"] = {"action": action, "audit": audit, "calls": calls}
        check("selector_uses_tardiness_gradient",
              action == ("1:stage:0", 0.04) and audit["accepted"]
              and audit["candidate_discriminability"] == "distinguished",
              {"action": action, "accepted": audit["accepted"],
               "discriminability": audit["candidate_discriminability"]},
              "a non-default action wins when only total deadline excess improves")

        def tied(jobs_value, *args, **kwargs):
            return {"good": 1, "tardiness": 0.0, "mean": 2.0,
                    "finishes": {job["key"]: 2.0 for job in jobs_value},
                    "batch_trace": [{"keys": ["0:stage:0"], "slots": 1, "lease_after": 0.03}]}

        tied_action, tied_audit = candidate_action(
            view, 0, set(members), predictor(), perf, {index: 0 for index in range(5)},
            compound_members=members, project_fn=tied)
        check("equivalent_objective_and_finish_do_not_accept",
              not tied_audit["accepted"] and tied_action == ("0:stage:0", 0.04)
              and tied_audit["candidate_discriminability"] == "invariant",
              {"action": tied_action, "accepted": tied_audit["accepted"],
               "discriminability": tied_audit["candidate_discriminability"]},
              "a numerical or cosmetic action cannot be accepted without objective and finish differences")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "objective gate completed")

    passed = all(item["pass"] for item in checks.values())
    source = Path(__file__).resolve()
    result = {
        "schema": "stage7-objective-gate/v1", "attempt": attempt,
        "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
        "gpu_used": False, "next_step_allowed": passed,
        "sources": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_OBJECTIVE_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
