import json
import os
from pathlib import Path
import sys
import traceback

from candidate_evaluator import candidate_action
from contracts import CacheView, CallView
from controlled_scheduler import (ToyPerformance, add, make_engine, predictor,
                                  public_infos, read_batches)
from execution_view import ExecutionView
from lease_reference import project
from v2_contracts import GoalView


def direct_view(count=2, deadlines=None, slots=None):
    deadlines = deadlines or [20.0 + 10 * index for index in range(count)]
    slots = slots or [1] * count
    goals = []
    members = {}
    for index in range(count):
        key = f"{index}:stage:0"
        call = CallView(key, index, 0, 256 * slots[index], 0, 0, 0, 0.0, 0.0,
                        0.0, 0.0, False, 0.0)
        goals.append(GoalView(index, 0.0, deadlines[index], "READY", (), call))
        members[key] = [f"{key}:member:{member}" for member in range(slots[index])]
    view = ExecutionView(0.0, tuple(goals), CacheView(8192, 8192, ()),
                         tuple((index, 0.0) for index in range(count)))
    return view, members


def run_parity(root, output):
    infos = public_infos(root)
    stage_counts = {index: 1 for index in range(5)}
    engine, scheduler = make_engine(root, output, stage_counts, "4:stage:0", 50)
    scheduler.policy.candidate_selector = candidate_action
    try:
        for index in range(5):
            add(engine, infos, f"single-{index}", index, 0, 384 if index == 4 else 256,
                slo=(60.0 - 10.0 * index) if index < 4 else 3.0)
        for _ in range(4):
            engine.engine.step()
        events = scheduler.policy.audit_events
        candidate = next(event for event in events if event["kind"] == "v2_candidate")
        action = next(item for item in candidate["audit"]["actions"]
                      if item["key"] == candidate["anchor"] and item["h"] == candidate["remaining"])
        members = candidate["compound_members"]
        key_for_request = {request_id: key for key, request_ids in members.items()
                           for request_id in request_ids}
        actual = [[key_for_request[request_id] for request_id in batch]
                  for batch in read_batches(output)]
        after = [event for event in events if event["kind"] == "v2_after_step"]
        result = {
            "accepted": candidate["audit"].get("accepted"),
            "anchor": candidate["anchor"],
            "horizon": candidate["remaining"],
            "discriminability": candidate["audit"].get("candidate_discriminability"),
            "timeouts": candidate["audit"].get("timeouts"),
            "action_summaries": [{"key": item["key"], "h": item["h"],
                                  "good": item["good"], "mean": item["mean"],
                                  "differs": item.get("score_differs_from_baseline")}
                                 for item in candidate["audit"]["actions"]],
            "expected_keys": [item["keys"] for item in action["batch_trace"]],
            "actual_keys": actual,
            "expected_lease": [item["lease_after"] for item in action["batch_trace"]],
            "actual_lease": [event["remaining"] for event in after],
            "expected_slots": [item["slots"] for item in action["batch_trace"]],
            "actual_slots": [len(batch) for batch in read_batches(output)],
            "support_cache": scheduler.policy.support.snapshot(),
        }
    finally:
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()
    return result


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only gate")
    checks = {}
    details = {}
    perf = ToyPerformance()
    pred = predictor()

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        view, members = direct_view(1)
        action, audit = candidate_action(view, 0, set(members), pred, perf, {0: 0},
                                         compound_members=members)
        check("single_goal_is_explicitly_degenerate",
              not audit["accepted"] and audit["candidate_discriminability"] == "degenerate"
              and not audit["actions"],
              {"action": action, "audit": audit},
              "single goal is not sent through a guaranteed-tie rollout")

        view, members = direct_view(2, deadlines=[20.0, 20.0])
        action, audit = candidate_action(view, 0, set(members), pred, perf,
                                         {0: 0, 1: 0}, compound_members=members)
        check("symmetric_underfilled_state_is_degenerate",
              not audit["accepted"] and audit["candidate_discriminability"] == "degenerate",
              audit, "identical jobs inside capacity are not mislabeled as model evidence")

        view, members = direct_view(5)
        calls = {"count": 0}

        def flaky(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise TimeoutError("injected candidate timeout")
            return project(*args, **kwargs)

        action, audit = candidate_action(view, 0, set(members), pred, perf,
                                         {index: 0 for index in range(5)}, compound_members=members,
                                         project_fn=flaky)
        check("candidate_timeout_preserves_completed_scores",
              audit["timeouts"] == 1 and audit["actions"] and audit["fallback"] == "no_model_gain",
              {"action": action, "timeouts": audit["timeouts"],
               "completed_actions": len(audit["actions"]), "fallback": audit["fallback"]},
              "one later timeout preserves the completed baseline and does not become unsupported")

        jobs = []
        for index, (slots, deadline) in enumerate(((2, 20.0), (2, 30.0), (1, 40.0))):
            jobs.append({"key": str(index), "state": "READY", "deadline": deadline,
                         "instance": 0, "context": 100, "current": 2.0,
                         "phases": (("compute", 2.0),), "slots": slots,
                         "level": 0, "ready_wait": 0.0, "virtual": 0.0})
        score = project(jobs, ("0", 0.04), 0.0, perf, 0)
        trace = score["batch_trace"]
        check("compound_trace_and_lease_units",
              len(trace) >= 4 and all(abs(item["lease_after"] - expected) < 1e-9
                                      for item, expected in zip(trace[:4], [0.03, 0.02, 0.01, 0.0]))
              and all(item["slots"] <= 5 for item in trace),
              {"trace": trace, "max_batch_slots": score["max_batch_slots"]},
              "two-member anchor consumes one decode-equivalent lease per engine step")

        away_jobs = [
            {"key": "a", "state": "READY", "deadline": 10.0, "instance": 0,
             "context": 100, "current": 0.02,
             "phases": (("compute", 0.02), ("away", 0.03), ("compute", 0.02)),
             "slots": 1, "level": 0, "ready_wait": 0.0, "virtual": 0.0},
            {"key": "b", "state": "READY", "deadline": 10.0, "instance": 0,
             "context": 100, "current": 0.01, "phases": (("compute", 0.01),),
             "slots": 1, "level": 0, "ready_wait": 0.0, "virtual": 0.0},
        ]
        away = project(away_jobs, ("a", 0.04), 0.0, perf, 0)
        check("away_timer_denominator_and_clock",
              away["denominator"] == 2 and away["timer_intervals"] == 1
              and len(away["finishes"]) == 2 and away["idle"] > 0,
              {key: away[key] for key in ("denominator", "timer_intervals", "busy", "idle", "elapsed")},
              "AWAY timer, complete denominator and clock conservation hold")

        parity = run_parity(root, root / "stage6_mechanism_repair/controlled" / attempt / "parity")
        details["parity"] = parity
        lease_count = min(len(parity["expected_lease"]), len(parity["actual_lease"]))
        check("accepted_multi_goal_action_present",
              parity["accepted"] and parity["discriminability"] == "distinguished",
              {"accepted": parity["accepted"], "discriminability": parity["discriminability"]},
              "controlled multi-goal state accepts a genuinely distinguished action")
        check("first_action_and_lease_match",
              parity["expected_keys"] and parity["actual_keys"]
              and set(parity["expected_keys"][0]) == set(parity["actual_keys"][0])
              and parity["expected_slots"][0] == parity["actual_slots"][0]
              and lease_count > 0 and all(abs(a - b) < 1e-9 for a, b in
                      zip(parity["expected_lease"][:lease_count], parity["actual_lease"][:lease_count])),
              {"expected_first": parity["expected_keys"][0],
               "actual_first": parity["actual_keys"][0],
               "expected_lease": parity["expected_lease"][:lease_count],
               "actual_lease": parity["actual_lease"][:lease_count],
               "later_expected": parity["expected_keys"][1:lease_count],
               "later_actual": parity["actual_keys"][1:lease_count]},
              "first real-KV plan and lease consumption match; later rollout remains an explicit approximation")
        check("accepted_action_preserves_support",
              all(entry["unsupported_from"] is None for entry in parity["support_cache"].values()),
              parity["support_cache"], "accepted action does not poison support")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "all controlled checks completed")

    passed = all(item["pass"] for item in checks.values())
    result = {"schema": "step3-evaluation-gate/v1", "attempt": attempt,
              "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
              "gpu_used": False, "next_step_allowed": passed}
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_EVALUATION_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
