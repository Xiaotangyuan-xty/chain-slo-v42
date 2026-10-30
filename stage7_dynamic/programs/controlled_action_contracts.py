from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import traceback

from candidate_evaluator import candidate_action
from controlled_evaluation import direct_view
from controlled_scheduler import (add, make_engine, public_infos, run_forced,
                                  selector)
from controlled_net_value import CostPerformance
from controlled_scheduler import predictor


def synthetic(other_mean, other_tardiness=0.0):
    def score(jobs, action, *args, **kwargs):
        key, _ = action
        baseline = key == "0:stage:0"
        return {
            "good": 1, "tardiness": 1.0 if baseline else other_tardiness,
            "mean": 100.0 if baseline else other_mean,
            "switch_cost": 0.0,
            "finishes": {job["key"]: (2.0 if baseline else 1.0) for job in jobs},
            "batch_trace": [{"keys": [key], "slots": 1, "lease_after": 0.03}],
        }
    return score


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only action contract gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        view, members = direct_view(5)
        resources = {
            goal.ready.key: {"slots": 1, "tokens": 1, "required_blocks": 0,
                             "resident_blocks": 0, "status": "RUNNING", "gpu_tokens": 0}
            for goal in view.goals
        }
        view = replace(view, resource_units=tuple(resources.items()),
                       free_kv_blocks=100, watermark_blocks=0,
                       native_priorities=tuple((key, (0, index))
                                               for index, key in enumerate(sorted(resources))))
        owners = {index: 0 for index in range(5)}
        _, negative = candidate_action(view, 0, set(members), predictor(), CostPerformance(),
                                       owners, compound_members=members,
                                       project_fn=synthetic(110.0))
        details["negative_mean"] = negative
        check("negative_mean_is_rejected_even_with_tardiness_gain",
              not negative["accepted"] and negative["fallback_diagnosis"] == "negative_mean_gain",
              {"accepted": negative["accepted"], "diagnosis": negative["fallback_diagnosis"],
               "rejected_action": negative.get("rejected_action")},
              "good does not increase and mean worsens, so the action is rejected")
        action, neutral = candidate_action(view, 0, set(members), predictor(), CostPerformance(),
                                           owners, compound_members=members,
                                           project_fn=synthetic(100.0))
        check("tardiness_gain_with_neutral_mean_remains_allowed",
              action == ("1:stage:0", 0.04) and neutral["accepted"],
              {"action": action, "accepted": neutral["accepted"]},
              "tardiness improvement with no mean regression and no switch cost is allowed")

        output = root / "stage6_mechanism_repair/controlled" / attempt / "execution"
        infos = public_infos(root)
        engine, scheduler = make_engine(root, output, {0: 1, 4: 1}, "0:stage:0", 50)
        try:
            add(engine, infos, "z-request", 0, 0, 64, slo=60.0)
            add(engine, infos, "a-request", 4, 0, 64, slo=60.0)
            group = next(group for group in scheduler.waiting if group.request_id == "z-request")
            scheduler.policy.before_schedule(list(scheduler.waiting), time.perf_counter(),
                                             scheduler.v2_ledger)
            scheduler.policy.model_action = True
            scheduler.policy.pending_action = {
                "id": 1, "anchor": "0:stage:0", "expected_first": ["4:stage:0"],
                "scheduled": False, "executed": False,
            }
            scheduler.policy.note_selection([group], time.perf_counter(), {"deferred": []})
            mismatch = scheduler.policy.pending_action
            scheduler.policy.after_model_step(0.01, time.perf_counter())
            mismatch_events = [event for event in scheduler.policy.audit_events
                               if event["kind"] in ("v2_action_scheduled", "v2_action_executed")]
            check("mismatched_anchor_is_not_counted_as_executed",
                  mismatch["matches_projection"] is False and not mismatch["executed"]
                  and not any(event["kind"] == "v2_action_executed" for event in mismatch_events),
                  {"pending_action": mismatch, "events": mismatch_events},
                  "anchor presence with the wrong full unit set cannot become an execution witness")

            scheduler.policy.pending_action = {
                "id": 2, "anchor": "0:stage:0", "expected_first": ["0:stage:0"],
                "scheduled": False, "executed": False,
            }
            scheduler.policy.note_selection([group], time.perf_counter(), {"deferred": []})
            scheduler.policy.after_model_step(0.01, time.perf_counter())
            matched = scheduler.policy.pending_action
            matched_events = [event for event in scheduler.policy.audit_events
                              if event["kind"] in ("v2_action_scheduled", "v2_action_executed")]
            check("matched_action_has_complete_execution_witness",
                  matched["matches_projection"] is True and matched["executed"]
                  and any(event["kind"] == "v2_action_executed" and event["id"] == 2
                          for event in matched_events),
                  {"pending_action": matched, "events": matched_events},
                  "a complete matching first unit set produces an execution witness")
        finally:
            engine.shutdown_background_loop()
            engine.engine.model_executor.finalize()

        infos = public_infos(root)
        forced_output = root / "stage6_mechanism_repair/controlled" / attempt / "forced"
        holder = {}
        original_make = make_engine

        def counting_engine(root_value, output_value, *args, **kwargs):
            engine, scheduler = original_make(root_value, output_value, *args, **kwargs)
            holder["scheduler"] = scheduler
            original_selector = scheduler.policy.candidate_selector
            holder["forced_calls"] = 0

            def counted(*args, **kwargs):
                if scheduler.policy.forced is not None:
                    holder["forced_calls"] += 1
                return original_selector(*args, **kwargs)

            scheduler.policy.candidate_selector = counted
            return engine, scheduler

        import controlled_scheduler
        original = controlled_scheduler.make_engine
        controlled_scheduler.make_engine = counting_engine
        try:
            forced = run_forced(root, forced_output)
        finally:
            controlled_scheduler.make_engine = original
        details["forced"] = forced
        check("forced_skips_candidate_evaluation_and_executes",
              holder["forced_calls"] == 0 and forced["forced"] == "3:stage:0"
              and "single-3" in forced["forced_batch"] and not forced["unfinished"],
              {"forced_calls": holder["forced_calls"], "forced": forced["forced"],
               "forced_batch": forced["forced_batch"], "unfinished": forced["unfinished"]},
              "forced takes over without any candidate call and completes the waiting request")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "action contract gate completed")

    passed = all(item["pass"] for item in checks.values())
    result = {"schema": "stage11-action-contracts-gate/v1", "attempt": attempt,
              "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
              "gpu_used": False}
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_ACTION_CONTRACTS_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
