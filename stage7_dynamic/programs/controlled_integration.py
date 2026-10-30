import json
import os
from pathlib import Path
import sys
import time
import traceback

from candidate_evaluator import candidate_action
from conditional_predictor import ConditionalPredictor
from controlled_scheduler import add, make_engine, public_infos, read_batches
from pool_policies import Performance


ASSET_ROOT = Path("/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_finite_h4_confirm_gpu01_20261018T114337Z_resume_v1")


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only integration gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        perf = Performance(json.loads((ASSET_ROOT / "PERFORMANCE.json").read_text()))
        pred = ConditionalPredictor.from_dict(json.loads(
            (root / "stage6_mechanism_repair/assets/CONDITIONAL_PREDICTOR.json").read_text()))
        stage_counts = {0: 2, 1: 1, 2: 1, 3: 1, 4: 1}
        output = root / "stage6_mechanism_repair/controlled" / attempt / "integration"
        infos = public_infos(root)
        engine, scheduler = make_engine(root, output, stage_counts, "4:stage:0", 50,
                                        perf=perf, pred=pred)
        evaluation_seconds = float(os.environ.get("V2_EVAL_SECONDS", "0.75"))
        scheduler.policy.candidate_selector = (
            lambda *args, **kwargs: candidate_action(
                *args, evaluation_seconds=evaluation_seconds, **kwargs))
        try:
            add(engine, infos, "warmup-0", 0, 0, 256)
            steps = 0
            while engine.engine.has_unfinished_requests() and steps < 100:
                engine.engine.step()
                steps += 1
            for _ in range(2):
                engine.engine.step()
            main_start = len(scheduler.policy.audit_events)
            for index in range(5):
                add(engine, infos, f"main-{index}", index, 0, 384 if index == 4 else 256,
                    slo=(60.0 - 10.0 * index) if index < 4 else 141.5)
            away_witness = None
            while engine.engine.has_unfinished_requests() and steps < 200 and away_witness is None:
                engine.engine.step()
                steps += 1
                active = {group.collection_id for group in
                          list(scheduler.running) + list(scheduler.waiting) + list(scheduler.swapped)}
                away = scheduler.policy.state.away_goals(time.perf_counter(), active)
                if away:
                    away_witness = away[0]
            if away_witness is not None:
                scheduler.policy.remaining = 0.0
                engine.engine.step()
                steps += 1
            add(engine, infos, "main-0-stage-1", 0, 1, 256)
            while engine.engine.has_unfinished_requests() and steps < 300:
                engine.engine.step()
                steps += 1
            for _ in range(2):
                engine.engine.step()
            events = scheduler.policy.audit_events
            main_events = events[main_start:]
            candidates = [event for event in main_events if event["kind"] == "v2_candidate"]
            accepted = [event for event in candidates if event["audit"].get("accepted")]
            acceptances = [event for event in main_events if event["kind"] == "v2_action_accepted"]
            schedules = [event for event in main_events if event["kind"] == "v2_action_scheduled"]
            executions = [event for event in main_events if event["kind"] == "v2_action_executed"]
            accepted_ids = {event["id"] for event in acceptances}
            scheduled_ids = {event["id"] for event in schedules}
            executed = [{"id": event["id"], "anchor": event["anchor"],
                         "selected": event["selected"]}
                        for event in executions if event["id"] in accepted_ids]
            away_views = [event for event in candidates
                          if any(goal["state"] == "AWAY" for goal in event["view"]["goals"])]
            resets = [event for event in events if event["kind"] == "v2_chain_reset"]
            batches = read_batches(output)
            details.update({
                "steps": steps,
                "unfinished": engine.engine.has_unfinished_requests(),
                "accepted": len(accepted),
                "executed": executed,
                "away_views": len(away_views),
                "away_witness": {
                    "state": away_witness.state,
                    "tool_entered": away_witness.tool_entered,
                    "tool_elapsed": away_witness.tool_elapsed,
                } if away_witness else None,
                "resets": resets,
                "batches": batches,
                "support_cache": scheduler.policy.support.snapshot(),
                "chain_state": scheduler.policy.state.snapshot(),
                "candidate_summaries": [{
                    "n": event["n"], "fallback": event["audit"].get("fallback"),
                    "now": event["view"]["now"],
                    "goals": [{"program": goal["program"], "deadline": goal["deadline"]}
                              for goal in event["view"]["goals"]],
                    "discriminability": event["audit"].get("candidate_discriminability"),
                    "timeouts": event["audit"].get("timeouts"),
                    "seconds": event["audit"].get("seconds"),
                    "scores": [{"key": item["key"], "h": item["h"], "good": item["good"], "mean": item["mean"],
                                "finishes": item["finishes"],
                                "differs": item.get("score_differs_from_baseline")}
                               for item in event["audit"].get("actions", [])],
                } for event in candidates],
            })
            check("warmup_main_reset", len(resets) == 1 and resets[0]["previous_history"] == 1,
                  resets, "one reset with one previous completed call")
            fallback_explained = all(event["audit"].get("fallback") in ("no_model_gain", "degenerate")
                                     for event in candidates)
            check("accepted_action_execution_closed",
                  len(acceptances) == len(accepted) and accepted_ids == scheduled_ids
                  and accepted_ids == {event["id"] for event in executed}
                  and (accepted or (candidates and fallback_explained)),
                  {"accepted": len(accepted), "scheduled": sorted(scheduled_ids),
                   "executed": executed, "fallback_explained": fallback_explained},
                  "every accepted action has one scheduled/executed witness; zero acceptance requires explicit fallback evidence")
            check("away_goal_in_real_candidate_view", away_witness is not None and away_views,
                  {"witness": details["away_witness"], "away_views": len(away_views)},
                  "current AWAY appears in a real candidate view")
            check("accepted_support_remains_valid",
                  all(entry["unsupported_from"] is None
                      for entry in details["support_cache"].values()),
                  details["support_cache"], "no accepted action or timeout poisons support")
            check("all_calls_complete_and_release",
                  not details["unfinished"] and len(batches) > 0,
                  {"unfinished": details["unfinished"], "batches": len(batches), "steps": steps},
                  "warmup, all main chains and the AWAY follow-up stage complete")
        finally:
            engine.shutdown_background_loop()
            engine.engine.model_executor.finalize()
        summary = json.loads((output / "executor_summary.json").read_text())
        details["executor_summary"] = summary
        check("kv_release", summary["tracked_block_owners"] == 0,
              summary["tracked_block_owners"], "no tracked KV owners")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "integration completed")

    passed = all(item["pass"] for item in checks.values())
    result = {"schema": "stage6-integration-gate/v1", "attempt": attempt,
              "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
              "gpu_used": False, "small_gpu_qualification_allowed": passed}
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_INTEGRATION_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
