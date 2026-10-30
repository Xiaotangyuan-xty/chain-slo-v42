import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from candidate_evaluator import candidate_action
from controlled_action import run_running_anchor
from controlled_evaluation import direct_view
from controlled_scheduler import (ToyPerformance, add, make_engine, predictor,
                                  public_infos, read_batches, selector)


class CountingPredictor:
    def __init__(self):
        self.delegate = predictor()
        self.calls = 0

    def estimate(self, *args, **kwargs):
        self.calls += 1
        return self.delegate.estimate(*args, **kwargs)


def stable_run(root, output):
    infos = public_infos(root)
    engine, scheduler = make_engine(root, output, {index: 1 for index in range(5)},
                                    "0:stage:0", 50)
    scheduler.policy.candidate_selector = None
    preemptions = []
    original = scheduler._preempt

    def audited(group, blocks, mode):
        result = original(group, blocks, mode)
        preemptions.append({"request_id": group.request_id, "mode": str(result)})
        return result

    scheduler._preempt = audited
    try:
        for index in range(5):
            add(engine, infos, f"stable-{index}", index, 0, 256, slo=60.0)
        steps = 0
        while engine.engine.has_unfinished_requests() and steps < 300:
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        result = {
            "steps": steps, "batches": read_batches(output),
            "unfinished": engine.engine.has_unfinished_requests(),
            "preemptions": preemptions,
            "events": scheduler.policy.audit_events,
            "compound_events": scheduler.compound_events,
        }
    finally:
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()
    result["summary"] = json.loads((output / "executor_summary.json").read_text())
    return result


def swap_limit_run(root, output):
    infos = public_infos(root)
    engine, scheduler = make_engine(root, output, {0: 1, 1: 1}, "0:stage:0", 50)
    members = [f"limit-compound-{index}" for index in range(4)]
    try:
        for request_id in members:
            add(engine, infos, request_id, 0, 0, 64, slo=60.0)
        engine.engine.step()
        initial = read_batches(output)[-1]
        scheduler.policy.candidate_selector = selector("1:stage:0")
        scheduler.policy.remaining = 0.0
        add(engine, infos, "limit-target", 1, 0, 256, slo=3.0)
        engine.engine.step()
        replacement = read_batches(output)[-1]
        event = scheduler.compound_events[-1]
        steps = 2
        while engine.engine.has_unfinished_requests() and steps < 300:
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        result = {
            "initial": initial, "replacement": replacement, "event": event,
            "batches": read_batches(output), "steps": steps,
            "unfinished": engine.engine.has_unfinished_requests(),
        }
    finally:
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()
    result["summary"] = json.loads((output / "executor_summary.json").read_text())
    return result


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only overhead gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        perf = ToyPerformance()
        pred = CountingPredictor()

        def forbidden(*args, **kwargs):
            raise AssertionError("degenerate state reached projection")

        for count, reason in ((1, "single_goal"), (2, "all_ready_units_fit_fixed_batch")):
            view, members = direct_view(count)
            action, audit = candidate_action(view, 0, set(members), pred, perf,
                                             {index: 0 for index in range(count)},
                                             compound_members=members, project_fn=forbidden)
            details[f"preflight_{count}"] = {"action": action, "audit": audit}
            check(f"preflight_{count}_skips_prediction_and_projection",
                  audit["candidate_discriminability"] == "degenerate"
                  and audit["preflight_reason"] == reason
                  and audit["project_calls"] == 0 and not audit["actions"],
                  details[f"preflight_{count}"],
                  "no predictor or full rollout call is made for a structurally invariant state")
        check("preflight_avoids_all_forecast_calls", pred.calls == 0, pred.calls, 0)

        view, members = direct_view(5)
        projection_calls = []

        def tied(jobs, action, *args, **kwargs):
            projection_calls.append(action)
            finishes = {job["key"]: 1.0 for job in jobs}
            return {"good": 0, "tardiness": 1.0, "mean": 1.0,
                    "finishes": finishes,
                    "batch_trace": [{"keys": [action[0]], "slots": 1, "lease_after": 0.03}]}

        _, saturated = candidate_action(view, 0, set(members), pred, perf,
                                        {index: 0 for index in range(5)},
                                        compound_members=members, project_fn=tied)
        details["saturated"] = saturated
        check("contended_state_is_not_overfiltered",
              saturated["project_calls"] > 0 and saturated["completed_scan"]
              and saturated["project_calls"] == len(projection_calls)
              and saturated["candidate_discriminability"] == "invariant",
              {"project_calls": saturated["project_calls"],
               "actual_calls": len(projection_calls),
               "completed_scan": saturated["completed_scan"],
               "discriminability": saturated["candidate_discriminability"]},
              "only the contended state pays for projection and its accounting is exact")

        timeout_calls = []

        def timeout(jobs, action, *args, **kwargs):
            timeout_calls.append(action)
            if len(timeout_calls) == 2:
                raise TimeoutError("injected projection timeout")
            return tied(jobs, action, *args, **kwargs)

        _, partial = candidate_action(view, 0, set(members), pred, perf,
                                      {index: 0 for index in range(5)},
                                      compound_members=members, project_fn=timeout)
        details["partial"] = partial
        check("partial_scan_is_explicit_not_global_optimum",
              partial["project_calls"] == 2 and len(partial["actions"]) == 1
              and not partial["completed_scan"]
              and partial["unevaluated_actions"] == partial["planned_actions"] - 1,
              {"project_calls": partial["project_calls"],
               "completed": len(partial["actions"]), "planned": partial["planned_actions"],
               "completed_scan": partial["completed_scan"],
               "unevaluated": partial["unevaluated_actions"]},
              "a timeout preserves completed scores and marks every unscored action incomplete")

        stable = stable_run(root, root / "stage6_mechanism_repair/controlled" / attempt / "stable")
        details["stable"] = stable
        check("default_path_has_no_policy_preemption_or_swap",
              not stable["preemptions"]
              and stable["summary"]["backend"]["swap_bytes_out"] == 0,
              {"preemptions": stable["preemptions"],
               "swap_bytes": stable["summary"]["backend"]["swap_bytes_out"]},
              "five requests on the no-model-action path never swap or preempt for priority churn")
        check("default_path_completes_with_bounded_unique_batches",
              not stable["unfinished"] and stable["steps"] < 300
              and all(0 < len(batch) <= 4 and len(batch) == len(set(batch))
                      for batch in stable["batches"])
              and stable["summary"]["tracked_block_owners"] == 0,
              {"steps": stable["steps"], "unfinished": stable["unfinished"],
               "max_batch": max(map(len, stable["batches"])),
               "owners": stable["summary"]["tracked_block_owners"]},
              "stable execution does not starve requests, duplicate work or leak KV")

        running = run_running_anchor(
            root, root / "stage6_mechanism_repair/controlled" / attempt / "model")
        details["model"] = running
        preemptions = [item for event in running["compound_events"]
                       for item in event.get("preemptions", [])]
        check("model_replacement_is_single_attributed_preemption",
              len(preemptions) == 1 and preemptions[0]["cause"] == "model"
              and preemptions[0]["request_id"] == "old-3",
              preemptions, "the intentional waiting-anchor replacement preempts exactly one old member")

        limited = swap_limit_run(root, root / "stage6_mechanism_repair/controlled" / attempt / "swap_limit")
        details["swap_limit"] = limited
        members = {f"limit-compound-{index}" for index in range(4)}
        deferred = limited["event"]["deferred"]
        partial = [batch for batch in limited["batches"] if 0 < len(members & set(batch)) < 4]
        check("oversized_compound_eviction_is_deferred_atomically",
              set(limited["initial"]) == members and limited["replacement"] == ["limit-target"]
              and not limited["event"]["preemptions"]
              and any(set(item[0]) == members and item[2] == "swap_limit" for item in deferred)
              and not partial and not limited["unfinished"]
              and limited["summary"]["tracked_block_owners"] == 0,
              {"replacement": limited["replacement"], "deferred": deferred,
               "partial": partial, "unfinished": limited["unfinished"]},
              "four-member eviction exceeds the three-swap budget, so it is deferred without splitting or leaking")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "overhead gate completed")

    passed = all(item["pass"] for item in checks.values())
    source = Path(__file__).resolve()
    result = {
        "schema": "stage7-overhead-gate/v1", "attempt": attempt,
        "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
        "gpu_used": False, "next_step_allowed": passed,
        "sources": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_OVERHEAD_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
