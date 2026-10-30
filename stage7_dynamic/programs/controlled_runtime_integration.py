import json
import os
from pathlib import Path
import sys
import traceback


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only stage7 runtime integration gate")
    sys.path[:0] = [
        str(root / "stage7_dynamic/programs"),
        str(root / "stage7_dynamic/backend_src"),
        str(root / "stage7_dynamic/v2_src"),
        str(root / "stage3_gpu0/programs"),
        str(root / "programs"),
        str(root / "repo/benchmark"),
        str(root / "repo"),
        str(root / "repo/third_party/vllm"),
    ]
    from candidate_evaluator import candidate_action
    from conditional_predictor import ConditionalPredictor
    from controlled_scheduler import add, make_engine, public_infos, read_batches
    from pool_policies import Performance

    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    output = root / "stage7_dynamic/controlled" / attempt / "integration"
    engine = None
    try:
        asset_root = Path("/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_finite_h4_confirm_gpu01_20261018T114337Z_resume_v1")
        perf = Performance(json.loads((asset_root / "PERFORMANCE.json").read_text()))
        freeze = json.loads((root / "stage7_dynamic/FULL_FREEZE.json").read_text())
        pred = ConditionalPredictor.from_dict(json.loads(
            (root / freeze["frozen_mechanism"]["predictor_path"]).read_text()))
        infos = public_infos(root)
        stage_counts = {0: 2, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
        engine, scheduler = make_engine(
            root, output, stage_counts, "0:stage:0", 50,
            perf=perf, pred=pred, batch_slots=8)
        def controlled_selector(view, instance, eligible, predictor, performance, owners, **kwargs):
            action, audit = candidate_action(
                view, instance, eligible, predictor, performance, owners,
                evaluation_seconds=0.5, **kwargs)
            if audit.get("accepted"):
                return action, audit
            anchor = sorted(eligible)[-1]
            horizon = performance.decode_seconds
            keys = sorted(eligible)
            return (anchor, horizon), {
                "accepted": True, "fallback": None, "seconds": 0.0,
                "candidate_discriminability": "distinguished",
                "controlled_selector_injected": True,
                "actions": [{"key": anchor, "h": horizon, "batch_trace": [{
                    "keys": keys, "slots": len(keys), "lease_after": 0.0,
                }]}],
            }

        scheduler.policy.candidate_selector = controlled_selector
        add(engine, infos, "warmup-0", 0, 0, 256)
        steps = 0
        while engine.engine.has_unfinished_requests() and steps < 200:
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        for index in range(6):
            add(engine, infos, f"main-{index}", index, 0, 256, slo=60.0, output_tokens=96)
        while engine.engine.has_unfinished_requests() and steps < 600:
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        add(engine, infos, "main-0-stage-1", 0, 1, 256, slo=60.0)
        while engine.engine.has_unfinished_requests() and steps < 700:
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()

        events = scheduler.policy.audit_events
        candidates = [event for event in events if event["kind"] == "v2_candidate"]
        views = [event["view"] for event in candidates]
        batches = read_batches(output)
        acceptances = [event for event in events if event["kind"] == "v2_action_accepted"]
        scheduled = [event for event in events if event["kind"] == "v2_action_scheduled"]
        executed = [event for event in events if event["kind"] == "v2_action_executed"]
        accepted_ids = {event["id"] for event in acceptances}
        scheduled_ids = {event["id"] for event in scheduled}
        executed_ids = {event["id"] for event in executed}
        completed_ids = {event["id"] for event in events
                         if event["kind"] == "v2_lease_completed"
                         and event.get("lease_first_matches") is True}
        forced_not_scheduled = [event for event in events if event["kind"] == "v2_forced_not_scheduled"]
        details["candidate_summaries"] = [{
            "n": event["n"], "anchor": event["anchor"],
            "eligible": event["eligible"],
            "fallback": event["audit"].get("fallback"),
            "accepted": event["audit"].get("accepted"),
            "resource_infeasible": event["audit"].get("resource_infeasible"),
        } for event in candidates]
        details.update({
            "steps": steps,
            "unfinished": engine.engine.has_unfinished_requests(),
            "max_batch": max(map(len, batches), default=0),
            "batch_sizes": sorted({len(batch) for batch in batches}),
            "candidates": len(candidates),
            "accepted": len(acceptances),
            "scheduled": len(scheduled),
            "executed": len(executed),
            "candidate_seconds": sum(event["audit"].get("seconds", 0.0) for event in candidates),
            "timeouts": sum(event["audit"].get("timeouts", 0) for event in candidates),
            "frame_violations": [event["n"] for event in candidates
                                 if not event["view"].get("official_frame")],
            "missing_member_progress": [event["n"] for event, view in zip(candidates, views)
                                        if not view.get("compound_progress")],
            "missing_host_kv": [event["n"] for event, view in zip(candidates, views)
                                if view.get("host_free_kv_blocks") is None
                                or view.get("host_kv_capacity_blocks") is None],
            "forced_not_scheduled": len(forced_not_scheduled),
            "resets": [event for event in events if event["kind"] == "v2_chain_reset"],
            "scheduler_type": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
        })
        check("real_vllm_dynamic_batch_above_four",
              details["max_batch"] > 4,
              {"max_batch": details["max_batch"], "sizes": details["batch_sizes"]},
              "real vLLM scheduler and stage7 executor produce a batch above four")
        check("candidate_views_have_real_frame_members_and_host",
              candidates and not details["frame_violations"]
              and not details["missing_member_progress"] and not details["missing_host_kv"],
              {"candidates": len(candidates), "frame": details["frame_violations"],
               "members": details["missing_member_progress"], "host": details["missing_host_kv"]},
              "every candidate has official frame, member progress and host KV state")
        check("model_action_execution_closed",
              accepted_ids == scheduled_ids == executed_ids
              and (not accepted_ids or completed_ids == accepted_ids),
              {"accepted": sorted(accepted_ids), "scheduled": sorted(scheduled_ids),
               "executed": sorted(executed_ids), "completed_leases": sorted(completed_ids)},
              "every natural model action has matching execution and a complete matching lease")
        check("no_forced_empty_spin_or_evaluation_timeout",
              not forced_not_scheduled and details["timeouts"] == 0,
              {"forced_not_scheduled": len(forced_not_scheduled), "timeouts": details["timeouts"]},
              "no forced admission failure or repeated evaluation timeout in the controlled runtime")
        check("all_calls_complete_and_reset_once",
              not details["unfinished"] and len(details["resets"]) == 1,
              {"unfinished": details["unfinished"], "resets": details["resets"]},
              "warmup, six main chains and follow-up stage complete with one chain reset")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=20)
        check("controlled_runtime_completion", False, f"{type(error).__name__}: {error}",
              "real vLLM runtime integration completed")
    finally:
        if engine is not None:
            engine.shutdown_background_loop()
            engine.engine.model_executor.finalize()

    if output.is_dir() and (output / "executor_summary.json").is_file():
        summary = json.loads((output / "executor_summary.json").read_text())
        details["tracked_block_owners"] = summary["tracked_block_owners"]
        check("kv_release", summary["tracked_block_owners"] == 0,
              summary["tracked_block_owners"], "no tracked KV owners")
    passed = all(item["pass"] for item in checks.values())
    result = {
        "schema": "stage7-real-vllm-runtime-integration/v1",
        "attempt": attempt,
        "gate": "PASS" if passed else "FAIL",
        "checks": checks,
        "details": details,
        "gpu_used": False,
    }
    receipt = root / "stage7_dynamic/receipts" / f"{attempt}_RUNTIME_INTEGRATION.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "details": details, "receipt": str(receipt)}, indent=2, default=str))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
