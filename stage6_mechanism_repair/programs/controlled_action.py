import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from candidate_evaluator import candidate_action
from controlled_evaluation import run_parity
from controlled_scheduler import (add, make_engine, public_infos, read_batches,
                                  run_basic, run_swap)


def run_running_anchor(root, output):
    infos = public_infos(root)
    engine, scheduler = make_engine(root, output, {index: 1 for index in range(5)},
                                    "0:stage:0", 50)
    scheduler.policy.candidate_selector = candidate_action
    try:
        for index in range(4):
            add(engine, infos, f"old-{index}", index, 0, 256, slo=60.0)
        engine.engine.step()
        before = read_batches(output)[-1]

        def inject_nondefault(*args, **kwargs):
            _, audit = candidate_action(*args, **kwargs)
            chosen = next(item for item in audit["actions"]
                          if item["key"] == "4:stage:0" and item["h"] == 0.04)
            audit.update(accepted=True, injected_action=True, fallback=None)
            return (chosen["key"], chosen["h"]), audit

        scheduler.policy.candidate_selector = inject_nondefault
        add(engine, infos, "new-4", 4, 0, 384, slo=3.0)
        scheduler.policy.remaining = 0.0
        engine.engine.step()
        after = read_batches(output)[-1]
        candidate = [event for event in scheduler.policy.audit_events
                     if event["kind"] == "v2_candidate"][-1]
        chosen = next((item for item in candidate["audit"]["actions"]
                       if item["key"] == candidate["anchor"]
                       and item["h"] == candidate["remaining"]), None)
        return {
            "before": before, "after": after,
            "accepted": candidate["audit"]["accepted"],
            "anchor": candidate["anchor"], "audit": candidate["audit"],
            "expected_first": chosen["batch_trace"][0]["keys"] if chosen else None,
            "schedule_count": scheduler.policy.num_schedule_count,
            "queues": {name: [group.request_id for group in getattr(scheduler, name)]
                       for name in ("running", "waiting", "swapped")},
            "compound_events": scheduler.compound_events,
        }
    finally:
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only action gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        parity = run_parity(root, root / "stage6_mechanism_repair/controlled" / attempt / "action")
        details["parity"] = parity
        first_expected = parity["expected_keys"][0] if parity["expected_keys"] else []
        first_actual = parity["actual_keys"][0] if parity["actual_keys"] else []
        accepted = next((item for item in parity["action_summaries"]
                         if item["key"] == parity["anchor"]
                         and item["h"] == parity["horizon"] and item["differs"]), None)
        check("nondefault_action_is_distinguished",
              parity["accepted"] and accepted is not None
              and parity["anchor"] != parity["action_summaries"][0]["key"],
              {"accepted": parity["accepted"], "anchor": parity["anchor"],
               "candidate": accepted},
              "the actual winning non-default goal has a distinguished projection")
        check("nondefault_anchor_enters_first_real_batch",
              parity["anchor"] in first_actual,
              {"expected": first_expected, "actual": first_actual},
              "the waiting non-default anchor is really scheduled, not merely retained in running queues")
        check("first_batch_matches_projection", first_actual == first_expected,
              {"expected": first_expected, "actual": first_actual},
              "the first actual unit set equals the chosen action's projected unit set")
        check("fixed_four_request_envelope",
              parity["actual_slots"] and max(parity["actual_slots"]) <= 4
              and max(parity["expected_slots"]) <= 4,
              {"actual_max": max(parity["actual_slots"], default=0),
               "expected_max": max(parity["expected_slots"], default=0)},
              "neither projection nor execution expands beyond four real requests")
        after = parity["actual_lease"]
        expected_after = parity["expected_lease"]
        count = min(4, len(after), len(expected_after))
        check("accepted_lease_is_consumed",
              count > 0 and all(abs(actual - expected) < 1e-9
                                for actual, expected in zip(after[:count], expected_after[:count])),
              {"actual": after[:count], "expected": expected_after[:count],
               "chosen_horizon": parity["horizon"]},
              "the selected lease, whether short or long, is consumed exactly as projected")
        running = run_running_anchor(
            root, root / "stage6_mechanism_repair/controlled" / attempt / "running_anchor")
        details["running_anchor"] = running
        check("running_state_has_distinguished_nondefault_action",
              running["accepted"] and running["anchor"] == "4:stage:0",
              {"accepted": running["accepted"], "anchor": running["anchor"]},
              "the late-arriving urgent goal wins while four older requests are running")
        check("waiting_anchor_replaces_running_member",
              "new-4" in running["after"] and running["after"] != running["before"],
              {"before": running["before"], "after": running["after"],
               "schedule_count": running["schedule_count"], "queues": running["queues"]},
              "the accepted waiting anchor really enters on a non-frame tick")
        check("running_state_projection_matches_execution",
              running["after"] and running["expected_first"] is not None
              and len(running["after"]) == len(running["expected_first"])
              and set(running["after"]) == {
                  request_id for key in running["expected_first"]
                  for request_id in [f"new-{key.split(':')[0]}"] if key == "4:stage:0"
              } | {
                  request_id for key in running["expected_first"]
                  for request_id in [f"old-{key.split(':')[0]}"] if key != "4:stage:0"
              },
              {"expected": running["expected_first"], "actual": running["after"]},
              "the real member set matches the projected first action, independent of request aliases")
        compound = run_basic(root, root / "stage6_mechanism_repair/controlled" / attempt / "compound", 1)
        details["compound"] = compound
        partial = [batch for batch in compound["batches"]
                   if ("compound-1a" in batch) != ("compound-1b" in batch)]
        action_events = [event for event in compound["policy"]["events"]
                         if event["kind"] in ("v2_action_scheduled", "v2_action_executed")]
        check("compound_action_is_atomic_and_executed",
              compound["batches"] and not partial
              and {"compound-1a", "compound-1b"} <= set(compound["batches"][0])
              and any(event["kind"] == "v2_action_executed" for event in action_events),
              {"first": compound["batches"][0] if compound["batches"] else None,
               "partial": partial, "action_events": action_events},
              "both members enter together and an accepted action has an execution witness")
        check("compound_respects_fixed_capacity_and_releases",
              max(map(len, compound["batches"])) <= 4 and not compound["unfinished"]
              and compound["summary"]["tracked_block_owners"] == 0,
              {"max_batch": max(map(len, compound["batches"])),
               "unfinished": compound["unfinished"], "summary": compound["summary"]},
              "no batch exceeds four, all calls finish and all KV owners are released")
        swap = run_swap(root, root / "stage6_mechanism_repair/controlled" / attempt / "swap")
        details["swap"] = swap
        swap_partial = [batch for batch in swap["batches"]
                        if ("compound-1a" in batch) != ("compound-1b" in batch)]
        check("swap_completion_unique_and_atomic",
              not swap_partial and not swap["unfinished"]
              and swap["summary"]["tracked_block_owners"] == 0
              and all(len(batch) == len(set(batch)) and len(batch) <= 4
                      for batch in swap["batches"]),
              {"partial": swap_partial, "unfinished": swap["unfinished"],
               "preemptions": len(swap["preemptions"]), "summary": swap["summary"]},
              "KV pressure preserves atomic unique bounded batches and releases every owner")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "action gate completed")

    passed = all(item["pass"] for item in checks.values())
    source = Path(__file__).resolve()
    result = {
        "schema": "stage7-action-gate/v1", "attempt": attempt,
        "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
        "gpu_used": False, "next_step_allowed": passed,
        "sources": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_ACTION_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
