import ast
import json
import os
from pathlib import Path
import sys
import time
import traceback

from controlled_scheduler import add, make_engine, public_infos


def wrap_calls(path):
    tree = ast.parse(path.read_text())
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "wrap":
            continue
        if len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and isinstance(node.args[1], ast.Constant):
            calls.append((node.args[0].id, node.args[1].value))
    return calls


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only audit and projection contract gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        c_calls = wrap_calls(root / "stage6_mechanism_repair/programs/c_server_audit.py")
        b_calls = wrap_calls(root / "stage3_gpu0/programs/bc_b_server_audit.py")
        details["audit_wraps"] = {"C": c_calls, "B": b_calls}
        check("C_has_single_outer_schedule_wrapper",
              ("SLOScheduler", "schedule") not in c_calls
              and c_calls.count(("V2SLOScheduler", "schedule")) == 1,
              c_calls, "no nested base schedule wrapper; one V2 outer wrapper remains")
        check("B_keeps_single_native_schedule_wrapper",
              b_calls.count(("SLOScheduler", "schedule")) == 1,
              b_calls, "B-cap4 audit remains single-layer")

        output = root / "stage6_mechanism_repair/controlled" / attempt / "contracts"
        infos = public_infos(root)
        engine, scheduler = make_engine(root, output, {0: 1, 4: 1}, "0:stage:0", 50)
        try:
            add(engine, infos, "z-request", 0, 0, 64, slo=60.0)
            add(engine, infos, "a-request", 4, 0, 64, slo=60.0)
            groups = {group.request_id: group for group in scheduler.waiting}
            for group in groups.values():
                scheduler.policy.current[group.request_id]["ready_since"] = 100.0
            scheduler.policy.before_schedule(list(groups.values()), 101.0, scheduler.v2_ledger)
            scheduler.policy.model_action = True
            scheduler.policy.forced = None
            ordered = scheduler.policy.dispatch_order(list(groups.values()))
            details["tie_order"] = [group.request_id for group in ordered]
            check("goal_key_breaks_ties_not_request_uuid",
                  details["tie_order"] == ["z-request", "a-request"],
                  {"order": details["tie_order"],
                   "priorities": {name: scheduler.policy.priorities[name] for name in groups}},
                  "0:stage:0 precedes 4:stage:0 even though z-request sorts after a-request")

            scheduler.policy.pending_action = {
                "id": 1, "anchor": "0:stage:0", "expected_first": ["0:stage:0"],
                "scheduled": False, "executed": False,
            }
            report = {"initial_available_blocks": 10, "remaining_available_blocks": 7,
                      "accepted": [["z-request"]], "deferred": []}
            scheduler.policy.note_selection([groups["z-request"]], time.perf_counter(), report)
            event = scheduler.policy.audit_events[-1]
            details["scheduled_event"] = event
            check("scheduled_action_includes_actual_dispatch_report",
                  event["kind"] == "v2_action_scheduled"
                  and event["dispatch_report"] == report
                  and event["matches_projection"] is True,
                  event, "resource accounts are attached to the action witness")
            check("recompute_callback_is_installed",
                  scheduler.recompute_invalidate == engine.engine.model_executor.invalidate_request,
                  scheduler.recompute_invalidate is not None,
                  "the V2 scheduler notifies the same executor on RECOMPUTE")
        finally:
            engine.shutdown_background_loop()
            engine.engine.model_executor.finalize()
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "contract gate completed")

    passed = all(item["pass"] for item in checks.values())
    result = {"schema": "stage9-audit-projection-gate/v1", "attempt": attempt,
              "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
              "gpu_used": False}
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_CONTRACTS_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
