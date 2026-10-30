import json
import os
from pathlib import Path
import sys
import traceback

from contracts import CallView
from support_state import SupportState
from v2_contracts import GoalView, ObservedCall
from v2_predictor import Predictor


ASSET = Path("/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_finite_h4_confirm_gpu01_20261018T114337Z_resume_v1/PREDICTOR.json")


def toy_predictor():
    records = ((0, 48, 0.0, 15.0, (10.0, 5.0)),)
    return Predictor((((2, 0), records),), 0.0, ())


def ready_goal(key="chain", prompt=256, ordinal=0, emitted=0):
    call = CallView(key, key, ordinal, prompt, emitted, 0, 0, 0.0, 0.0,
                    0.0, 0.0, False, 0.0)
    return GoalView(key, 0.0, 100.0, "READY", (), call)


def away_goal(elapsed):
    history = (ObservedCall(0, 256, 48, 10.0, 1.0, 1.0),)
    return GoalView("away", 0.0, 100.0, "AWAY", history,
                    tool_entered=20.0, tool_elapsed=elapsed)


def main():
    root = Path(sys.argv[1]).resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        state = SupportState(toy_predictor())
        initial = state.ready(256, 0, 0)
        exhausted = state.ready(256, 0, 48)
        calls_after_exhaustion = state.ready_calls
        cached = state.ready(256, 0, 48)
        check("negative_cache_is_bounded", initial[0] and not exhausted[0] and not cached[0]
              and state.ready_calls == calls_after_exhaustion + 1,
              {"initial": initial, "exhausted": exhausted, "cached": cached},
              "initial support, exhausted support, one cached repeat")
        new_chain = state.ready(256, 0, 0)
        check("warmup_to_main_rechecks_earlier_emitted", new_chain[0]
              and state.ready_calls == calls_after_exhaustion + 2,
              {"new_chain": new_chain, "ready_calls": state.ready_calls},
              "new chain at emitted zero is supported and really rechecked")

        before = state.snapshot()
        success = state.observe_audit({"accepted": True, "fallback": None},
                                      [ready_goal()], 0.01)
        no_gain = state.observe_audit({"accepted": False, "fallback": "no_model_gain"},
                                      [ready_goal()], 0.01)
        timeout = state.observe_audit({"accepted": False, "fallback": "lease-aware reference budget"},
                                      [ready_goal()], 0.01)
        error = state.observe_audit({"accepted": False, "fallback": "synthetic evaluator failure"},
                                    [ready_goal()], 0.01)
        check("accepted_and_non_support_errors_never_poison",
              success == "success" and no_gain == "no_model_gain" and timeout == "timeout"
              and error == "evaluation_error" and state.snapshot() == before,
              {"success": success, "no_gain": no_gain, "timeout": timeout,
               "error": error, "snapshot_unchanged": state.snapshot() == before},
              "success, no_model_gain, timeout and evaluation_error preserve support")

        good = ready_goal("good", 256, 0, 0)
        bad = ready_goal("bad", 384, 0, 0)
        report = state.check_goals([good, bad], 0.01)
        check("single_goal_unsupported_does_not_poison_peers",
              not report.supported and report.denominator == 2
              and report.unsupported_keys == ["bad"]
              and state.ready(256, 0, 0)[0],
              {"denominator": report.denominator, "unsupported": report.unsupported_keys,
               "results": [result.__dict__ for result in report.results]},
              "unsupported goal is named, denominator stays complete, peer remains supported")

        eligible = {"good"}
        action, audit = state.fallback_action([good.ready, bad.ready], eligible, 0.01)
        empty_rejected = False
        try:
            state.fallback_action([good.ready, bad.ready], set(), 0.01)
        except ValueError:
            empty_rejected = True
        check("support_fallback_respects_eligible", action[0] == "good" and empty_rejected,
              {"action": action, "audit": audit, "empty_rejected": empty_rejected},
              "fallback uses an eligible composite and rejects an empty eligible set")

        away_state = SupportState(toy_predictor())
        away_initial = away_state.away(away_goal(0.0), 1.0)
        away_exhausted = away_state.away(away_goal(10.0), 1.0)
        away_calls = away_state.away_calls
        away_cached = away_state.away(away_goal(10.0), 1.0)
        away_earlier = away_state.away(away_goal(5.0), 1.0)
        check("away_survival_boundary_rechecks_earlier_elapsed",
              away_initial[0] and not away_exhausted[0] and not away_cached[0]
              and away_earlier[0] and away_state.away_calls == away_calls + 2,
              {"initial": away_initial, "exhausted": away_exhausted,
               "cached": away_cached, "earlier": away_earlier},
              "AWAY negative cache is elapsed-aware and earlier survival is rechecked")

        snapshot = state.snapshot()
        json.dumps(snapshot, allow_nan=False)
        key_count = len(snapshot)
        for _ in range(20):
            state.ready(384, 0, 0)
            state.ready(384, 0, 1)
        check("snapshot_serializable_and_bounded", len(state.snapshot()) == key_count,
              {"before": key_count, "after": len(state.snapshot()), "snapshot": state.snapshot()},
              "one bounded entry per state key")

        if ASSET.is_file():
            frozen = SupportState(Predictor.from_dict(json.loads(ASSET.read_text())))
            first = frozen.ready(375, 0, 0)
            exhausted_at = None
            for emitted in range(1025):
                if not frozen.ready(375, 0, emitted)[0]:
                    exhausted_at = emitted
                    break
            reset = frozen.ready(375, 0, 0) if exhausted_at is not None else (None, None)
            check("frozen_predictor_warmup_main_contract",
                  first[0] and exhausted_at is not None and reset[0],
                  {"initial": first, "unsupported_from": exhausted_at, "reset": reset},
                  "the actual frozen predictor supports the new main chain after warmup exhaustion")
            details["frozen_exhausted_at"] = exhausted_at
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=8)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "all checks completed")

    passed = all(item["pass"] for item in checks.values())
    full = ASSET.is_file()
    result = {
        "schema": "step1-support-gate/v1",
        "gate": ("PASS" if full else "REFERENCE_ONLY_PASS") if passed else "FAIL",
        "checks": checks,
        "details": details,
        "gpu_used": False,
        "next_step_allowed": passed and full,
    }
    attempt = sys.argv[2] if len(sys.argv) > 2 else "STEP1_SUPPORT"
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
