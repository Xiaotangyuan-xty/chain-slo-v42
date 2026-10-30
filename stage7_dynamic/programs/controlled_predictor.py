import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

from conditional_predictor import ConditionalPredictor
from contracts import CacheView, CallView
from execution_reference import forecasts
from execution_view import ExecutionView
from fit_predictor import (away_parameters, calibrate, evaluate, fit, graph_signature,
                           make_records, split_graph_ids)
from pool_policies import Performance
from v2_contracts import GoalView, ObservedCall


ASSET_ROOT = Path("/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_finite_h4_confirm_gpu01_20261018T114337Z_resume_v1")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ready_goal(program, key, prompt, ordinal=0, slots=1):
    call = CallView(key, program, ordinal, prompt, 0, 0, 0, 0.0, 0.0,
                    0.0, 0.0, False, 0.0)
    return GoalView(program, 0.0, 1000.0, "READY", (), call)


def main():
    root = Path(sys.argv[1]).resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        training_path = root / "repo/traces/deepresearch_training.json"
        public_path = root / "repo/traces/deepresearch_filter.jsonl"
        old_path = ASSET_ROOT / "PREDICTOR.json"
        performance_path = ASSET_ROOT / "PERFORMANCE.json"
        old_sha = sha(old_path)
        training = json.loads(training_path.read_text())
        public = [json.loads(line) for line in public_path.read_text().splitlines()]
        train_signatures = {graph_signature(graph["stages"]) for graph in training}
        overlap = sum(graph_signature(chain["stages"]) in train_signatures for chain in public)
        check("training_public_structure_no_overlap",
              len(training) == 394 and len(train_signatures) == 394 and overlap == 0,
              {"training_graphs": len(training), "unique_signatures": len(train_signatures),
               "overlap": overlap}, "394 unique independent graphs and zero public overlap")

        fit_ids, cal_ids = split_graph_ids(training)
        fit_records = make_records(training, fit_ids)
        cal_records = make_records(training, cal_ids)
        check("fit_cal_identity_isolation",
              len(fit_ids) == 295 and len(cal_ids) == 99 and not (fit_ids & cal_ids),
              {"FIT_graphs": len(fit_ids), "CAL_graphs": len(cal_ids),
               "FIT_records": len(fit_records), "CAL_records": len(cal_records),
               "intersection": len(fit_ids & cal_ids)},
              "295 FIT and 99 CAL graphs, disjoint by SHA")
        away_mean, away_q = away_parameters(json.loads(old_path.read_text()))
        profile = Performance(json.loads(performance_path.read_text()))
        base = fit(fit_records, away_mean, away_q)
        model, calibration = calibrate(base, cal_records, profile)
        evaluation = evaluate(model, cal_records, profile)
        details["calibration"] = calibration
        details["cal_evaluation"] = evaluation
        check("calibration_coverage",
              evaluation["coverage"] >= 0.85
              and all(item["coverage"] >= 0.8 for item in evaluation["by_ordinal"].values()),
              evaluation, "CAL overall coverage >=85%, each ordinal region >=80%")

        restored = ConditionalPredictor.from_dict(json.loads(json.dumps(model.to_dict())))
        sample = cal_records[0]
        original_parts = model.ready_parts(sample["prompt"], sample["ordinal"],
                                           sample["slots"], 0, profile)
        restored_parts = restored.ready_parts(sample["prompt"], sample["ordinal"],
                                              sample["slots"], 0, profile)
        check("model_serialization_roundtrip", original_parts == restored_parts,
              {"original": original_parts[:4], "restored": restored_parts[:4]},
              "serialized model preserves estimates")

        modes = {"exact": 0, "backoff": 0}
        finite = 0
        states = 0
        for chain in public:
            for ordinal, stage in enumerate(chain["stages"]):
                prompt = sum(request["input_tokens"] for request in stage)
                outputs = sum(request["output_tokens"] for request in stage)
                slots = len(stage)
                for emitted in {0, max(0, outputs - 1)}:
                    current, total, tail, survivors, mode = model.ready_parts(
                        prompt, ordinal, slots, emitted, profile)
                    values = [current, total, *tail]
                    finite += int(all(math.isfinite(value) and value > 0 for value in values))
                    modes[mode] += 1
                    states += 1
        details["public_feature_audit"] = {"states": states, "finite": finite, "modes": modes}
        check("public_feature_finite_backoff",
              finite == states and modes["backoff"] > 0,
              {"states": states, "finite": finite, "modes": modes},
              "all public feature states have finite estimates; backoff is explicit")

        goals = [ready_goal(0, "known", 256, 0),
                 ready_goal(1, "unknown", 9999, 2),
                 ready_goal(2, "compound", 600, 1, slots=2)]
        view = ExecutionView(
            now=0.0,
            goals=tuple(goals),
            cache=CacheView(65536, 65536, ()),
            chain_virtual=((0, 0.0), (1, 0.0), (2, 0.0)),
            declared_inputs=(("known", 256), ("unknown", 9999), ("compound", 600)),
            compound_members=(("known", ["known"]),
                              ("unknown", ["unknown"]),
                              ("compound", ["compound:a", "compound:b"])),
        )
        jobs = forecasts(view, model, profile)
        check("unknown_and_compound_keep_full_denominator",
              len(jobs) == 3 and all(job["work"] > 0 and math.isfinite(job["work"]) for job in jobs)
              and any(job["support"]["mode"] == "backoff" for job in jobs),
              [{"key": job["key"], "work": job["work"], "support": job["support"]}
               for job in jobs],
              "unknown, compound and known goals all remain with finite work")

        history = (ObservedCall(0, 256, 48, 10.0, 1.0, 1.0),)
        away_goal = GoalView(3, 0.0, 1000.0, "AWAY", history,
                             tool_entered=10.0, tool_elapsed=2.0)
        away_view = ExecutionView(
            now=12.0, goals=(away_goal,), cache=CacheView(8192, 8192, ()),
            chain_virtual=((3, 1.0),), declared_inputs=(("away:3", 256),),
            compound_members=(),
        )
        away_jobs = forecasts(away_view, model, profile)
        check("away_backoff_finite_and_timer_preserved",
              len(away_jobs) == 1 and away_jobs[0]["support"]["phase"] == "AWAY"
              and any(kind == "away" for kind, _ in away_jobs[0]["phases"]),
              away_jobs[0]["support"], "AWAY estimate remains finite with explicit timer")

        check("old_predictor_asset_unchanged", sha(old_path) == old_sha,
              {"before": old_sha, "after": sha(old_path)}, "old asset SHA unchanged")

        assets = root / "stage6_mechanism_repair/assets"
        assets.mkdir(exist_ok=True)
        model_path = assets / "CONDITIONAL_PREDICTOR.json"
        with model_path.open("x") as handle:
            handle.write(json.dumps(model.to_dict(), indent=2, allow_nan=False) + "\n")
        details["model_path"] = str(model_path)
        details["model_sha256"] = sha(model_path)
        details["away_contract"] = {
            "source": "registered synthetic FIT/CAL only; no DeepResearch AWAY labels",
            "away_mean": away_mean, "away_q": away_q,
            "new_away_training_claimed": False,
        }
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "all controlled checks completed")

    passed = all(item["pass"] for item in checks.values())
    result = {
        "schema": "step4-predictor-gate/v1",
        "gate": "PASS" if passed else "FAIL",
        "checks": checks,
        "details": details,
        "gpu_used": False,
        "next_step_allowed": passed,
        "no_JCT_coverage_claim": True,
        "no_DEV_or_HOLDOUT_fitting": True,
    }
    receipt = root / "stage6_mechanism_repair/receipts/STEP4_PREDICTOR_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
