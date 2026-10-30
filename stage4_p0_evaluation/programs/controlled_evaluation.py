import asyncio
import copy
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from candidate_evaluator import candidate_action
from contracts import CacheView, CallView
from lease_reference import project
from pool_policies import Performance
from v2_contracts import GoalView, View
from v2_predictor import Predictor


ASSET_ROOT = Path("/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_finite_h4_confirm_gpu01_20261018T114337Z_resume_v1")


class ToyPerformance:
    decode_seconds = 0.01

    def iteration(self, batch_slots, context):
        return 0.01 * batch_slots


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_new(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def direct_jobs(count=5):
    work = [1.5, 4.0, 6.0, 8.0, 10.0]
    deadlines = [104.0, 120.0, 118.0, 116.0, 114.0]
    return [dict(key=f"g{i}", state="READY", deadline=deadlines[i], instance=0,
                 context=100, current=work[i], phases=(("compute", work[i]),), slots=1)
            for i in range(count)]


def frozen_view(prompt=256, ordinal=0):
    goals = []
    for index in range(5):
        key = f"g{index}"
        call = CallView(key, key, ordinal, prompt, 0, 0, 0, 0.0, 0.0,
                        0.0, 0.0, False, 0.0)
        goals.append(GoalView(key, 100.0, 110.0 + 10 * index,
                              "READY", (), call))
    return View(100.0, tuple(goals), CacheView(8192, 8192, ()))


def run_compound_engine(root, perf, pred, attempt):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import SamplingParams
    from fake_backend import FakeBackend
    from harness_executor import HarnessAsyncExecutor, HarnessAsyncLLMEngine
    from v2_compound_policy_eval import install_v2_compound

    output = root / "stage4_p0_evaluation/controlled" / attempt
    output.mkdir(parents=True, exist_ok=False)
    prepare = json.loads((root / "PREPARE.json").read_text())
    HarnessAsyncExecutor.configure(output, FakeBackend)
    args = AsyncEngineArgs(
        model=prepare["model"], tokenizer=prepare["model"], device="cpu", dtype="bfloat16",
        scheduling_policy="fcfs", max_num_seqs=4, max_num_batched_tokens=2048,
        max_model_len=8192, num_gpu_blocks_override=512, swap_space=1,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        enforce_eager=True, disable_async_output_proc=True, disable_log_stats=True,
        seed=42,
    )
    engine = HarnessAsyncLLMEngine.from_engine_args(args, start_engine_loop=False)
    try:
        scheduler = install_v2_compound(engine, perf, pred,
                                        stage_counts={index: 6 for index in range(4)})
        original_execute = engine.engine.model_executor.execute_model

        @functools.wraps(original_execute)
        def timed_execute(execute_model_req):
            started = time.perf_counter()
            result = original_execute(execute_model_req=execute_model_req)
            scheduler.policy.after_model_step(time.perf_counter() - started,
                                              time.perf_counter())
            return result

        engine.engine.model_executor.execute_model = timed_execute
        from jitserve.request_info import RequestType as OfficialRequestType
        from trace.trace_deepresearch import DeepResearchTrace
        infos = DeepResearchTrace.load_trace(
            str(root / "repo/traces/deepresearch_filter.jsonl"))[0].to_request_infos()
        collections = [0, 1, 1, 2, 3]
        for index, (info, collection) in enumerate(zip(infos, collections)):
            request_info = copy.deepcopy(info)
            request_info.request_type = OfficialRequestType.COLLECTIVE
            request_info.collection_id = collection
            request_info.stage_id = 0
            request_info.deadline = 120.0 + 60.0 * index
            request_info.slo_constraint = (2.0, 0.1, 20.0 + 10.0 * index)
            request_info.output_len = 1024
            asyncio.run(engine.engine.add_request_async(
                f"eval-{index}", prompt=TokensPrompt(prompt_token_ids=[42] * 256),
                params=SamplingParams(temperature=0.0, top_p=1.0, top_k=1,
                                      max_tokens=4, seed=42),
                request_info=request_info, arrival_time=time.time(),
            ))
        pending = [group.request_id for group in scheduler.pending]
        steps = 0
        while engine.engine.has_unfinished_requests() and steps < 100:
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        policy = scheduler.policy.snapshot()
        candidate_events = [event for event in policy["events"]
                            if event["kind"] == "v2_candidate"]
        if not candidate_events:
            raise RuntimeError("compound engine produced no candidate event")
        events = [json.loads(line) for line in
                  (output / "executor_events.jsonl").read_text().splitlines()]
        batches = [[group["request_id"] for group in event["groups"]]
                   for event in events if event["kind"] == "execute_model"]
        partial = [batch for batch in batches
                   if ("eval-1" in batch) != ("eval-2" in batch)]
        return {
            "pending": pending,
            "steps": steps,
            "unfinished": engine.engine.has_unfinished_requests(),
            "batches": batches,
            "partial_compound_batches": partial,
            "over_slot_batches": [batch for batch in batches if len(batch) > 4],
            "policy": policy,
            "first_candidate": candidate_events[0],
            "candidate_path": inspect.getfile(candidate_action),
            "policy_path": inspect.getfile(install_v2_compound),
            "finish_count": sum(event["kind"] == "v2_finish"
                                for event in policy["events"]),
        }
    finally:
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    reference_only = "--reference-only" in sys.argv[3:]
    receipt = root / "stage4_p0_evaluation/receipts" / f"{attempt}_EVALUATION_GATE.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    checks = {}
    details = {}
    source = json.loads((root / "stage4_p0_evaluation/SOURCE.json").read_text())

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        check("cpu_only", os.environ.get("CUDA_VISIBLE_DEVICES") == "",
              os.environ.get("CUDA_VISIBLE_DEVICES"), "empty CUDA_VISIBLE_DEVICES")
        perf = ToyPerformance()
        symmetric = direct_jobs(2)
        for job in symmetric:
            job["deadline"] = 120.0
            job["phases"] = (("compute", 8.0),)
        symmetric_scores = [project(symmetric, (f"g{index}", 0.04),
                                    100.0, perf, 0) for index in range(2)]
        check("underfilled_symmetry_not_mislabeled",
              symmetric_scores[0]["good"] == symmetric_scores[1]["good"]
              and math.isclose(symmetric_scores[0]["mean"], symmetric_scores[1]["mean"]),
              [{"good": score["good"], "mean": score["mean"]}
               for score in symmetric_scores], "symmetric candidates may tie")

        scores = {f"g{index}": project(direct_jobs(), (f"g{index}", 0.04),
                                      100.0, perf, 0) for index in range(5)}
        distinguished = len({(score["good"], round(score["mean"], 10))
                             for score in scores.values()}) > 1
        check("over_capacity_multi_goal_discrimination", distinguished,
              {key: {"good": score["good"], "mean": score["mean"]}
               for key, score in scores.items()}, "at least two distinct scores")

        short = project(direct_jobs(), ("g1", 0.04), 100.0, perf, 0)
        long = project(direct_jobs(), ("g1", 0.64), 100.0, perf, 0)
        check("lease_horizon_changes_outcome",
              not math.isclose(short["mean"], long["mean"], rel_tol=1e-12, abs_tol=1e-12)
              and short["lease_runs"] > 0 and long["lease_runs"] > 0,
              {"short": {"mean": short["mean"], "lease_runs": short["lease_runs"]},
               "long": {"mean": long["mean"], "lease_runs": long["lease_runs"]}},
              "the first action persists beyond one dispatch")

        compound = direct_jobs(3)
        for job, slots, work in zip(compound, (2, 2, 1), (4.0, 6.0, 8.0)):
            job["slots"] = slots
            job["current"] = work
            job["phases"] = (("compute", work),)
        compound_score = project(compound, ("g0", 0.64), 100.0, perf, 0)
        check("compound_atomic_request_slots",
              compound_score["max_batch_slots"] <= 4
              and compound_score["max_batch_goals"] < 3
              and compound_score["denominator"] == 3,
              {key: compound_score[key] for key in
               ("denominator", "max_batch_slots", "max_batch_goals")},
              "five initial slots cannot execute as three independent singleton jobs")
        try:
            too_wide = direct_jobs(1)
            too_wide[0]["slots"] = 5
            project(too_wide, ("g0", 0.04), 100.0, perf, 0)
            wide_rejected = False
        except ValueError:
            wide_rejected = True
        check("oversized_compound_rejected_atomically", wide_rejected,
              wide_rejected, True)

        away = direct_jobs(2)
        away[0]["phases"] = (("compute", 1.0), ("away", 3.0), ("compute", 1.0))
        away[1]["phases"] = (("compute", 1.0),)
        away_score = project(away, ("g0", 0.04), 100.0, perf, 0)
        check("multi_goal_away_calendar_and_conservation",
              away_score["denominator"] == 2 and away_score["timer_intervals"] == 1
              and away_score["idle"] > 0
              and math.isclose(away_score["elapsed"],
                               away_score["busy"] + away_score["idle"], abs_tol=1e-8)
              and len(away_score["finishes"]) == 2,
              {key: away_score[key] for key in
               ("denominator", "timer_intervals", "busy", "idle", "elapsed")},
              "independent AWAY timer, complete denominator and conserved clock")

        if not reference_only:
            predictor_path = ASSET_ROOT / "PREDICTOR.json"
            performance_path = ASSET_ROOT / "PERFORMANCE.json"
            before = {path.name: sha(path) for path in (predictor_path, performance_path)}
            perf = Performance(json.loads(performance_path.read_text()))
            pred = Predictor.from_dict(json.loads(predictor_path.read_text()))
            members = {"g0": ["g0:a", "g0:b"], "g1": ["g1:a", "g1:b"]}
            action, audit = candidate_action(
                frozen_view(), 0, {f"g{index}" for index in range(5)}, pred, perf,
                {f"g{index}": 0 for index in range(5)}, compound_members=members)
            details["frozen_controlled_candidate"] = {"action": action, "audit": audit}
            check("frozen_predictor_compound_discriminability",
                  audit["candidate_discriminability"] == "distinguished"
                  and all(item["denominator"] == 5 and item["max_batch_slots"] <= 4
                          for item in audit["actions"]),
                  {"discriminability": audit["candidate_discriminability"],
                   "accepted": audit["accepted"], "fallback": audit["fallback"],
                   "scores": [{"key": item["key"], "h": item["h"],
                               "good": item["good"], "mean": item["mean"],
                               "differs": item.get("score_differs_from_baseline")}
                              for item in audit["actions"]]},
                  "same frozen predictor, distinguished multi-goal candidates; acceptance not required")

            _, unsupported = candidate_action(
                frozen_view(prompt=2933, ordinal=2), 0, {"g0"}, pred, perf,
                {f"g{index}": 0 for index in range(5)})
            check("unsupported_state_still_falls_back",
                  not unsupported["accepted"] and "support" in str(unsupported["fallback"]),
                  {"accepted": unsupported["accepted"], "fallback": unsupported["fallback"]},
                  "explicit prediction-support fallback")

            engine = run_compound_engine(root, perf, pred, attempt)
            details["compound_engine"] = engine
            first = engine["first_candidate"]
            check("experimental_compound_path_and_slot_config",
                  first["audit"].get("evaluation_version") == "lease-slots-v1"
                  and first["audit"].get("batch_slots") == 4
                  and engine["policy"]["mechanism_version"] == "v2_compound_stage_eval_v1"
                  and "stage4_p0_evaluation" in engine["candidate_path"]
                  and "stage4_p0_evaluation" in engine["policy_path"],
                  {"evaluation_version": first["audit"].get("evaluation_version"),
                   "batch_slots": first["audit"].get("batch_slots"),
                   "mechanism_version": engine["policy"]["mechanism_version"],
                   "candidate_path": engine["candidate_path"],
                   "policy_path": engine["policy_path"]},
                  "new evaluator only in the new compound evaluation path")
            check("engine_compound_discriminability",
                  first["audit"].get("candidate_discriminability") == "distinguished",
                  {"discriminability": first["audit"].get("candidate_discriminability"),
                   "fallback": first["audit"].get("fallback"),
                   "accepted": first["audit"].get("accepted")},
                  "five real request slots in four compound goals produce distinguishable scores")
            check("engine_compound_execution_complete_and_atomic",
                  not engine["pending"] and not engine["unfinished"]
                  and not engine["partial_compound_batches"]
                  and not engine["over_slot_batches"] and engine["finish_count"] == 5,
                  {"pending": engine["pending"], "unfinished": engine["unfinished"],
                   "partial": engine["partial_compound_batches"],
                   "over_slot": engine["over_slot_batches"],
                   "finish_count": engine["finish_count"], "steps": engine["steps"]},
                  "five calls, no same-chain hold, no split group, at most four requests per batch")

            after = {path.name: sha(path) for path in (predictor_path, performance_path)}
            upstream_bad = [name for name, expected in source["upstream"].items()
                            if sha(root / name) != expected]
            check("frozen_assets_and_upstream_sources_unchanged",
                  before == source["frozen_assets"] == after and not upstream_bad,
                  {"before": before, "after": after, "upstream_bad": upstream_bad},
                  "predictor, calibration and all upstream source hashes unchanged")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=8)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "all controlled checks completed")

    passed = all(item["pass"] for item in checks.values())
    gate = ("REFERENCE_ONLY_PASS" if reference_only else "PASS") if passed else "FAIL"
    result = {
        "schema": "p0-evaluation-gate/v1",
        "attempt": attempt,
        "gate": gate,
        "checks": checks,
        "details": details,
        "scope": "Deterministic controlled evaluation only; not a GPU, public-trace performance, DEV, HOLDOUT or mechanism-benefit result",
        "predictor_changed": False,
        "formal_matrix_allowed": False,
        "holdout_allowed": False,
    }
    dump_new(receipt, result)
    print(json.dumps({"gate": gate,
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
