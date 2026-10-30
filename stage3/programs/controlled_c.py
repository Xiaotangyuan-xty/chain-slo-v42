import asyncio
import copy
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

from fake_backend import FakeBackend
from harness_executor import HarnessAsyncExecutor, HarnessAsyncLLMEngine
from pool_engine import candidate_action
from pool_policies import Performance
from v2_policy import install_v2
from v2_predictor import Predictor


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    output = root / "stage3/controlled" / attempt
    output.mkdir(parents=True, exist_ok=False)
    prepare = json.loads((root / "PREPARE.json").read_text())
    asset_root = Path("/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_finite_h4_confirm_gpu01_20261018T114337Z_resume_v1")
    perf = Performance(json.loads((asset_root / "PERFORMANCE.json").read_text()))
    pred = Predictor.from_dict(json.loads((asset_root / "PREDICTOR.json").read_text()))
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
    scheduler = install_v2(engine, perf, pred)
    original_execute = engine.engine.model_executor.execute_model

    @functools.wraps(original_execute)
    def timed_execute(execute_model_req):
        started = time.perf_counter()
        result = original_execute(execute_model_req=execute_model_req)
        scheduler.policy.after_model_step(time.perf_counter() - started, time.perf_counter())
        return result

    engine.engine.model_executor.execute_model = timed_execute
    from trace.trace_deepresearch import DeepResearchTrace
    from jitserve.request_info import RequestType as OfficialRequestType
    infos = DeepResearchTrace.load_trace(str(root / "repo/traces/deepresearch_filter.jsonl"))[0].to_request_infos()
    selected = [infos[0], infos[1], infos[2], infos[3]]
    collections = [0, 1, 1, 2]
    for index, (info, collection) in enumerate(zip(selected, collections)):
        request_info = copy.deepcopy(info)
        request_info.request_type = OfficialRequestType.COLLECTIVE
        request_info.collection_id = collection
        request_info.deadline = 120.0
        request_info.output_len = 1024
        asyncio.run(engine.engine.add_request_async(
            f"c-{index}", prompt=info.prompt,
            params=SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens=4, seed=42),
            request_info=request_info, arrival_time=100.0,
        ))
    held_initially = [group.request_id for group in scheduler.pending]
    unfinished_with_hold = engine.engine.has_unfinished_requests()
    steps = 0
    while engine.engine.has_unfinished_requests() and steps < 100:
        engine.engine.step()
        steps += 1
    for _ in range(2):
        engine.engine.step()
    reset_info = copy.deepcopy(selected[0])
    reset_info.request_type = OfficialRequestType.COLLECTIVE
    reset_info.collection_id = 0
    reset_info.deadline = 120.0
    reset_info.output_len = 1024
    asyncio.run(engine.engine.add_request_async(
        "c-reset", prompt=selected[0].prompt,
        params=SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens=4, seed=42),
        request_info=reset_info, arrival_time=100.0,
    ))
    while engine.engine.has_unfinished_requests() and steps < 100:
        engine.engine.step()
        steps += 1
    for _ in range(2):
        engine.engine.step()
    engine.shutdown_background_loop()
    engine.engine.model_executor.finalize()
    events = [json.loads(line) for line in (output / "executor_events.jsonl").read_text().splitlines()]
    policy = scheduler.policy.snapshot()
    checks = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    check("cpu_only", os.environ.get("CUDA_VISIBLE_DEVICES") == "", os.environ.get("CUDA_VISIBLE_DEVICES"), "")
    check("same_collection_held", held_initially == ["c-2"] and unfinished_with_hold,
          {"pending": held_initially, "unfinished": unfinished_with_hold}, "c-2 held and still counted unfinished")
    batches = [[group["request_id"] for group in event["groups"]]
               for event in events if event["kind"] == "execute_model"]
    duplicate = []
    collection_by_id = {"c-0": 0, "c-1": 1, "c-2": 1, "c-3": 2, "c-reset": 0}
    for batch in batches:
        values = [collection_by_id[request_id] for request_id in batch]
        if len(values) != len(set(values)):
            duplicate.append(batch)
    check("same_collection_serialization", not duplicate, duplicate, [])
    candidate_events = [event for event in policy["events"] if event["kind"] == "v2_candidate"]
    first = candidate_events[0]
    expected_action, expected_audit = candidate_action(
        scheduler.policy.first_view, 0, scheduler.policy.first_legal, pred, perf,
        {collection: 0 for collection in collections})
    check("frozen_candidate_exact", first["anchor"] == expected_action[0]
          and first["remaining"] == expected_action[1]
          and first["audit"]["actions"] == expected_audit["actions"],
          {"actual_anchor": first["anchor"], "expected_anchor": expected_action[0],
           "actual_h": first["remaining"], "expected_h": expected_action[1],
           "score_actions_equal": first["audit"]["actions"] == expected_audit["actions"]},
          "same anchor, h and every candidate score")
    check("frozen_source_identity", hashlib.sha256((root / "stage3/v2_src/pool_engine.py").read_bytes()).hexdigest()
          == hashlib.sha256((asset_root / "src/pool_engine.py").read_bytes()).hexdigest()
          and inspect.getfile(candidate_action).startswith(str(root / "stage3/v2_src")),
          inspect.getfile(candidate_action), "byte-identical frozen pool_engine.py loaded from stage3/v2_src")
    check("same_collection_release_recorded", any(event["kind"] == "v2_hold_same_collection" for event in policy["events"])
          and any(event["kind"] == "v2_release_same_collection" for event in policy["events"]),
          [event["kind"] for event in policy["events"]], "hold and later release")
    check("lease_and_fairness_updated", any(event["kind"] == "v2_after_step" and event["remaining"] < 4 * perf.decode_seconds for event in policy["events"]),
          {"decode_seconds": perf.decode_seconds, "events": [event for event in policy["events"] if event["kind"] == "v2_after_step"][:3]},
          "anchor lease decreases after actual selected execution")
    check("warmup_main_chain_reset", any(event["kind"] == "v2_chain_reset" and event["previous_history"] > 0 for event in policy["events"]),
          [event for event in policy["events"] if event["kind"] == "v2_chain_reset"],
          "new stage-0 resets prior chain history before the next execution")
    finish_events = [event for event in policy["events"] if event["kind"] == "v2_finish"]
    check("finish_history_and_no_drop", len(finish_events) == 5 and all(event["service"] > 0 and event["wait"] >= 0 for event in finish_events)
          and steps < 100 and not scheduler.pending,
          {"finish_events": finish_events, "steps": steps, "pending": len(scheduler.pending)},
          "all five calls finish with service/wait history and no pending")
    backend = json.loads((output / "executor_summary.json").read_text())["backend"]
    released = all(entry["gpu"] == 0 and entry["host"] == 0 for entry in backend["ledger"] if entry["key"].startswith("c-"))
    check("final_kv_release", released, backend["ledger"], "all controlled request KV released")
    result = {
        "attempt": attempt,
        "gate": "PASS" if all(item["pass"] for item in checks.values()) else "FAIL",
        "checks": checks,
        "first_candidate": first,
        "expected_candidate": {"action": expected_action, "audit": expected_audit},
        "policy": policy,
        "scope": "CPU-only C controlled semantic acceptance with the exact B executor and frozen v2 source/assets",
    }
    dump(root / "stage3/receipts" / f"{attempt}_C_GATE.json", result)
    if result["gate"] == "PASS":
        dump(root / "stage3/receipts/C_CONTROLLED_PASS.json", result)
    print(json.dumps({"gate": result["gate"], "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(root / "stage3/receipts" / f"{attempt}_C_GATE.json")}, indent=2))
    return 0 if result["gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
