import json
import os
from pathlib import Path
import sys
import traceback

from controlled_kv_atomic import long_predictor
from controlled_scheduler import add, make_engine, public_infos, selector
from cap4_slo_scheduler import Cap4SLOScheduler


def events(output):
    result = []
    for line in (output / "executor_events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["kind"] == "execute_model":
            result.append([[group["request_id"], group["token_chunk_size"]]
                           for group in event["groups"]])
    return result


def run_engine(root, output, native):
    infos = public_infos(root)
    engine, scheduler = make_engine(root, output, {index: 1 for index in range(5)},
                                    "0:stage:0", 50)
    if native:
        old = scheduler
        scheduler = Cap4SLOScheduler(old.scheduler_config, old.cache_config,
                                     old.lora_config, batch_slots=4)
        scheduler.recompute_invalidate = engine.engine.model_executor.invalidate_request
        engine.engine.scheduler[0] = scheduler
    else:
        scheduler.policy.candidate_selector = None
    try:
        for index in range(5):
            add(engine, infos, f"continuous-{index}", index, 0,
                384 if index == 4 else 256, slo=120.0)
            group = scheduler.policy.seq_group_dict[index][0]
            group.sampling_params.max_tokens = 64
            group.request_info.output_len = 64
            group.predict_output_length = 64
        steps = 0
        while engine.engine.has_unfinished_requests() and steps < 400:
            if not native:
                now = __import__("time").perf_counter()
                for state in scheduler.policy.current.values():
                    state["ready_since"] = now
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        result = {"events": events(output), "steps": steps,
                  "unfinished": engine.engine.has_unfinished_requests()}
    finally:
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()
    result["summary"] = json.loads((output / "executor_summary.json").read_text())
    return result


def run_model_prefill(root, output):
    infos = public_infos(root)
    engine, scheduler = make_engine(root, output, {0: 1}, "0:stage:0", 50,
                                    pred=long_predictor())
    scheduler.policy.candidate_selector = selector("0:stage:0")
    try:
        add(engine, infos, "model-long", 0, 0, 2933, slo=120.0)
        engine.engine.step()
        engine.engine.step()
        batches = events(output)
        steps = 2
        while engine.engine.has_unfinished_requests() and steps < 100:
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        result = {"batches": batches, "steps": steps,
                  "unfinished": engine.engine.has_unfinished_requests()}
    finally:
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()
    result["summary"] = json.loads((output / "executor_summary.json").read_text())
    return result


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only continuous-prefill gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        native = run_engine(root, root / "stage6_mechanism_repair/controlled" / attempt / "native", True)
        fallback = run_engine(root, root / "stage6_mechanism_repair/controlled" / attempt / "fallback", False)
        details["native"] = native
        details["fallback"] = fallback
        check("no_model_action_matches_native_nonframe_path",
              fallback["events"] == native["events"] and not fallback["unfinished"]
              and not native["unfinished"],
              {"native_steps": native["steps"], "fallback_steps": fallback["steps"],
               "native_first": native["events"][:4], "fallback_first": fallback["events"][:4]},
              "same members, token chunks and completion on the no-model-action path")
        check("fallback_releases_kv",
              native["summary"]["tracked_block_owners"] == 0
              and fallback["summary"]["tracked_block_owners"] == 0,
              {"native": native["summary"]["tracked_block_owners"],
               "fallback": fallback["summary"]["tracked_block_owners"]}, 0)

        model = run_model_prefill(root, root / "stage6_mechanism_repair/controlled" / attempt / "model")
        details["model"] = model
        first_chunks = [group[1] for batch in model["batches"][:2] for group in batch]
        check("model_long_prefill_uses_real_budget",
              first_chunks == [2048, 885],
              {"first_chunks": first_chunks, "first_batches": model["batches"][:2]},
              "2933-token prefill completes in chunks of 2048 and 885, not repeated two-token steps")
        check("model_long_prefill_completes_and_releases",
              not model["unfinished"] and model["steps"] < 100
              and model["summary"]["tracked_block_owners"] == 0,
              {"unfinished": model["unfinished"], "steps": model["steps"],
               "owners": model["summary"]["tracked_block_owners"]},
              "long request completes without deadlock or KV leaks")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "continuous-prefill gate completed")

    passed = all(item["pass"] for item in checks.values())
    result = {"schema": "stage10-continuous-prefill-gate/v1", "attempt": attempt,
              "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
              "gpu_used": False}
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_CONTINUOUS_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
