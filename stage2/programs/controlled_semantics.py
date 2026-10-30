import asyncio
import copy
import json
import os
from pathlib import Path
import sys

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

from fake_backend import FakeBackend
from harness_executor import HarnessAsyncExecutor, HarnessAsyncLLMEngine


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def engine_args(prepare, **overrides):
    values = dict(
        model=prepare["model"],
        tokenizer=prepare["model"],
        device="cpu",
        dtype="bfloat16",
        scheduling_policy="jitserve",
        penalty_factor=100,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        max_model_len=8192,
        num_gpu_blocks_override=512,
        swap_space=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_log_stats=True,
        disable_async_output_proc=True,
        seed=42,
    )
    values.update(overrides)
    return AsyncEngineArgs(**values)


def make_engine(kind, args, output):
    HarnessAsyncExecutor.configure(output, FakeBackend)
    if kind == "A":
        config = args.create_engine_config()
        config.parallel_config.distributed_executor_backend = HarnessAsyncExecutor
        return AsyncLLMEngine.from_engine_args(
            args, engine_config=config, start_engine_loop=False)
    return HarnessAsyncLLMEngine.from_engine_args(args, start_engine_loop=False)


def output_summary(outputs):
    return [
        {
            "request_id": output.request_id,
            "finished": output.finished,
            "token_ids": [list(item.token_ids) for item in output.outputs],
            "finish_reason": [item.finish_reason for item in output.outputs],
            "stop_reason": [item.stop_reason for item in output.outputs],
        }
        for output in outputs
    ]


def semantic_events(path):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    result = []
    for event in events:
        if event["kind"] == "execute_model":
            result.append({
                "kind": event["kind"],
                "groups": event["groups"],
                "sampled": event["sampled"],
            })
        elif event["kind"] in {"official_swap_out_block", "official_swap_in_blocks", "backend_swap_out", "backend_release"}:
            result.append({key: value for key, value in event.items() if key not in {"n", "wall"}})
    return result


def backend_semantics(summary):
    return {key: value for key, value in summary.items() if not key.endswith("_seconds")}


def run_pair(name, args, requests, output_root):
    engines = {}
    snapshots = {}
    steps = []
    for kind in ["A", "B"]:
        output = output_root / name / kind
        output.mkdir(parents=True, exist_ok=False)
        engines[kind] = make_engine(kind, args, output)
    for request in requests:
        for kind, engine in engines.items():
            info = copy.deepcopy(request["request_info"])
            asyncio.run(engine.engine.add_request_async(
                request["request_id"],
                prompt=request["prompt"],
                request_info=info,
                params=copy.deepcopy(request["sampling_params"]),
                arrival_time=100.0,
            ))
            if len(snapshots) < 2 * len(requests):
                group = engine.engine.scheduler[0].waiting[-1]
                group.request_info.output_len = request["prediction"]
                snapshots[f"{kind}:{request['request_id']}"] = {
                    "snapshot": group.predict_output_length,
                    "cache_after_ready": group.request_info.output_len,
                    "request_info_identity": id(group.request_info) == id(info),
                }
    for _ in range(200):
        row = {}
        for kind, engine in engines.items():
            row[kind] = output_summary(engine.engine.step())
        steps.append(row)
        if not any(engine.engine.has_unfinished_requests() for engine in engines.values()):
            break
    for _ in range(2):
        for engine in engines.values():
            engine.engine.step()
    for engine in engines.values():
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()
    events = {
        kind: semantic_events(output_root / name / kind / "executor_events.jsonl")
        for kind in engines
    }
    backend_summaries = {
        kind: backend_semantics(json.loads((output_root / name / kind / "executor_summary.json").read_text())["backend"])
        for kind in engines
    }
    return {
        "snapshots": snapshots,
        "steps_A": [row["A"] for row in steps],
        "steps_B": [row["B"] for row in steps],
        "events_A": events["A"],
        "events_B": events["B"],
        "backend_A": backend_summaries["A"],
        "backend_B": backend_summaries["B"],
        "scheduler_types": {kind: type(engine.engine.scheduler[0]).__name__ for kind, engine in engines.items()},
        "steps_equal": [row["A"] for row in steps] == [row["B"] for row in steps],
        "events_equal": events["A"] == events["B"],
        "backend_equal": backend_summaries["A"] == backend_summaries["B"],
    }


def main():
    root = Path(sys.argv[1]).resolve()
    attempt_name = sys.argv[2]
    output_root = root / "stage2/controlled" / attempt_name
    output_root.mkdir(parents=True, exist_ok=False)
    prepare = json.loads((root / "PREPARE.json").read_text())
    freeze = json.loads((root / "stage2/STAGE2_FREEZE.json").read_text())
    checks = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    check("cpu_only", os.environ.get("CUDA_VISIBLE_DEVICES") == "", os.environ.get("CUDA_VISIBLE_DEVICES"), "")

    from trace.trace_deepresearch import DeepResearchTrace
    from jitserve.request_analyzer import prediction
    from jitserve.request_analyzer.deepresearch_trace_reader import read_deepresearch_traces
    from jitserve.request_analyzer.graph_context import GraphMatchContext
    from jitserve.scheduler.slo_scheduler import SLOScheduler
    from jitserve.request_info import RequestType as OfficialRequestType

    collective = DeepResearchTrace.load_trace(str(root / freeze["public_input"]["trace"]))[0]
    infos = collective.to_request_infos()
    model, vectorizer = prediction.load_model(prepare["qrf_model"], prepare["qrf_vectorizer"])
    prediction_values = {}
    for stage in [0, 1]:
        prompt = infos[stage].prompt
        first = prediction.predict(vectorizer, model, [prompt])[0]
        second = prediction.predict(vectorizer, model, [prompt])[0]
        prediction_values[str(stage)] = {"A": first, "B": second}
        check(f"prediction_stage_{stage}", first == second and isinstance(first, int) and first > 0,
              prediction_values[str(stage)], "exact positive integer equality")

    tokenizer = get_tokenizer(prepare["model"])
    graphs = read_deepresearch_traces(str(root / "repo/traces/deepresearch_training.json"), False)
    shares = {}
    subdeadlines = {}
    for kind in ["A", "B"]:
        context = GraphMatchContext(
            generation_tokenizer=tokenizer,
            use_all_node=False,
            stage_ratio_method="execution_time",
            graph_structure_type="deepresearch",
            graph_matching_mode="online",
            use_total_deadline=False,
        )
        context.add_deepresearch_base_graphs(copy.deepcopy(graphs))
        context.init_collection(0, True, 6, [1, 2, 1, 1, 1, 1])
        shares[kind] = []
        subdeadlines[kind] = []
        for stage in [0, 1]:
            share = context.calculate_stage_ratio(
                6, [1, 2, 1, 1, 1, 1], infos[stage].prompt, 0,
                is_deepresearch=True, stage_id=stage, accumulate_stage_ratio=0.0)
            shares[kind].append(share)
            subdeadlines[kind].append(120.0 * share - 5.0)
    check("stage_share_exact", shares["A"] == shares["B"], shares, "exact A/B equality")
    check("subdeadline_exact", all(abs(a - b) <= 1e-9 for a, b in zip(subdeadlines["A"], subdeadlines["B"])),
          subdeadlines, "absolute error <= 1e-9")
    check("official_base_graph_count", len(graphs) == 394, len(graphs), 394)

    requests = []
    for index, info in enumerate(infos[:4]):
        request_info = copy.deepcopy(info)
        request_info.request_type = OfficialRequestType.COLLECTIVE
        request_info.collection_id = index
        request_info.deadline = 120.0
        request_info.output_len = 1024
        request_info.slo_constraint = (2.0, 0.1, 20.0)
        requests.append({
            "request_id": f"main-{index}",
            "prompt": info.prompt,
            "request_info": request_info,
            "prediction": prediction_values["0" if index == 0 else "1"]["A"],
            "sampling_params": SamplingParams(temperature=0.0, top_p=1.0, top_k=1,
                                               max_tokens=4, seed=42, ignore_eos=False),
        })
    main_result = run_pair("main", engine_args(prepare), requests, output_root)
    check("ordered_frame_members_exact", main_result["events_equal"],
          {"A": main_result["events_A"], "B": main_result["events_B"]}, "exact ordered request IDs and chunks")
    check("snapshots_exact_and_not_refreshed", all(
        item["snapshot"] == 1024 and item["cache_after_ready"] != 1024 and item["request_info_identity"]
        for item in main_result["snapshots"].values()), main_result["snapshots"],
        "all initial snapshots remain 1024 while request_info cache updates")
    check("admission_finish_and_output_exact", main_result["steps_equal"],
          {"A": main_result["steps_A"], "B": main_result["steps_B"]}, "exact admission, output and finish sequence")
    check("single_engine_finish_kv_release", main_result["backend_equal"] and all(
        entry["gpu"] == 0 and entry["host"] == 0
        for entry in main_result["backend_A"]["ledger"] if entry["key"].startswith("main-")),
          {"A": main_result["backend_A"], "B": main_result["backend_B"]},
          "identical single-engine finish and zero-residency released KV")

    preempt_requests = []
    prompt_tokens = list(tokenizer.encode(infos[0].prompt, add_special_tokens=True))[:16]
    for index in range(2):
        request_info = copy.deepcopy(infos[0])
        request_info.request_type = OfficialRequestType.COLLECTIVE
        request_info.collection_id = 100 + index
        request_info.deadline = 120.0
        request_info.output_len = 1024
        request_info.slo_constraint = (2.0, 0.1, 20.0)
        preempt_requests.append({
            "request_id": f"preempt-{index}",
            "prompt": TokensPrompt(prompt_token_ids=prompt_tokens),
            "request_info": request_info,
            "prediction": 1024,
            "sampling_params": SamplingParams(temperature=0.0, top_p=1.0, top_k=1,
                                               max_tokens=4, seed=42, ignore_eos=False),
        })
    preempt_args = engine_args(
        prepare,
        max_num_seqs=2,
        max_num_batched_tokens=16,
        max_model_len=32,
        num_gpu_blocks_override=2,
    )
    preempt_result = run_pair("preemption", preempt_args, preempt_requests, output_root)
    preempt_events = [event for event in preempt_result["events_A"]
                      if event["kind"] in {"official_swap_out_block", "official_swap_in_blocks"}]
    check("preemption_swap_observed", any(event["kind"] == "official_swap_out_block" for event in preempt_events)
          and any(event["kind"] == "official_swap_in_blocks" for event in preempt_events),
          preempt_events, "at least one swap-out and swap-in")
    check("preemption_transitions_exact", preempt_result["events_equal"] and preempt_result["steps_equal"],
          {"events_A": preempt_result["events_A"], "events_B": preempt_result["events_B"],
           "steps_A": preempt_result["steps_A"], "steps_B": preempt_result["steps_B"]},
          "exact queues inferred by ordered execution, preemption blocks and finish")
    check("preemption_kv_accounting_exact", preempt_result["backend_equal"],
          {"A": preempt_result["backend_A"], "B": preempt_result["backend_B"]},
          "identical GPU/host/release ledger and no silent eviction")
    check("official_scheduler_identity", main_result["scheduler_types"] == {"A": "SLOScheduler", "B": "SLOScheduler"}
          and preempt_result["scheduler_types"] == {"A": "SLOScheduler", "B": "SLOScheduler"},
          {"main": main_result["scheduler_types"], "preemption": preempt_result["scheduler_types"]},
          f"both use {SLOScheduler.__module__}.{SLOScheduler.__name__}")

    result = {
        "attempt": attempt_name,
        "gate": "PASS" if all(item["pass"] for item in checks.values()) else "FAIL",
        "cpu_only": True,
        "checks": checks,
        "prediction_values": prediction_values,
        "shares": shares,
        "subdeadlines": subdeadlines,
        "main_frame_count": sum(event["kind"] == "execute_model" for event in main_result["events_A"]),
        "preemption_events": preempt_events,
        "scope": "Controlled official-lifecycle comparison with deterministic fake backend plus real CPU QRF/graph assets; no GPU or end-to-end performance claim.",
    }
    dump(root / "stage2/receipts" / f"{attempt_name}.json", result)
    if result["gate"] == "PASS":
        dump(root / "stage2/receipts/S0_S1_CONTROLLED_PASS.json", result)
    print(json.dumps({"gate": result["gate"], "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(root / "stage2/receipts" / f"{attempt_name}.json")}, indent=2))
    return 0 if result["gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
