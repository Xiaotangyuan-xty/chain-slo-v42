import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def request_summary(output):
    return {
        "collection_id": output.collection_id,
        "stage_id": output.stage_id,
        "request_id": output.request_id,
        "state": output.state,
        "success": output.success,
        "error": output.error,
        "stop_reason": output.stop_reason,
        "finish_before_ddl": output.finish_before_ddl,
        "request_ttft": output.request_ttft,
        "request_tbt": output.request_tbt,
        "request_service_gain": output.request_service_gain,
        "request_latency": output.request_latency,
        "request_finish_time": output.request_finish_time,
    }


def collective_summary(output, phase):
    return {
        "phase": phase,
        "collection_id": output.collection_id,
        "success": output.success,
        "total_latency": output.total_latency,
        "stages": [
            {
                "stage_id": stage.stage_id,
                "success": stage.success,
                "stage_latency": stage.stage_latency,
                "requests": [request_summary(request) for request in stage.stage_outputs],
            }
            for stage in output.stage_outputs
        ],
    }


def main():
    root = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    smoke = json.loads((root / "stage6_mechanism_repair/SMOKE_FREEZE.json").read_text())
    prepare = json.loads((root / "PREPARE.json").read_text())
    repo = Path(prepare["repo"])
    sys.path[:0] = [
        str(repo / "benchmark"),
        str(repo),
        str(repo / "third_party/vllm"),
        prepare["sklearn_deps"],
        prepare["support_deps"],
    ]
    from trace import client_deepresearch as client_module
    import benchmark_scheduler_deepresearch as benchmark_module

    calls = []
    arrival = {"origin": None}
    original_send = client_module.send_deepresearch_collective_request

    async def audited_send(*args, **kwargs):
        phase = "warmup" if not calls else "main"
        started = time.perf_counter()
        try:
            result = await original_send(*args, **kwargs)
            summary = collective_summary(result, phase)
            summary["arrival_offset_seconds"] = (
                None if arrival["origin"] is None else started - arrival["origin"])
            calls.append(summary)
        except BaseException as error:
            calls.append({"phase": phase, "exception": repr(error), "traceback": traceback.format_exc()})
            raise
        dump(output_dir / "client_calls.json", calls)
        return result

    client_module.send_deepresearch_collective_request = audited_send
    original_simulator = benchmark_module.deepresearch_client_simulator

    async def audited_simulator(*args, **kwargs):
        arrival["origin"] = time.perf_counter()
        return await original_simulator(*args, **kwargs)

    benchmark_module.deepresearch_client_simulator = audited_simulator
    original_metrics = benchmark_module.calculate_deepresearch_metrics

    def audited_metrics(collective_outputs, *args, **kwargs):
        metrics = original_metrics(collective_outputs, *args, **kwargs)
        dump(output_dir / "official_metrics.json", asdict(metrics))
        return metrics

    benchmark_module.calculate_deepresearch_metrics = audited_metrics
    resources = smoke["resources"]
    generation = smoke["generation_and_deadline"]
    args = argparse.Namespace(
        base_url=None,
        host="127.0.0.1",
        port=18080,
        endpoint="/generate",
        trace_path=str(root / smoke["input"]["trace"]),
        num_prompts=smoke["selection"]["main_chain_count"],
        arrival_rate=generation["arrival_rate"],
        burst=generation["burst"],
        slo_constraint=",".join(str(value) for value in generation["slo_constraint_seconds"]),
        penalty_factor=resources["penalty_factor"],
        max_concurrency=None,
        policy="jitserve",
        model=prepare["model"],
        tokenizer=None,
        n=1,
        best_of=1,
        logprobs=None,
        max_output_len=generation["max_output_len"],
        seed=generation["seed"],
        trust_remote_code=False,
        disable_tqdm=True,
        ignore_eos=generation["ignore_eos"],
        metric_percentiles="50,90,95,99",
        show_id_length=False,
        is_stream=generation["is_stream"],
    )
    try:
        benchmark_module.main(args)
        dump(output_dir / "CLIENT_DONE.json", {"complete": True, "calls": len(calls), "wall": time.time()})
    except BaseException as error:
        dump(output_dir / "CLIENT_FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
