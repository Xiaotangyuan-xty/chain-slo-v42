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
    freeze = json.loads((root / "FREEZE.json").read_text())
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
    original_send = client_module.send_deepresearch_collective_request

    async def audited_send(*args, **kwargs):
        phase = "warmup" if not calls else "main"
        try:
            result = await original_send(*args, **kwargs)
            calls.append(collective_summary(result, phase))
        except BaseException as error:
            calls.append({"phase": phase, "exception": repr(error), "traceback": traceback.format_exc()})
            raise
        dump(output_dir / "client_calls.json", calls)
        return result

    client_module.send_deepresearch_collective_request = audited_send
    original_metrics = benchmark_module.calculate_deepresearch_metrics

    def audited_metrics(collective_outputs, *args, **kwargs):
        metrics = original_metrics(collective_outputs, *args, **kwargs)
        dump(output_dir / "official_metrics.json", asdict(metrics))
        return metrics

    benchmark_module.calculate_deepresearch_metrics = audited_metrics
    server = freeze["server"]
    client = freeze["client"]
    model = freeze["model"]
    args = argparse.Namespace(
        base_url=None,
        host=server["host"],
        port=server["port"],
        endpoint="/generate",
        trace_path=str(root / freeze["input"]["public_trace"]),
        num_prompts=1,
        arrival_rate=client["arrival_rate"],
        burst=client["burst"],
        slo_constraint=",".join(str(value) for value in freeze["deadline"]["slo_constraint_seconds"]),
        penalty_factor=server["penalty_factor"],
        max_concurrency=None,
        policy=server["scheduling_policy"],
        model=prepare["model"],
        tokenizer=None,
        n=1,
        best_of=1,
        logprobs=None,
        max_output_len=model["max_output_len"],
        seed=client["seed"],
        trust_remote_code=False,
        disable_tqdm=True,
        ignore_eos=client["ignore_eos"],
        metric_percentiles="50,90,95,99",
        show_id_length=False,
        is_stream=client["is_stream"],
    )
    try:
        benchmark_module.main(args)
        dump(output_dir / "CLIENT_DONE.json", {"complete": True, "calls": len(calls), "wall": time.time()})
    except BaseException as error:
        dump(output_dir / "CLIENT_FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
