import asyncio
import copy
import functools
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


def main():
    root = Path(sys.argv[1]).resolve()
    arm = sys.argv[2]
    output = Path(sys.argv[3]).resolve()
    freeze = json.loads((root / "stage3/BACKEND_FREEZE.json").read_text())
    prepare = json.loads((root / "PREPARE.json").read_text())
    repo = Path(prepare["repo"])
    sys.path[:0] = [
        str(root / "stage3/programs"),
        str(root / "stage3/backend_src"),
        str(repo / "benchmark"),
        str(repo),
        str(repo / "third_party/vllm"),
        prepare["sklearn_deps"],
        prepare["support_deps"],
    ]
    from scoped_cuda_platform import cuda_driver_uuid
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2" or cuda_driver_uuid() != freeze["gpu_uuid"]:
        raise RuntimeError("Backend calibration requires logical CUDA0 on physical GPU2")
    import torch
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.usage.usage_lib import UsageContext
    from trace.trace_deepresearch import DeepResearchTrace
    from jitserve.request_info import RequestType as OfficialRequestType

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible GPU is required")
    if arm == "B":
        from harness_executor import HarnessAsyncExecutor, HarnessAsyncLLMEngine
        HarnessAsyncExecutor.configure(output)
        engine_class = HarnessAsyncLLMEngine
    elif arm == "A":
        engine_class = AsyncLLMEngine
    else:
        raise ValueError("Only A or B is supported")

    execution = freeze["execution"]
    engine_args = AsyncEngineArgs(
        model=freeze["model"],
        tokenizer=freeze["model"],
        dtype=freeze["dtype"],
        scheduling_policy=execution["scheduling_policy"],
        max_num_seqs=execution["max_num_seqs"],
        max_num_batched_tokens=execution["max_num_batched_tokens"],
        max_model_len=execution["max_model_len"],
        num_gpu_blocks_override=execution["num_gpu_blocks_override"],
        swap_space=execution["swap_space_gib"],
        enable_chunked_prefill=execution["enable_chunked_prefill"],
        enable_prefix_caching=execution["enable_prefix_caching"],
        enforce_eager=execution["enforce_eager"],
        disable_async_output_proc=execution["disable_async_output_proc"],
        disable_log_stats=True,
        seed=execution["seed"],
    )
    engine = None
    execute_records = []
    try:
        init_start = time.perf_counter()
        engine = engine_class.from_engine_args(
            engine_args, start_engine_loop=False, usage_context=UsageContext.API_SERVER)
        init_seconds = time.perf_counter() - init_start
        original_execute = engine.engine.model_executor.execute_model

        @functools.wraps(original_execute)
        def timed_execute(execute_model_req):
            started = time.perf_counter()
            result = original_execute(execute_model_req=execute_model_req)
            execute_records.append({
                "seconds": time.perf_counter() - started,
                "groups": [
                    {
                        "request_id": metadata.request_id,
                        "token_chunk_size": metadata.token_chunk_size,
                        "is_prompt": metadata.is_prompt,
                        "do_sample": metadata.do_sample,
                    }
                    for metadata in execute_model_req.seq_group_metadata_list
                ],
            })
            return result

        engine.engine.model_executor.execute_model = timed_execute
        collective = DeepResearchTrace.load_trace(str(root / freeze["input"]["trace"]))[0]
        infos = collective.to_request_infos()
        by_stage_request = {(info.stage_id, info.request_id): info for info in infos}

        def run_case(spec, warmup):
            case_name = spec.get("name", "warmup")
            output_tokens = spec.get("output_tokens_per_request", spec.get("output_tokens"))
            request_ids = spec.get("request_indices_within_stage", [spec.get("request_index_within_stage", 0)])
            for request_index in request_ids:
                info = copy.deepcopy(by_stage_request[(spec["stage"], request_index)])
                info.request_type = OfficialRequestType.LATENCY
                info.output_len = output_tokens
                info.deadline = 120.0
                request_id = f"{'warmup' if warmup else 'main'}-{case_name}-{request_index}"
                params = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=1,
                    max_tokens=output_tokens,
                    ignore_eos=True,
                    seed=execution["seed"],
                )
                asyncio.run(engine.engine.add_request_async(
                    request_id,
                    prompt=info.prompt,
                    params=params,
                    request_info=info,
                    arrival_time=100.0,
                ))
            execute_start = len(execute_records)
            started = time.perf_counter()
            steps = 0
            final = {}
            while engine.engine.has_unfinished_requests():
                outputs = engine.engine.step()
                steps += 1
                for item in outputs:
                    if item.finished:
                        final[item.request_id] = item
                if steps > 1000:
                    raise RuntimeError("Case exceeded frozen step bound")
            wall = time.perf_counter() - started
            for _ in range(2):
                engine.engine.step()
            outputs = []
            for request_id, item in sorted(final.items()):
                if len(item.outputs) != 1:
                    raise RuntimeError("Unexpected output fanout")
                token_ids = list(item.outputs[0].token_ids)
                if len(token_ids) != output_tokens:
                    raise RuntimeError(f"Output length mismatch for {request_id}: {len(token_ids)} != {output_tokens}")
                outputs.append({
                    "request_id": request_id,
                    "token_ids": token_ids,
                    "output_tokens": len(token_ids),
                    "finish_reason": item.outputs[0].finish_reason,
                    "stop_reason": item.outputs[0].stop_reason,
                })
            return {
                "name": case_name,
                "wall_seconds": wall,
                "engine_steps": steps,
                "execute_model_seconds": sum(record["seconds"] for record in execute_records[execute_start:]),
                "execute_records": execute_records[execute_start:],
                "outputs": outputs,
                "warmup": warmup,
            }

        warmup = run_case(freeze["input"]["warmup"], True)
        cases = [run_case(spec, False) for spec in freeze["input"]["cases"]]
        result = {
            "complete": True,
            "arm": arm,
            "model_init_seconds": init_seconds,
            "warmup": warmup,
            "cases": cases,
            "gpu_uuid": freeze["gpu_uuid"],
            "engine_class": engine_class.__name__,
            "scheduler_class": type(engine.engine.scheduler[0]).__name__,
            "scope": "Fixed-action fixed-output backend calibration; no policy or mechanism claim",
        }
        dump(output / "backend_results.json", result)
    except BaseException as error:
        dump(output / "BACKEND_FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if engine is not None:
            engine.shutdown_background_loop()
            if arm == "B":
                engine.engine.model_executor.finalize()
            else:
                engine.engine.model_executor.shutdown()


if __name__ == "__main__":
    main()
