import argparse
import asyncio
from collections import Counter
import functools
import inspect
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def main():
    root = Path(sys.argv[1]).resolve()
    stage = Path(sys.argv[2]).resolve()
    output_dir = Path(sys.argv[3]).resolve()
    sys.argv = [sys.argv[0]] + sys.argv[4:]
    prepare = json.loads((root / "PREPARE.json").read_text())
    authorization = json.loads((stage / "GPU1_AUTHORIZATION.json").read_text())
    repo = Path(prepare["repo"])
    sys.path[:0] = [
        str(root / "stage7_dynamic/programs"),
        str(root / "stage7_dynamic/backend_src"),
        str(root / "stage7_dynamic/v2_src"),
        str(root / "stage3_gpu0/programs"),
        str(root / "programs"),
        str(repo),
        str(repo / "third_party/vllm"),
        prepare["sklearn_deps"],
        prepare["support_deps"],
    ]
    from gpu1_platform import install
    device_receipt = install(root, stage, shared_backend=True)
    from scoped_cuda_platform import cuda_driver_uuid

    if os.environ.get("CUDA_VISIBLE_DEVICES") != authorization["gpu_uuid"] or cuda_driver_uuid() != authorization["gpu_uuid"]:
        raise RuntimeError("B requires logical CUDA0 on physical GPU1")
    import torch
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sequence import SequenceGroup
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils import FlexibleArgumentParser
    from jitserve import server
    from jitserve.request_analyzer.graph_context import GraphMatchContext
    from jitserve.scheduler.policy import SLOPolicy
    from jitserve.scheduler.slo_scheduler import SLOScheduler
    from harness_executor import HarnessAsyncExecutor, HarnessAsyncLLMEngine

    from abc_observation import install_observation
    observation = install_observation(server, AsyncLLMEngine, output_dir, "B", root, device_receipt)

    started = time.monotonic()
    lock = threading.Lock()
    events_path = output_dir / "server_events.jsonl"
    events_file = events_path.open("x")
    counts = Counter()
    ready_objects = set()
    scheduler_stats = Counter()

    def record(kind, **fields):
        with lock:
            event = {
                "n": counts["events"],
                "kind": kind,
                "monotonic": time.monotonic() - started,
                "wall": time.time(),
                "pid": os.getpid(),
                **fields,
            }
            counts["events"] += 1
            counts[kind] += 1
            events_file.write(json.dumps(event, allow_nan=False, default=str) + "\n")
            events_file.flush()

    def info_fields(info):
        return {
            "info_object": id(info),
            "collection_id": getattr(info, "collection_id", None),
            "stage_id": getattr(info, "stage_id", None),
            "request_id": getattr(info, "request_id", None),
            "cache_output_len": getattr(info, "output_len", None),
        }

    def wrap(obj, name, factory):
        original = getattr(obj, name)
        setattr(obj, name, functools.wraps(original)(factory(original)))

    def prediction_submit(original):
        def wrapped(info, prompt):
            record("prediction_submit", **info_fields(info))
            result = original(info, prompt)
            record("prediction_submit_return", **info_fields(info))
            return result
        return wrapped

    def prediction_ready(original):
        def wrapped(info, prompt):
            record("prediction_worker_begin", **info_fields(info))
            result = original(info, prompt)
            ready_objects.add(id(info))
            record("prediction_ready", **info_fields(info))
            return result
        return wrapped

    def assign(original):
        def wrapped(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            record(
                "sequence_group_assign",
                request_uuid=self.request_id,
                **info_fields(self.request_info),
                snapshot_output_len=self.predict_output_length,
                ready_before_assign=id(self.request_info) in ready_objects,
                arrival_time=self.arrival_time,
                deadline=self.deadline,
            )
            return result
        return wrapped

    def allocation(original):
        def wrapped(self, seq_group):
            peers = list(self.seq_group_dict[seq_group.collection_id])
            before = [
                {
                    "request_uuid": peer.request_id,
                    "stage_id": peer.request_info.stage_id,
                    "snapshot_output_len": peer.predict_output_length,
                    "cache_output_len": peer.request_info.output_len,
                    "decode_len": peer.first_seq.get_decode_len(),
                    "info_object": id(peer.request_info),
                }
                for peer in peers
            ]
            result = original(self, seq_group)
            record(
                "collective_allocation_read",
                request_uuid=seq_group.request_id,
                collection_id=seq_group.collection_id,
                peers=before,
                remaining_time=result,
                schedule_count=self.num_schedule_count,
            )
            return result
        return wrapped

    def graph_share(original):
        def wrapped(self, *args, **kwargs):
            started = time.perf_counter()
            result = original(self, *args, **kwargs)
            elapsed = time.perf_counter() - started
            scheduler_stats["graph_share_wall_seconds"] += elapsed
            numeric = isinstance(result, (int, float)) and math.isfinite(result)
            record(
                "graph_stage_share",
                collection_id=args[3] if len(args) > 3 else kwargs.get("collection_id"),
                stage_id=kwargs.get("stage_id", 0),
                share=result,
                finite=numeric,
                seconds=elapsed,
                base_graphs=len(self.deepresearch_base_graphs),
                clustering=self.dynamic_clustering.get_cluster_info(),
            )
            return result
        return wrapped

    def graph_completion(original):
        def wrapped(self, collection_id, is_deepresearch, request_input_length, request_output_length):
            before = self.dynamic_clustering.get_cluster_info()
            result = original(self, collection_id, is_deepresearch, request_input_length, request_output_length)
            after = self.dynamic_clustering.get_cluster_info()
            record(
                "graph_completion",
                collection_id=collection_id,
                is_deepresearch=is_deepresearch,
                request_input_length=request_input_length,
                request_output_length=request_output_length,
                finished_before=before["n_requests"],
                finished_after=after["n_requests"],
                clustering=after,
            )
            return result
        return wrapped

    def schedule(original):
        def wrapped(self, *args, **kwargs):
            queues_before = {
                "waiting": len(self.waiting),
                "running": len(self.running),
                "swapped": len(self.swapped),
            }
            started = time.perf_counter()
            result = original(self, *args, **kwargs)
            scheduler_stats["schedule_wall_seconds"] += time.perf_counter() - started
            _, outputs, _ = result
            scheduled = list(outputs.scheduled_seq_groups)
            items = [
                {
                    "request_uuid": item.seq_group.request_id,
                    "collection_id": item.seq_group.collection_id,
                    "stage_id": item.seq_group.request_info.stage_id,
                    "token_chunk_size": item.token_chunk_size,
                    "prompt_len": item.seq_group.first_seq.get_prompt_len(),
                    "prefill": item.seq_group.is_prefill(),
                }
                for item in scheduled
            ]
            official_frame = self.policy.num_schedule_count % 50 == 0
            preempted = len(outputs.preempted) if hasattr(outputs.preempted, "__len__") else outputs.preempted
            ignored = [group.request_id for group in outputs.ignored_seq_groups]
            swap_in = len(getattr(outputs, "blocks_to_swap_in", {}))
            swap_out = len(getattr(outputs, "blocks_to_swap_out", {}))
            scheduler_stats["ticks"] += 1
            scheduler_stats["official_frames"] += int(official_frame)
            scheduler_stats["scheduled_groups"] += len(items)
            scheduler_stats["prefill_groups"] += sum(bool(item["prefill"]) for item in items)
            scheduler_stats["prefill_chunks"] += sum(bool(item["prefill"]) and item["token_chunk_size"] < item["prompt_len"] for item in items)
            scheduler_stats["preempted"] += int(preempted or 0)
            scheduler_stats["ignored"] += len(ignored)
            scheduler_stats["swap_in_block_groups"] += swap_in
            scheduler_stats["swap_out_block_groups"] += swap_out
            if official_frame or any(item["prefill"] for item in items) or preempted or ignored or swap_in or swap_out:
                record(
                    "scheduler_step",
                    schedule_count=self.policy.num_schedule_count,
                    official_frame=official_frame,
                    scheduler_type=f"{type(self).__module__}.{type(self).__qualname__}",
                    max_num_seqs=self.scheduler_config.max_num_seqs,
                    max_batch_size=self.max_batch_size,
                    search_strategy=self.search_strategy,
                    queues_before=queues_before,
                    scheduled=items,
                    preempted=preempted,
                    ignored=ignored,
                    swap_in_block_groups=swap_in,
                    swap_out_block_groups=swap_out,
                )
            return result
        return wrapped

    def engine_step(original):
        async def wrapped(self, virtual_engine):
            started = time.perf_counter()
            result = await original(self, virtual_engine)
            scheduler_stats["engine_steps"] += 1
            scheduler_stats["engine_step_wall_seconds"] += time.perf_counter() - started
            return result
        return wrapped

    wrap(server, "_maybe_start_prediction", prediction_submit)
    wrap(server, "send_and_update_request_info", prediction_ready)
    wrap(SequenceGroup, "__init__", assign)
    wrap(SLOPolicy, "allocation_control", allocation)
    wrap(GraphMatchContext, "calculate_stage_ratio", graph_share)
    wrap(GraphMatchContext, "track_collection_completion", graph_completion)
    wrap(SLOScheduler, "schedule", schedule)
    wrap(AsyncLLMEngine, "engine_step", engine_step)

    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs", type=str, default=None)
    parser.add_argument("--ssl-cert-reqs", type=int, default=0)
    parser.add_argument("--root-path", type=str, default=None)
    parser.add_argument("--disable-prediction", action="store_true")
    parser.add_argument("--disable-graph-matching", action="store_true")
    parser.add_argument("--use-default-length", action="store_true")
    parser.add_argument("--prediction-model-path", type=str, required=True)
    parser.add_argument("--prediction-tokenizer-path", type=str, required=True)
    parser.add_argument("--graph-structure-type", choices=["tot", "deepresearch"], default="tot")
    parser.add_argument("--graph-matching-mode", choices=["none", "static", "online", "precise"], default="online")
    parser.add_argument("--use-total-deadline", action="store_true")
    parser.add_argument("--use-all-node", action="store_true")
    parser.add_argument("--stage-ratio-method", choices=["execution_time", "output_length"], default="execution_time")
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    record(
        "launcher_start",
        args={key: value for key, value in vars(args).items() if "keyfile" not in key and "cert" not in key},
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        source_paths={
            "server": inspect.getfile(server),
            "sequence_group": inspect.getfile(SequenceGroup),
            "scheduler": inspect.getfile(SLOScheduler),
            "policy": inspect.getfile(SLOPolicy),
            "graph": inspect.getfile(GraphMatchContext),
        },
    )
    failure = None
    engine = None
    dynamic_scheduler = None
    try:
        HarnessAsyncExecutor.configure(output_dir)
        engine_args = AsyncEngineArgs.from_cli_args(args)
        engine_args.max_model_len = 8192
        engine = HarnessAsyncLLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.API_SERVER)
        dynamic_scheduler = engine.engine.scheduler[0]
        observation.engine_ready(engine)
        record(
            "harness_engine_ready",
            arm_identity="B-full",
            scheduler_type=f"{type(dynamic_scheduler).__module__}.{type(dynamic_scheduler).__qualname__}",
            max_num_seqs=dynamic_scheduler.scheduler_config.max_num_seqs,
            max_batch_size=dynamic_scheduler.max_batch_size,
            search_strategy=dynamic_scheduler.search_strategy,
            max_swaps_per_iter=dynamic_scheduler.max_swaps_per_iter,
            executor=inspect.getfile(HarnessAsyncExecutor),
            gpu_blocks=engine.engine.cache_config.num_gpu_blocks,
            cpu_blocks=engine.engine.cache_config.num_cpu_blocks,
            scheduler=inspect.getfile(type(dynamic_scheduler)),
        )
        asyncio.run(server.run_server(args, llm_engine=engine))
    except BaseException as error:
        failure = {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        record("server_exit", failure_type=failure["type"], failure=failure["error"])
    finally:
        if engine is not None:
            engine.shutdown_background_loop()
            engine.engine.model_executor.finalize()
        graph = getattr(server, "graph_ctx", None)
        dump(
            output_dir / "server_audit_summary.json",
            {
                "arm_identity": "B-full",
                "scheduler_type": (f"{type(dynamic_scheduler).__module__}.{type(dynamic_scheduler).__qualname__}"
                                   if dynamic_scheduler is not None else None),
                "max_num_seqs": (dynamic_scheduler.scheduler_config.max_num_seqs
                                 if dynamic_scheduler is not None else None),
                "max_batch_size": (dynamic_scheduler.max_batch_size
                                   if dynamic_scheduler is not None else None),
                "search_strategy": (dynamic_scheduler.search_strategy
                                    if dynamic_scheduler is not None else None),
                "max_swaps_per_iter": (dynamic_scheduler.max_swaps_per_iter
                                       if dynamic_scheduler is not None else None),
                "scheduler_source": (inspect.getfile(type(dynamic_scheduler))
                                     if dynamic_scheduler is not None else None),
                "failure": failure,
                "event_counts": dict(counts),
                "scheduler_stats": dict(scheduler_stats),
                "base_graphs": len(graph.deepresearch_base_graphs) if graph is not None else 0,
                "clustering": graph.dynamic_clustering.get_cluster_info() if graph is not None else None,
                "use_prediction": getattr(server, "use_prediction", None),
                "use_graph_matching": getattr(server, "use_graph_matching", None),
                "use_default_length": getattr(server, "use_default_length", None),
                "source_paths": {
                    "server": inspect.getfile(server),
                    "sequence_group": inspect.getfile(SequenceGroup),
                    "scheduler": inspect.getfile(SLOScheduler),
                    "policy": inspect.getfile(SLOPolicy),
                    "graph": inspect.getfile(GraphMatchContext),
                },
                "harness_source": inspect.getfile(HarnessAsyncExecutor),
                "wall": time.time(),
            },
        )
        observation.finalize()
        events_file.close()
    if failure is not None and failure["type"] != "KeyboardInterrupt":
        raise RuntimeError(failure["error"])


if __name__ == "__main__":
    main()
