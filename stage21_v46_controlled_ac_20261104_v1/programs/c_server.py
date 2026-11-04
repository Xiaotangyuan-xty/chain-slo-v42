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
    repo = Path(prepare["repo"])
    sys.path[:0] = [
        str(root / "programs"),
        str(repo),
        str(repo / "third_party/vllm"),
        prepare["sklearn_deps"],
        prepare["support_deps"],
    ]
    from gpu2_platform import install

    device_receipt = install(root, stage, shared_backend=False)
    from safe_statistics import install as install_safe_statistics
    install_safe_statistics(root,stage,output_dir)
    from lifetime_observation import install as install_lifetime
    install_lifetime(root,stage,output_dir)
    import torch
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sequence import SequenceGroup
    from vllm.utils import FlexibleArgumentParser
    from jitserve import server
    from jitserve.request_analyzer.graph_context import GraphMatchContext
    from jitserve.scheduler.policy import SLOPolicy
    from jitserve.scheduler.slo_scheduler import SLOScheduler

    from scheduler_v46 import install as install_v46
    install_v46(server)
    protocol=json.loads((stage/'PROTOCOL.json').read_text())
    variant=protocol.get('attempt_variants',{}).get(output_dir.name,'native')
    if variant=='corrected':
        from baseline_budget import install as install_budget_adapter
        install_budget_adapter(output_dir)
    from observation_v46 import install_execution_observation
    install_execution_observation(server,output_dir,'v46_shared_execution_observer_v1',variant)
    from abc_observation import install_observation
    observation = install_observation(server, AsyncLLMEngine, output_dir, "C", root, device_receipt)

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
            result = original(self, *args, **kwargs)
            numeric = isinstance(result, (int, float)) and math.isfinite(result)
            record(
                "graph_stage_share",
                collection_id=args[3] if len(args) > 3 else kwargs.get("collection_id"),
                stage_id=kwargs.get("stage_id", 0),
                share=result,
                finite=numeric,
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
            result = original(self, *args, **kwargs)
            _, outputs, _ = result
            scheduled = outputs.scheduled_seq_groups
            preempted = len(outputs.preempted) if hasattr(outputs.preempted, "__len__") else outputs.preempted
            scheduler_stats["ticks"] += 1
            scheduler_stats["official_frames"] += int(self.policy.num_schedule_count % 50 == 0)
            scheduler_stats["scheduled_groups"] += len(scheduled)
            scheduler_stats["prefill_groups"] += sum(x.seq_group.is_prefill() for x in scheduled)
            scheduler_stats["preempted"] += int(preempted or 0)
            scheduler_stats["ignored"] += len(outputs.ignored_seq_groups)
            scheduler_stats["swap_in_block_groups"] += len(getattr(outputs, "blocks_to_swap_in", {}))
            scheduler_stats["swap_out_block_groups"] += len(getattr(outputs, "blocks_to_swap_out", {}))
            return result
        return wrapped

    original_init_app = server.init_app
    async def observed_init_app(*args, **kwargs):
        app = await original_init_app(*args, **kwargs)
        observation.engine_ready(server.engine)
        return app
    server.init_app = observed_init_app

    wrap(server, "_maybe_start_prediction", prediction_submit)
    wrap(server, "send_and_update_request_info", prediction_ready)
    wrap(SequenceGroup, "__init__", assign)
    wrap(GraphMatchContext, "calculate_stage_ratio", graph_share)
    wrap(GraphMatchContext, "track_collection_completion", graph_completion)
    wrap(SLOScheduler, "schedule", schedule)

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
    try:
        asyncio.run(server.run_server(args))
    except BaseException as error:
        failure = {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        record("server_exit", failure_type=failure["type"], failure=failure["error"])
    finally:
        graph = getattr(server, "graph_ctx", None)
        dump(
            output_dir / "server_audit_summary.json",
            {
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
                "wall": time.time(),
            },
        )
        observation.finalize()
        events_file.close()
    if failure is not None and failure["type"] != "KeyboardInterrupt":
        raise RuntimeError(failure["error"])


if __name__ == "__main__":
    main()
