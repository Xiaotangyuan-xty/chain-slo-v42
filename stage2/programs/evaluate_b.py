import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def main():
    root = Path(sys.argv[1]).resolve()
    attempt_name = sys.argv[2]
    attempt = root / "stage2/attempts" / attempt_name
    freeze = json.loads((root / "stage2/STAGE2_FREEZE.json").read_text())
    calls = json.loads((attempt / "client_calls.json").read_text()) if (attempt / "client_calls.json").exists() else []
    metrics = json.loads((attempt / "official_metrics.json").read_text()) if (attempt / "official_metrics.json").exists() else None
    server = json.loads((attempt / "server_audit_summary.json").read_text()) if (attempt / "server_audit_summary.json").exists() else None
    executor = json.loads((attempt / "executor_summary.json").read_text()) if (attempt / "executor_summary.json").exists() else None
    receipt = json.loads((attempt / "controller_receipt.json").read_text()) if (attempt / "controller_receipt.json").exists() else None
    events = [json.loads(line) for line in (attempt / "server_events.jsonl").read_text().splitlines()] if (attempt / "server_events.jsonl").exists() else []
    executor_events = [json.loads(line) for line in (attempt / "executor_events.jsonl").read_text().splitlines()] if (attempt / "executor_events.jsonl").exists() else []
    kinds = {}
    for event in events:
        kinds.setdefault(event["kind"], []).append(event)
    checks = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    check("client_and_controller_success", receipt is not None and receipt.get("failure") is None and receipt.get("client_returncode") == 0,
          {"failure": receipt.get("failure") if receipt else None, "client_returncode": receipt.get("client_returncode") if receipt else None},
          {"failure": None, "client_returncode": 0})
    compound = []
    for call in calls:
        compound.append({
            "phase": call.get("phase"),
            "success": call.get("success"),
            "stages": len(call.get("stages", [])),
            "requests_per_stage": [len(stage.get("requests", [])) for stage in call.get("stages", [])],
            "all_requests_success": all(request.get("success") for stage in call.get("stages", []) for request in stage.get("requests", [])),
        })
    check("warmup_main_compound", len(calls) == 2 and all(
        item["success"] and item["stages"] == 6 and item["requests_per_stage"] == [1, 2, 1, 1, 1, 1] and item["all_requests_success"]
        for item in compound), compound, "2 successful executions, each 6 stages and 7 calls")
    for kind in ["prediction_submit", "prediction_worker_begin", "prediction_ready", "sequence_group_assign", "graph_stage_share", "graph_completion"]:
        check(f"event_count_{kind}", len(kinds.get(kind, [])) == 14, len(kinds.get(kind, [])), 14)
    lifecycle = {}
    for event in events:
        info = event.get("info_object")
        if info is not None:
            lifecycle.setdefault(info, {})[event["kind"]] = event["monotonic"]
    bad = []
    for info, item in lifecycle.items():
        required = {"prediction_submit", "prediction_worker_begin", "prediction_ready", "sequence_group_assign"}
        if not required.issubset(item) or not (
                item["prediction_submit"] <= item["prediction_worker_begin"] <= item["prediction_ready"]
                and item["prediction_submit"] <= item["sequence_group_assign"]):
            bad.append(info)
    check("prediction_lifecycle", len(lifecycle) == 14 and not bad, {"calls": len(lifecycle), "bad": bad}, "14 ordered complete lifecycles")
    check("actual_collective_reads", len(kinds.get("collective_allocation_read", [])) > 0,
          len(kinds.get("collective_allocation_read", [])), ">0")
    shares = [event["share"] for event in kinds.get("graph_stage_share", [])]
    check("graph_shares", len(shares) == 14 and all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 for value in shares),
          shares, "14 finite shares in [0,1]")
    clustering = server.get("clustering") if server else None
    check("graph_state", server is not None and server.get("base_graphs") == 394 and clustering is not None and clustering.get("n_requests") == 2,
          {"base_graphs": server.get("base_graphs") if server else None, "clustering": clustering}, "394 base graphs and 2 finished graphs")
    scheduler = server.get("scheduler_stats", {}) if server else {}
    check("official_scheduler_frames", scheduler.get("official_frames", 0) > 0 and scheduler.get("prefill_chunks", 0) > 0,
          scheduler, "official frames and chunked prefill observed")
    launcher = kinds.get("launcher_start", [{}])[0] if kinds.get("launcher_start") else {}
    args = launcher.get("args", {})
    resources = freeze["matched_resources"]
    expected_args = {
        "max_num_seqs": resources["max_num_seqs"],
        "max_num_batched_tokens": resources["max_num_batched_tokens"],
        "num_gpu_blocks_override": resources["num_gpu_blocks_override"],
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "disable_async_output_proc": True,
    }
    check("matched_launch_arguments", all(args.get(key) == value for key, value in expected_args.items()),
          {key: args.get(key) for key in expected_args}, expected_args)
    check("gpu0_uuid_mask", launcher.get("cuda_visible_devices") == "0", launcher.get("cuda_visible_devices"), "0")
    harness = kinds.get("harness_engine_ready", [{}])[0] if kinds.get("harness_engine_ready") else {}
    check("harness_engine_and_blocks", harness.get("gpu_blocks") == 512 and harness.get("cpu_blocks") == 1170,
          {"gpu_blocks": harness.get("gpu_blocks"), "cpu_blocks": harness.get("cpu_blocks")},
          {"gpu_blocks": 512, "cpu_blocks": 1170})
    check("harness_executor_summary", executor is not None and executor.get("cache_block_bytes") == 917504
          and executor.get("backend", {}).get("persistent_quota_tokens") == 8192
          and executor.get("backend", {}).get("host_quota_tokens") == 18720,
          executor, "917504 bytes/block, 8192 GPU and 18720 host tokens")

    from trace.trace_deepresearch import DeepResearchTrace
    from vllm.transformers_utils.tokenizer import get_tokenizer
    prepare = json.loads((root / "PREPARE.json").read_text())
    tokenizer = get_tokenizer(prepare["model"])
    infos = DeepResearchTrace.load_trace(str(root / freeze["public_input"]["trace"]))[0].to_request_infos()
    expected_hashes = {}
    for info in infos:
        tokens = list(tokenizer.encode(info.prompt, add_special_tokens=True))
        expected_hashes[(info.stage_id, info.request_id)] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
    first_prompt = {}
    for event in executor_events:
        if event["kind"] != "execute_model":
            continue
        for group in event["groups"]:
            first_prompt.setdefault(group["request_id"], group["prompt_sha256"])
    prompt_actual = []
    for assign in kinds.get("sequence_group_assign", []):
        expected = expected_hashes[(assign["stage_id"], assign["request_id"])]
        actual = first_prompt.get(assign["request_uuid"])
        prompt_actual.append({"request_uuid": assign["request_uuid"], "match": actual == expected})
    check("actual_public_prompt_tokens", len(prompt_actual) == 14 and all(item["match"] for item in prompt_actual),
          prompt_actual, "all 14 first-execution prompt hashes match public trace")

    assigned_ids = {event["request_uuid"] for event in kinds.get("sequence_group_assign", [])}
    released_ids = {event["request_id"] for event in executor_events if event["kind"] == "backend_release"}
    backend = executor.get("backend", {}) if executor else {}
    check("finish_and_final_kv_release", len(assigned_ids) == 14 and assigned_ids <= released_ids
          and executor.get("tracked_block_owners") == 0 and backend.get("evictions") == 0,
          {"assigned": len(assigned_ids), "released": len(assigned_ids & released_ids),
           "tracked_block_owners": executor.get("tracked_block_owners") if executor else None,
           "evictions": backend.get("evictions")},
          "all 14 released, no owners or autonomous LRU evictions")
    check("backend_real_forward", backend.get("actual_forward_query_tokens", 0) > 0,
          backend.get("actual_forward_query_tokens"), ">0")

    strict = None
    if metrics and len(calls) == 2:
        main_latency = calls[1]["total_latency"]
        strict = calls[1]["success"] and main_latency <= 120.0
        expected_collections = int(strict)
        expected_requests = 7 if strict else 0
        expected_goodput = expected_requests / metrics["duration"]
        check("official_denominators", metrics["total_collections"] == 1 and metrics["completed_collections"] == 1
              and metrics["total_stages"] == 6 and metrics["total_requests"] == 7 and metrics["completed_requests"] == 7,
              {key: metrics[key] for key in ["total_collections", "completed_collections", "total_stages", "total_requests", "completed_requests"]},
              "1 collection, 6 stages, 7 completed calls")
        check("strict_slo_recomputed", metrics["slo_met_collections"] == expected_collections
              and math.isclose(metrics["slo_met_ratio"], expected_collections, abs_tol=1e-12),
              {"latency": main_latency, "strict": strict, "official": metrics["slo_met_collections"]},
              "finish <= 120 seconds")
        check("native_goodput_recomputed", math.isclose(metrics["slo_goodput"], expected_goodput, rel_tol=1e-9, abs_tol=1e-9),
              {"official": metrics["slo_goodput"], "recomputed": expected_goodput}, "equal within 1e-9")
    else:
        for name in ["official_denominators", "strict_slo_recomputed", "native_goodput_recomputed"]:
            check(name, False, None, "available metrics")
    gpu = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"], text=True).strip()
    processes = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True).strip()
    registered_pids = {str(item["pid"]) for item in receipt["processes"]} if receipt else set()
    own_remaining = [line for line in processes.splitlines() if line.split(",", 1)[0].strip() in registered_pids]
    check("gpu0_released", gpu.startswith("0, GPU-ff8c7d5c-f75c-26e1-43de-b40ca283f938,")
          and receipt is not None and receipt.get("gpu_released") is True and not own_remaining,
          {"gpu": gpu, "own_remaining": own_remaining, "all_gpu_processes": processes},
          "GPU0 UUID, controller release receipt, and no study-owned GPU process")

    passed = all(item["pass"] for item in checks.values())
    result = {
        "attempt": attempt_name,
        "arm": "B",
        "gate": "PASS" if passed else "FAIL",
        "qualification_only": True,
        "checks": checks,
        "main_strict_slo": strict,
        "official_metrics": metrics,
        "scheduler_stats": scheduler,
        "clustering": clustering,
        "backend_summary": backend,
        "next_step": "Run the frozen A→B small pair on GPU0" if passed else "Stop before A/B; preserve and diagnose B",
    }
    dump(root / "stage2/receipts" / f"{attempt_name}_B_GATE.json", result)
    if passed:
        dump(root / "stage2/receipts/S2_B_QUALIFICATION_PASS.json", result)
    print(json.dumps({"gate": result["gate"], "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(root / "stage2/receipts" / f"{attempt_name}_B_GATE.json")}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
