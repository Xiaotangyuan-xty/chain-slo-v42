import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def sha(path):
    h = __import__("hashlib").sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(32 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def main():
    root = Path(sys.argv[1]).resolve()
    freeze = json.loads((root / "FREEZE.json").read_text())
    attempt = root / "attempts/A_chain0_02"
    checks = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    def compare_manifest(name, entries, resolver):
        missing = []
        mismatch = []
        matched = 0
        for item in entries:
            path = Path(resolver(item))
            if not path.is_file():
                missing.append(str(path))
                continue
            if sha(path) != item["sha256"]:
                mismatch.append(str(path))
            else:
                matched += 1
        check(name, not missing and not mismatch,
              {"matched": matched, "missing": missing, "mismatch": mismatch},
              {"matched": len(entries), "missing": [], "mismatch": []})

    official = json.loads((root / "manifests/official_source_hashes.json").read_text())
    source_entries = [{"path": path, **value} for path, value in official["files"].items()]
    compare_manifest("official_source_new_1297", source_entries, lambda item: root / "repo" / item["path"])
    old_root = Path("/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/slo_declaration_robustness/gpu_runtime_20261014T050639Z_v1/deployment_v2/repo")
    compare_manifest("official_source_old_unchanged_1297", source_entries, lambda item: old_root / item["path"])
    native = json.loads((root / "manifests/native_binary_reuse.json").read_text())
    compare_manifest("native_runtime_5", native["files"], lambda item: root / item["path"])
    assets = json.loads((root / "manifests/assets.json").read_text())
    compare_manifest("qrf_and_trace_assets_4", assets["files"], lambda item: item["path"])
    model = json.loads((root / "manifests/model.json").read_text())
    compare_manifest("model_assets_13", model["files"], lambda item: item["path"])
    launch = json.loads((root / "manifests/launch_freeze.json").read_text())
    compare_manifest("frozen_programs_6", launch["programs"], lambda item: item["path"])
    compare_manifest("frozen_contracts_5", launch["contracts"], lambda item: item["path"])
    native_launch = launch["native_runtime_manifest"]
    check("native_manifest_launch_binding", sha(root / "manifests/native_binary_reuse.json") == native_launch["sha256"],
          sha(root / "manifests/native_binary_reuse.json"), native_launch["sha256"])

    calls = json.loads((attempt / "client_calls.json").read_text())
    metrics = json.loads((attempt / "official_metrics.json").read_text())
    server = json.loads((attempt / "server_audit_summary.json").read_text())
    receipt = json.loads((attempt / "controller_receipt.json").read_text())
    events = [json.loads(line) for line in (attempt / "server_events.jsonl").read_text().splitlines()]
    kinds = {}
    for event in events:
        kinds.setdefault(event["kind"], []).append(event)
    expected_pattern = [1, 2, 1, 1, 1, 1]
    compound = [
        {
            "phase": call["phase"],
            "success": call["success"],
            "latency": call["total_latency"],
            "stages": len(call["stages"]),
            "requests_per_stage": [len(stage["requests"]) for stage in call["stages"]],
            "all_requests_success": all(request["success"] for stage in call["stages"] for request in stage["requests"]),
        }
        for call in calls
    ]
    check("warmup_main_full_compound", len(calls) == 2 and all(
        call["success"] and len(call["stages"]) == 6
        and [len(stage["requests"]) for stage in call["stages"]] == expected_pattern
        and all(request["success"] for stage in call["stages"] for request in stage["requests"])
        for call in calls), compound, "2 successful executions, each 6 stages and 7 calls")

    for kind in ["prediction_submit", "prediction_worker_begin", "prediction_ready", "sequence_group_assign", "graph_stage_share", "graph_completion"]:
        check(f"raw_event_count_{kind}", len(kinds.get(kind, [])) == 14, len(kinds.get(kind, [])), 14)
    lifecycle = {}
    for event in events:
        info = event.get("info_object")
        if info is not None:
            lifecycle.setdefault(info, {})[event["kind"]] = event["monotonic"]
    lifecycle_bad = []
    for info, item in lifecycle.items():
        required = {"prediction_submit", "prediction_worker_begin", "prediction_ready", "sequence_group_assign"}
        if not required.issubset(item):
            lifecycle_bad.append({"info": info, "missing": sorted(required - set(item))})
            continue
        if not (item["prediction_submit"] <= item["prediction_worker_begin"] <= item["prediction_ready"]
                and item["prediction_submit"] <= item["sequence_group_assign"]):
            lifecycle_bad.append({"info": info, "times": item})
    check("prediction_lifecycle_per_14_calls", len(lifecycle) == 14 and not lifecycle_bad,
          {"calls": len(lifecycle), "bad": lifecycle_bad}, "14 complete and ordered lifecycles")
    check("actual_collective_reads", len(kinds.get("collective_allocation_read", [])) == 172,
          len(kinds.get("collective_allocation_read", [])), 172)
    ready_before_assign = sum(event["ready_before_assign"] for event in kinds["sequence_group_assign"])
    snapshot_1024 = sum(event["snapshot_output_len"] == 1024 for event in kinds["sequence_group_assign"])
    shares = [event["share"] for event in kinds["graph_stage_share"]]
    check("raw_graph_shares", len(shares) == 14 and all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 for value in shares),
          shares, "14 finite values in [0,1]")
    check("raw_graph_final_state", server["base_graphs"] == 394 and server["clustering"]["n_requests"] == 2
          and server["clustering"]["n_clusters"] == 0 and server["clustering"]["medoids"] == [],
          {"base_graphs": server["base_graphs"], "clustering": server["clustering"]},
          "394 official base graphs, 2 finished, no medoids below 100-graph threshold")
    check("raw_scheduler_activity", server["scheduler_stats"]["official_frames"] == 130
          and server["scheduler_stats"]["ticks"] == 6523 and server["scheduler_stats"]["preempted"] == 0,
          server["scheduler_stats"], "6523 ticks, 130 official frames, no preemption")

    main_call = calls[1]
    strict = main_call["success"] and main_call["total_latency"] <= 120.0
    expected_collections = int(strict)
    expected_requests = 7 if strict else 0
    expected_goodput = expected_requests / metrics["duration"]
    check("strict_chain_slo_independent", metrics["slo_met_collections"] == expected_collections
          and metrics["slo_met_ratio"] == expected_collections / metrics["completed_collections"],
          {"main_latency": main_call["total_latency"], "deadline": 120.0, "strict": strict,
           "official_collections": metrics["slo_met_collections"], "official_ratio": metrics["slo_met_ratio"]},
          "finish <= deadline with completed-collection denominator")
    check("native_goodput_independent", math.isclose(metrics["slo_goodput"], expected_goodput, rel_tol=1e-12, abs_tol=1e-12),
          {"official": metrics["slo_goodput"], "recomputed": expected_goodput}, "equal within 1e-12")
    check("official_denominators_independent", metrics["total_collections"] == 1 and metrics["completed_collections"] == 1
          and metrics["total_stages"] == 6 and metrics["total_requests"] == 7 and metrics["completed_requests"] == 7,
          {key: metrics[key] for key in ["total_collections", "completed_collections", "total_stages", "total_requests", "completed_requests"]},
          {"total_collections": 1, "completed_collections": 1, "total_stages": 6, "total_requests": 7, "completed_requests": 7})

    launcher = kinds["launcher_start"][0]
    check("gpu_mask_and_chunked_prefill_from_raw_launch", launcher["cuda_visible_devices"] == freeze["resources"]["gpu_uuid"]
          and launcher["args"]["enable_chunked_prefill"] is True,
          {"cuda_visible_devices": launcher["cuda_visible_devices"], "enable_chunked_prefill": launcher["args"]["enable_chunked_prefill"]},
          {"cuda_visible_devices": freeze["resources"]["gpu_uuid"], "enable_chunked_prefill": True})
    check("registered_processes_exited", all(item["returncode"] is not None for item in receipt["processes"]),
          [{"name": item["name"], "returncode": item["returncode"]} for item in receipt["processes"]], "all registered processes exited")
    gpu = subprocess.check_output(["nvidia-smi", "-i", "2", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"], text=True).strip()
    gpu_processes = subprocess.check_output(["nvidia-smi", "-i", "2", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True).strip()
    check("live_gpu2_release", gpu.startswith("2, GPU-e085e0e1-a283-82f2-325b-01981b51f984,") and not gpu_processes,
          {"gpu": gpu, "processes": gpu_processes}, "correct GPU2 UUID and no compute processes")
    port_status = {}
    for port in [18080, 65433]:
        try:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
            port_status[str(port)] = "free"
        except OSError as error:
            port_status[str(port)] = str(error)
    check("http_and_prediction_ports_released", set(port_status.values()) == {"free"}, port_status, "both free")
    attempt_dirs = sorted(path.name for path in (root / "attempts").iterdir() if path.is_dir())
    check("only_two_A_attempts_no_BC", attempt_dirs == ["A_chain0_01", "A_chain0_02"], attempt_dirs,
          ["A_chain0_01", "A_chain0_02"])
    check("attempt1_failure_preserved", (root / "RECEIPT/A_chain0_01_GATE.json").is_file()
          and json.loads((root / "RECEIPT/A_chain0_01_GATE.json").read_text())["gate"] == "FAIL"
          and (attempt.parent / "A_chain0_01/DIAGNOSIS.json").is_file(),
          "attempt1 FAIL gate and diagnosis", "preserved")

    result = {
        "study_id": freeze["study_id"],
        "attempt": "A_chain0_02",
        "final_gate": "PASS" if all(item["pass"] for item in checks.values()) else "FAIL",
        "qualification_only": True,
        "performance_conclusion": False,
        "checks": checks,
        "key_observations": {
            "warmup_and_main": compound,
            "ready_before_assign": ready_before_assign,
            "snapshot_1024_at_assign": snapshot_1024,
            "collective_reads": len(kinds["collective_allocation_read"]),
            "graph_shares": shares,
            "official_metrics": metrics,
            "post_cleanup_transient": {
                "pid": 2977937,
                "registered_study_pid": False,
                "signal_sent": False,
                "state_at_identity_check": "already exited",
                "final_gpu2": gpu,
            },
        },
        "next_step": "A native smoke is qualified. No B/C performance work was started; any later B semantic gate requires separate instruction.",
        "utc": time.time(),
    }
    dump(root / "RECEIPT/FINAL_ACCEPTANCE.json", result)
    print(json.dumps({"final_gate": result["final_gate"], "check_count": len(checks),
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(root / "RECEIPT/FINAL_ACCEPTANCE.json")}, indent=2))
    return 0 if result["final_gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
