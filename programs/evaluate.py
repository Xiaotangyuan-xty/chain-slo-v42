import json
import math
import os
from pathlib import Path
import sys


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def load(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def main():
    root = Path(sys.argv[1]).resolve()
    attempt_name = sys.argv[2]
    attempt = root / "attempts" / attempt_name
    freeze = json.loads((root / "FREEZE.json").read_text())
    calls = load(attempt / "client_calls.json", [])
    metrics = load(attempt / "official_metrics.json")
    server = load(attempt / "server_audit_summary.json")
    receipt = load(attempt / "controller_receipt.json")
    controller = load(attempt / "controller_status.json")
    events = []
    parse_errors = []
    events_path = attempt / "server_events.jsonl"
    if events_path.exists():
        for line_number, line in enumerate(events_path.read_text().splitlines(), 1):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                parse_errors.append({"line": line_number, "error": str(error)})
    by_kind = {}
    for event in events:
        by_kind.setdefault(event["kind"], []).append(event)

    checks = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    check("official_client_exit_zero", receipt is not None and receipt.get("client_returncode") == 0,
          receipt.get("client_returncode") if receipt else None, 0)
    check("controller_reached_client_complete", controller is not None and controller.get("phase") in {"CLEANUP_COMPLETE", "FAILED_PRESERVED"} and receipt is not None and receipt.get("failure") is None,
          controller.get("phase") if controller else None, "successful controller and cleanup")
    check("no_event_parse_errors", not parse_errors, parse_errors, [])
    check("two_chain_executions", len(calls) == 2, len(calls), 2)
    expected_pattern = freeze["input"]["expected_requests_per_stage"]
    compound_actual = []
    for call in calls:
        compound_actual.append({
            "phase": call.get("phase"),
            "success": call.get("success"),
            "stages": len(call.get("stages", [])),
            "requests_per_stage": [len(stage.get("requests", [])) for stage in call.get("stages", [])],
            "all_requests_success": all(request.get("success") for stage in call.get("stages", []) for request in stage.get("requests", [])),
        })
    compound_ok = len(calls) == 2 and all(
        call.get("success")
        and len(call.get("stages", [])) == 6
        and [len(stage.get("requests", [])) for stage in call.get("stages", [])] == expected_pattern
        and all(request.get("success") for stage in call.get("stages", []) for request in stage.get("requests", []))
        for call in calls
    )
    check("compound_warmup_and_main_complete", compound_ok, compound_actual,
          {"executions": 2, "stages_each": 6, "requests_per_stage": expected_pattern, "calls_each": 7})

    event_expectations = {
        "prediction_submit": 14,
        "prediction_worker_begin": 14,
        "prediction_ready": 14,
        "sequence_group_assign": 14,
        "graph_stage_share": 14,
        "graph_completion": 14,
    }
    for kind, expected in event_expectations.items():
        check(f"event_count_{kind}", len(by_kind.get(kind, [])) == expected,
              len(by_kind.get(kind, [])), expected)
    check("collective_prediction_actually_read", len(by_kind.get("collective_allocation_read", [])) > 0,
          len(by_kind.get("collective_allocation_read", [])), ">0")
    shares = [event.get("share") for event in by_kind.get("graph_stage_share", [])]
    shares_valid = len(shares) == 14 and all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 for value in shares)
    check("graph_shares_finite_and_bounded", shares_valid, shares, "14 finite shares in [0,1]")
    check("official_training_graphs_loaded", server is not None and server.get("base_graphs") == 394,
          server.get("base_graphs") if server else None, 394)
    clustering = server.get("clustering") if server else None
    check("online_graph_state_updated", clustering is not None and clustering.get("n_requests") == 2,
          clustering, {"n_requests": 2, "medoids": "expected empty below 100-graph threshold"})
    check("prediction_and_graph_enabled", server is not None and server.get("use_prediction") is True and server.get("use_graph_matching") is True and server.get("use_default_length") is False,
          {key: server.get(key) for key in ["use_prediction", "use_graph_matching", "use_default_length"]} if server else None,
          {"use_prediction": True, "use_graph_matching": True, "use_default_length": False})
    source_paths = server.get("source_paths", {}) if server else {}
    repo_prefix = str(root / "repo") + os.sep
    check("loaded_sources_from_new_official_repo", bool(source_paths) and all(str(path).startswith(repo_prefix) for path in source_paths.values()),
          source_paths, f"all paths under {repo_prefix}")
    scheduler_stats = server.get("scheduler_stats", {}) if server else {}
    check("official_scheduler_frames_observed", scheduler_stats.get("official_frames", 0) > 0,
          scheduler_stats, "SLOScheduler official_frames > 0")

    strict = None
    goodput_expected = None
    if metrics and len(calls) == 2 and calls[1].get("total_latency") is not None:
        main = calls[1]
        strict = main["success"] and main["total_latency"] <= freeze["deadline"]["whole_chain_deadline_seconds"]
        expected_slo_collections = int(strict)
        expected_slo_requests = 7 if strict else 0
        goodput_expected = expected_slo_requests / metrics["duration"] if metrics["duration"] > 0 else None
        check("official_collection_denominator", metrics.get("total_collections") == 1 and metrics.get("completed_collections") == 1,
              {"total": metrics.get("total_collections"), "completed": metrics.get("completed_collections")}, {"total": 1, "completed": 1})
        check("official_stage_and_call_denominator", metrics.get("total_stages") == 6 and metrics.get("total_requests") == 7 and metrics.get("completed_requests") == 7,
              {"stages": metrics.get("total_stages"), "requests": metrics.get("total_requests"), "completed_requests": metrics.get("completed_requests")},
              {"stages": 6, "requests": 7, "completed_requests": 7})
        check("strict_chain_slo_recomputed", metrics.get("slo_met_collections") == expected_slo_collections,
              {"official": metrics.get("slo_met_collections"), "recomputed": expected_slo_collections, "latency": main["total_latency"]},
              "finish <= 120 seconds")
        check("native_slo_goodput_recomputed", goodput_expected is not None and math.isclose(metrics.get("slo_goodput", -1), goodput_expected, rel_tol=1e-9, abs_tol=1e-9),
              {"official": metrics.get("slo_goodput"), "recomputed": goodput_expected}, "equal within 1e-9")
    else:
        for name in ["official_collection_denominator", "official_stage_and_call_denominator", "strict_chain_slo_recomputed", "native_slo_goodput_recomputed"]:
            check(name, False, None, "available official metrics and main output")

    check("gpu2_released", receipt is not None and receipt.get("gpu_released") is True,
          receipt.get("final_gpu2") if receipt else None, "no compute processes on physical GPU2")
    passed = all(item["pass"] for item in checks.values())
    result = {
        "study_id": freeze["study_id"],
        "attempt": attempt_name,
        "arm": "A",
        "gate": "PASS" if passed else "FAIL",
        "qualification_only": True,
        "performance_conclusion": False,
        "checks": checks,
        "summary": {
            "chain_executions": len(calls),
            "main_strict_slo": strict,
            "official_metrics": metrics,
            "scheduler_stats": scheduler_stats,
            "clustering": clustering,
            "gpu_released": receipt.get("gpu_released") if receipt else None,
        },
        "next_step": "A is qualified for a separately authorized B semantic gate" if passed else "Stop at stage1; do not start B/C performance runs",
    }
    dump(root / "RECEIPT" / f"{attempt_name}_GATE.json", result)
    dump(attempt / "GATE.json", result)
    print(json.dumps({"gate": result["gate"], "failed_checks": [name for name, item in checks.items() if not item["pass"]], "receipt": str(root / "RECEIPT" / f"{attempt_name}_GATE.json")}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
