import json
import math
import os
from pathlib import Path
import statistics
import sys


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def load_attempt(root, arm, name):
    path = root / "stage2/attempts" / name
    return {
        "name": name,
        "path": path,
        "calls": json.loads((path / "client_calls.json").read_text()),
        "metrics": json.loads((path / "official_metrics.json").read_text()),
        "server": json.loads((path / "server_audit_summary.json").read_text()),
        "receipt": json.loads((path / "controller_receipt.json").read_text()),
        "executor": json.loads((path / "executor_summary.json").read_text()) if (path / "executor_summary.json").exists() else None,
    }


def request_rows(attempt):
    rows = {}
    main = attempt["calls"][1]
    for stage in main["stages"]:
        for request in stage["requests"]:
            tbt = request.get("request_tbt") or []
            rows[(request["stage_id"], request["request_id"])] = {
                "success": request["success"],
                "latency": request["request_latency"],
                "ttft": request["request_ttft"],
                "tbt_mean": statistics.fmean(tbt) if tbt else None,
                "output_tokens": len(tbt) if tbt else None,
                "stop_reason": request["stop_reason"],
                "finish_before_ddl": request["finish_before_ddl"],
                "service_gain": request["request_service_gain"],
            }
    return rows


def stage_rows(attempt):
    return {
        stage["stage_id"]: {"success": stage["success"], "latency": stage["stage_latency"],
                            "calls": len(stage["requests"])}
        for stage in attempt["calls"][1]["stages"]
    }


def main():
    root = Path(sys.argv[1]).resolve()
    a_name = sys.argv[2]
    b_name = sys.argv[3]
    result_name = sys.argv[4]
    a = load_attempt(root, "A", a_name)
    b = load_attempt(root, "B", b_name)
    b_gate_path = root / "stage2/receipts" / f"{b_name}_B_GATE.json"
    b_gate = json.loads(b_gate_path.read_text())
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    require(b_gate["gate"] == "PASS", "B qualification gate is not PASS")
    for arm, attempt in [("A", a), ("B", b)]:
        require(attempt["receipt"]["failure"] is None and attempt["receipt"]["client_returncode"] == 0,
                f"{arm} controller/client failed")
        require(len(attempt["calls"]) == 2, f"{arm} does not have warmup and main")
        require(all(call["success"] and len(call["stages"]) == 6
                    and [len(stage["requests"]) for stage in call["stages"]] == [1, 2, 1, 1, 1, 1]
                    and all(request["success"] for stage in call["stages"] for request in stage["requests"])
                    for call in attempt["calls"]), f"{arm} compound execution incomplete")
        require(attempt["receipt"]["gpu_released"] is True, f"{arm} GPU0 was not released")

    a_rows = request_rows(a)
    b_rows = request_rows(b)
    require(set(a_rows) == set(b_rows), "A/B request IDs differ")
    request_comparison = {}
    completion_differences = []
    generation_differences = []
    for key in sorted(set(a_rows) & set(b_rows)):
        ra = a_rows[key]
        rb = b_rows[key]
        completion_diff = ra["success"] != rb["success"]
        generation_diff = (ra["output_tokens"] != rb["output_tokens"] or ra["stop_reason"] != rb["stop_reason"])
        if completion_diff:
            completion_differences.append(key)
        if generation_diff:
            generation_differences.append(key)
        request_comparison[str(key)] = {
            "A": ra,
            "B": rb,
            "completion_differs": completion_diff,
            "generation_differs": generation_diff,
            "latency_difference_seconds": rb["latency"] - ra["latency"],
            "ttft_difference_seconds": (rb["ttft"] - ra["ttft"]) if ra["ttft"] is not None and rb["ttft"] is not None else None,
            "tbt_mean_difference_seconds": (rb["tbt_mean"] - ra["tbt_mean"]) if ra["tbt_mean"] is not None and rb["tbt_mean"] is not None else None,
        }
    require(not completion_differences, f"Same-ID completion differs: {completion_differences}")
    require(all(row["output_tokens"] is not None for row in a_rows.values())
            and all(row["output_tokens"] is not None for row in b_rows.values()),
            "Output token observability is missing")

    a_main = a["calls"][1]
    b_main = b["calls"][1]
    a_strict = a_main["success"] and a_main["total_latency"] <= 120.0
    b_strict = b_main["success"] and b_main["total_latency"] <= 120.0
    require(a_strict == b_strict, f"Strict chain SLO differs: A={a_strict}, B={b_strict}")
    latency_relative_difference = abs(b_main["total_latency"] - a_main["total_latency"]) / max(a_main["total_latency"], 1e-9)
    latency_warning = latency_relative_difference > 0.10
    material = completion_differences or generation_differences or a_strict != b_strict or latency_warning

    def metrics_summary(attempt):
        metrics = attempt["metrics"]
        return {
            "main_latency": attempt["calls"][1]["total_latency"],
            "duration": metrics["duration"],
            "slo_met_collections": metrics["slo_met_collections"],
            "slo_met_ratio": metrics["slo_met_ratio"],
            "slo_goodput": metrics["slo_goodput"],
            "total_output_tokens": metrics["total_output_tokens"],
            "mean_ttft_ms": metrics["mean_ttft_ms"],
            "scheduler": attempt["server"]["scheduler_stats"],
            "backend": attempt["executor"]["backend"] if attempt["executor"] else None,
        }

    result = {
        "A_attempt": a_name,
        "B_attempt": b_name,
        "qualification_only": True,
        "statistical_equivalence": False,
        "material_harness_difference": bool(material),
        "material_reasons": {
            "completion_differences": completion_differences,
            "generation_differences": generation_differences,
            "strict_slo_difference": a_strict != b_strict,
            "latency_relative_difference_above_10_percent": latency_warning,
        },
        "A": metrics_summary(a),
        "B": metrics_summary(b),
        "strict_slo": {"A": a_strict, "B": b_strict},
        "latency_relative_difference": latency_relative_difference,
        "stage_comparison": {key: {"A": stage_rows(a).get(key), "B": stage_rows(b).get(key)}
                             for key in sorted(set(stage_rows(a)) | set(stage_rows(b)))},
        "request_comparison": request_comparison,
        "qualification_failures": failures,
        "interpretation": "One A/B qualification block. It measures native-backend/harness differences and is not a B/C mechanism comparison.",
        "next_step": "Do not interpret B/C mechanism differences until harness differences are explained" if material else "No material harness difference under the frozen small-block heuristic; still not a statistical equivalence claim",
    }
    dump(root / "stage2/receipts" / f"{result_name}_AB_COMPARE.json", result)
    print(json.dumps({
        "material_harness_difference": result["material_harness_difference"],
        "strict_slo": result["strict_slo"],
        "A_latency": a_main["total_latency"],
        "B_latency": b_main["total_latency"],
        "latency_relative_difference": latency_relative_difference,
        "generation_differences": generation_differences,
        "qualification_failures": failures,
        "receipt": str(root / "stage2/receipts" / f"{result_name}_AB_COMPARE.json"),
    }, indent=2))
    return 2 if failures else (1 if material else 0)


if __name__ == "__main__":
    sys.exit(main())
