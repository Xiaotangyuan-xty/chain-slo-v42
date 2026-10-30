import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(32 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def main():
    root = Path(sys.argv[1]).resolve()
    checks = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    def compare(name, entries, resolver):
        missing = []
        mismatch = []
        matched = 0
        for item in entries:
            path = Path(resolver(item))
            if not path.is_file():
                missing.append(str(path))
            elif sha(path) != item["sha256"]:
                mismatch.append(str(path))
            else:
                matched += 1
        check(name, not missing and not mismatch,
              {"matched": matched, "missing": missing, "mismatch": mismatch},
              {"matched": len(entries), "missing": [], "mismatch": []})

    official = json.loads((root / "manifests/official_source_hashes.json").read_text())
    entries = [{"path": path, **value} for path, value in official["files"].items()]
    compare("official_source_1297", entries, lambda item: root / "repo" / item["path"])
    launch = json.loads((root / "stage2/manifests/launch.json").read_text())
    for group in ["programs", "backend", "contracts", "runtime_gpu0"]:
        compare(f"stage2_{group}_sha", launch[group], lambda item: item["path"])
    assets = json.loads((root / "manifests/assets.json").read_text())
    compare("qrf_trace_assets_sha", assets["files"], lambda item: item["path"])
    model = json.loads((root / "manifests/model.json").read_text())
    compare("model_assets_sha", model["files"], lambda item: item["path"])

    controlled = json.loads((root / "stage2/receipts/controlled_08.json").read_text())
    b1_gate = json.loads((root / "stage2/receipts/B_chain0_01_B_GATE.json").read_text())
    b2_gate = json.loads((root / "stage2/receipts/B_chain0_stage2_01_B_GATE.json").read_text())
    check("controlled_and_B_gates", controlled["gate"] == "PASS" and b1_gate["gate"] == "PASS" and b2_gate["gate"] == "PASS",
          {"controlled": controlled["gate"], "B_qualification": b1_gate["gate"], "B_pair": b2_gate["gate"]}, "all PASS")

    names = ["A_chain0_stage2_01", "B_chain0_stage2_01"]
    attempts = {}
    for name in names:
        path = root / "stage2/attempts" / name
        attempts[name] = {
            "calls": json.loads((path / "client_calls.json").read_text()),
            "metrics": json.loads((path / "official_metrics.json").read_text()),
            "receipt": json.loads((path / "controller_receipt.json").read_text()),
            "server": json.loads((path / "server_audit_summary.json").read_text()),
            "executor": json.loads((path / "executor_summary.json").read_text()) if (path / "executor_summary.json").exists() else None,
        }
    check("AB_attempts_complete", all(item["receipt"]["failure"] is None and item["receipt"]["client_returncode"] == 0 for item in attempts.values()),
          {name: {"failure": item["receipt"]["failure"], "client": item["receipt"]["client_returncode"]} for name, item in attempts.items()},
          "both complete without controller failure")
    check("A_before_B", attempts["A_chain0_stage2_01"]["receipt"]["utc"] < attempts["B_chain0_stage2_01"]["receipt"]["utc"],
          {"A": attempts["A_chain0_stage2_01"]["receipt"]["utc"], "B": attempts["B_chain0_stage2_01"]["receipt"]["utc"]}, "A→B order")
    attempt_dirs = sorted(path.name for path in (root / "stage2/attempts").iterdir() if path.is_dir())
    check("only_qualification_and_pair_attempts", attempt_dirs == ["A_chain0_stage2_01", "B_chain0_01", "B_chain0_stage2_01"],
          attempt_dirs, ["A_chain0_stage2_01", "B_chain0_01", "B_chain0_stage2_01"])

    strict = {}
    recomputed = {}
    request_latencies = {}
    for name, item in attempts.items():
        main = item["calls"][1]
        on_time = main["success"] and main["total_latency"] <= 120.0
        strict[name] = on_time
        expected_requests = 7 if on_time else 0
        goodput = expected_requests / item["metrics"]["duration"]
        recomputed[name] = goodput
        check(f"strict_and_goodput_{name}", item["metrics"]["slo_met_collections"] == int(on_time)
              and math.isclose(item["metrics"]["slo_goodput"], goodput, rel_tol=1e-9, abs_tol=1e-9),
              {"strict": on_time, "official_goodput": item["metrics"]["slo_goodput"], "recomputed": goodput},
              "official strict SLO and native goodput match")
        request_latencies[name] = [request["request_latency"] for stage in main["stages"] for request in stage["requests"]]

    gained = [0] if strict["B_chain0_stage2_01"] and not strict["A_chain0_stage2_01"] else []
    lost = [0] if strict["A_chain0_stage2_01"] and not strict["B_chain0_stage2_01"] else []
    check("same_ID_chain_ledger", gained == [] and lost == [0], {"gained": gained, "lost": lost}, "B loses chain0 relative to A")

    comparison = json.loads((root / "stage2/receipts/pair_01_AB_COMPARE.json").read_text())
    latency_relative = abs(attempts["B_chain0_stage2_01"]["calls"][1]["total_latency"] - attempts["A_chain0_stage2_01"]["calls"][1]["total_latency"]) / attempts["A_chain0_stage2_01"]["calls"][1]["total_latency"]
    check("material_difference_recomputed", comparison["material_harness_difference"] is True
          and strict["A_chain0_stage2_01"] != strict["B_chain0_stage2_01"] and latency_relative > 0.10
          and len(comparison["material_reasons"]["generation_differences"]) > 0,
          {"strict_A": strict["A_chain0_stage2_01"], "strict_B": strict["B_chain0_stage2_01"],
           "latency_relative": latency_relative, "generation_differences": comparison["material_reasons"]["generation_differences"]},
          "material harness difference; stop mechanism interpretation")

    registered = {}
    for name, item in attempts.items():
        for process in item["receipt"]["processes"]:
            registered[str(process["pid"])] = name
    gpu_processes = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True).strip()
    own_remaining = [line for line in gpu_processes.splitlines() if line.split(",", 1)[0].strip() in registered]
    gpu = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"], text=True).strip()
    check("own_GPU_processes_released", not own_remaining and all(item["receipt"]["gpu_released"] for item in attempts.values()),
          {"gpu": gpu, "own_remaining": own_remaining, "all_gpu_processes": gpu_processes}, "no study-owned GPU process; foreign processes untouched")
    check("all_registered_processes_exited", all(process["returncode"] is not None for item in attempts.values() for process in item["receipt"]["processes"]),
          {name: [{"pid": process["pid"], "returncode": process["returncode"]} for process in item["receipt"]["processes"]]
           for name, item in attempts.items()}, "all registered PIDs exited")

    decision_pass = all(item["pass"] for item in checks.values())
    result = {
        "audit": "PASS" if decision_pass else "FAIL",
        "decision": "STOP_BEFORE_FORMAL_MATRIX",
        "reason": "Material native-backend/harness difference: strict chain SLO, collection latency and generation boundaries differ.",
        "checks": checks,
        "strict_slo": strict,
        "same_ID_gained": gained,
        "same_ID_lost": lost,
        "native_goodput_recomputed": recomputed,
        "latency": {
            "A_main_JCT": attempts["A_chain0_stage2_01"]["calls"][1]["total_latency"],
            "B_main_JCT": attempts["B_chain0_stage2_01"]["calls"][1]["total_latency"],
            "relative_difference": latency_relative,
            "A_request_P90_P95": [percentile(request_latencies["A_chain0_stage2_01"], 90), percentile(request_latencies["A_chain0_stage2_01"], 95)],
            "B_request_P90_P95": [percentile(request_latencies["B_chain0_stage2_01"], 90), percentile(request_latencies["B_chain0_stage2_01"], 95)],
        },
        "costs": {
            "A": attempts["A_chain0_stage2_01"]["server"]["scheduler_stats"],
            "B": attempts["B_chain0_stage2_01"]["server"]["scheduler_stats"],
            "B_backend": {key: value for key, value in attempts["B_chain0_stage2_01"]["executor"]["backend"].items()
                          if key != "native_decode_batches"},
        },
        "formal_matrix_started": False,
        "C_started": False,
        "HOLDOUT_started": False,
        "mechanism_conclusion_allowed": False,
        "next_step": "Stop and report the harness difference. Do not repair, narrow or expand experiments automatically.",
    }
    dump(root / "stage2/receipts/FINAL_STAGE2_AUDIT.json", result)
    print(json.dumps({"audit": result["audit"], "decision": result["decision"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(root / "stage2/receipts/FINAL_STAGE2_AUDIT.json")}, indent=2))
    return 0 if decision_pass else 1


if __name__ == "__main__":
    sys.exit(main())
