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
    position = (len(values) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def main():
    root = Path(sys.argv[1]).resolve()
    attempts = {}
    for name in ["BC_B_01", "BC_C_01"]:
        path = root / "stage3/attempts" / name
        attempts[name] = {
            "calls": json.loads((path / "client_calls.json").read_text()),
            "metrics": json.loads((path / "official_metrics.json").read_text()),
            "receipt": json.loads((path / "controller_receipt.json").read_text()),
            "server": json.loads((path / "server_audit_summary.json").read_text()),
            "executor": json.loads((path / "executor_summary.json").read_text()),
            "executor_events": [json.loads(line) for line in (path / "executor_events.jsonl").read_text().splitlines()],
            "server_events": [json.loads(line) for line in (path / "server_events.jsonl").read_text().splitlines()],
        }
    checks = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    for name, item in attempts.items():
        calls = item["calls"]
        compound = len(calls) == 2 and all(call["success"] and len(call["stages"]) == 6
            and [len(stage["requests"]) for stage in call["stages"]] == [1, 2, 1, 1, 1, 1]
            and all(request["success"] for stage in call["stages"] for request in stage["requests"])
            for call in calls)
        check(f"compound_{name}", compound, "warmup and main each 6 stages/7 calls", "complete")
        check(f"controller_{name}", item["receipt"]["failure"] is None and item["receipt"]["client_returncode"] == 0,
              {"failure": item["receipt"]["failure"], "client": item["receipt"]["client_returncode"]}, "success")
        check(f"backend_release_{name}", item["executor"]["tracked_block_owners"] == 0
              and item["executor"]["backend"].get("evictions") == 0,
              {"owners": item["executor"]["tracked_block_owners"], "evictions": item["executor"]["backend"].get("evictions")},
              "no owners or autonomous evictions")

    b_policy = attempts["BC_C_01"]["server"].get("v2_policy")
    events = b_policy["events"]
    candidates = [event for event in events if event["kind"] == "v2_candidate"]
    resets = [event for event in events if event["kind"] == "v2_chain_reset"]
    holds = [event for event in events if event["kind"] == "v2_hold_same_collection"]
    releases = [event for event in events if event["kind"] == "v2_release_same_collection"]
    finishes = [event for event in events if event["kind"] == "v2_finish"]
    check("C_v2_lifecycle", len(resets) == 1 and len(holds) == 2 and len(releases) == 2 and len(finishes) == 14,
          {"resets": len(resets), "holds": len(holds), "releases": len(releases), "finishes": len(finishes)},
          {"resets": 1, "holds": 2, "releases": 2, "finishes": 14})
    accepted = sum(bool(event["audit"].get("accepted")) for event in candidates)
    fallback_counts = {}
    for event in candidates:
        key = str(event["audit"].get("fallback"))
        fallback_counts[key] = fallback_counts.get(key, 0) + 1
    check("C_v2_fallback_accounted", len(candidates) > 0 and accepted == 0,
          {"candidates": len(candidates), "accepted": accepted, "fallbacks": fallback_counts},
          "all candidates accounted; no accepted model action in this public chain")

    prompt_hashes = {}
    for name, item in attempts.items():
        first = {}
        for event in item["executor_events"]:
            if event["kind"] == "execute_model":
                for group in event["groups"]:
                    first.setdefault(group["request_id"], group["prompt_sha256"])
        assigns = [event for event in item["server_events"] if event["kind"] == "sequence_group_assign"]
        prompt_hashes[name] = [first.get(event["request_uuid"]) for event in assigns]
        check(f"prompt_hashes_{name}", len(prompt_hashes[name]) == 14 and all(prompt_hashes[name]),
              len(prompt_hashes[name]), 14)
    check("BC_identical_public_prompts", prompt_hashes["BC_B_01"] == prompt_hashes["BC_C_01"],
          "14/14 prompt hashes match", "exact match")

    strict = {}
    goodput = {}
    latencies = {}
    for name, item in attempts.items():
        main = item["calls"][1]
        strict[name] = main["success"] and main["total_latency"] <= 120.0
        expected = 7 if strict[name] else 0
        goodput[name] = expected / item["metrics"]["duration"]
        check(f"strict_goodput_{name}", item["metrics"]["slo_met_collections"] == int(strict[name])
              and math.isclose(item["metrics"]["slo_goodput"], goodput[name], rel_tol=1e-9, abs_tol=1e-9),
              {"strict": strict[name], "official": item["metrics"]["slo_goodput"], "recomputed": goodput[name]},
              "official strict SLO and goodput match")
        latencies[name] = [request["request_latency"] for stage in main["stages"] for request in stage["requests"]]
    gained = [0] if strict["BC_C_01"] and not strict["BC_B_01"] else []
    lost = [0] if strict["BC_B_01"] and not strict["BC_C_01"] else []
    check("same_ID_ledger", not gained and not lost, {"gained": gained, "lost": lost}, "no chain gained or lost; both miss")

    b_latency = attempts["BC_B_01"]["calls"][1]["total_latency"]
    c_latency = attempts["BC_C_01"]["calls"][1]["total_latency"]
    jct_relative = (c_latency - b_latency) / b_latency
    b_engine = attempts["BC_B_01"]["server"]["scheduler_stats"]["engine_step_wall_seconds"]
    c_engine = attempts["BC_C_01"]["server"]["scheduler_stats"]["engine_step_wall_seconds"]
    output_equal = attempts["BC_B_01"]["metrics"]["total_output_tokens"] == attempts["BC_C_01"]["metrics"]["total_output_tokens"]
    mechanism_gain = strict["BC_C_01"] and not strict["BC_B_01"]
    check("no_C_over_B_gain", not mechanism_gain and c_latency >= b_latency and accepted == 0,
          {"strict_B": strict["BC_B_01"], "strict_C": strict["BC_C_01"], "C_over_B": mechanism_gain,
           "JCT_relative": jct_relative, "accepted_v2_actions": accepted},
          "no C-over-B SLO or latency gain and no accepted model action")

    official = json.loads((root / "manifests/official_source_hashes.json").read_text())
    source_entries = [{"path": path, **value} for path, value in official["files"].items()]
    source_bad = []
    for item in source_entries:
        path = root / "repo" / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            source_bad.append(item["path"])
    check("official_source_1297", not source_bad, {"matched": 1297 - len(source_bad), "bad": source_bad[:10]}, "all 1297 match")
    launch = json.loads((root / "stage3/manifests/launch.json").read_text())
    launch_bad = []
    for group in ["programs", "backend", "v2_src", "contracts"]:
        for item in launch.get(group, []):
            path = Path(item["path"])
            if not path.is_file() or sha(path) != item["sha256"]:
                launch_bad.append(str(path))
    check("stage3_launch_sha", not launch_bad, {"bad": launch_bad[:10]}, "all stage3 launch files match")
    registered = {str(process["pid"]) for item in attempts.values() for process in item["receipt"]["processes"]}
    gpu_processes = subprocess.check_output(["nvidia-smi", "-i", "2", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True).strip()
    own_remaining = [line for line in gpu_processes.splitlines() if line.split(",", 1)[0].strip() in registered]
    gpu = subprocess.check_output(["nvidia-smi", "-i", "2", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"], text=True).strip()
    check("GPU2_release", not own_remaining and all(item["receipt"]["gpu_released"] for item in attempts.values()),
          {"gpu": gpu, "own_remaining": own_remaining, "all_gpu_processes": gpu_processes}, "no study-owned process; foreign processes untouched")

    valid = all(item["pass"] for item in checks.values())
    result = {
        "comparison_valid": valid,
        "decision": "STOP_NO_C_OVER_B_MECHANISM_SIGNAL",
        "checks": checks,
        "strict_slo": strict,
        "same_ID_gained": gained,
        "same_ID_lost": lost,
        "native_goodput": goodput,
        "JCT": {"B": b_latency, "C": c_latency, "C_over_B_relative": jct_relative},
        "engine_step_seconds": {"B": b_engine, "C": c_engine, "C_over_B": c_engine / b_engine},
        "request_latency_P90_P95": {
            "B": [percentile(latencies["BC_B_01"], 90), percentile(latencies["BC_B_01"], 95)],
            "C": [percentile(latencies["BC_C_01"], 90), percentile(latencies["BC_C_01"], 95)],
        },
        "output_tokens_equal": output_equal,
        "v2_candidates": {"total": len(candidates), "accepted": accepted, "fallbacks": fallback_counts},
        "formal_matrix_allowed": False,
        "HOLDOUT_allowed": False,
        "interpretation": "On this public chain, the frozen v2 accepts no model action and C is slower than full JITServe on the same backend. This gives no basis for a forward-looking mechanism claim or a formal mechanism matrix. Predictor support exhaustion is reported explicitly and is not relabeled as a general falsification of lookahead.",
        "next_step": "Stop and report; do not repair the predictor, expand the matrix or open HOLDOUT automatically.",
    }
    dump(root / "stage3/receipts/SMALL_BC_COMPARISON.json", result)
    print(json.dumps({"comparison_valid": valid, "decision": result["decision"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "JCT": result["JCT"], "v2_candidates": result["v2_candidates"],
                      "receipt": str(root / "stage3/receipts/SMALL_BC_COMPARISON.json")}, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
