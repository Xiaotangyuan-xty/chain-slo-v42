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
    for name in ["COMPOUND_B_GPU0_01", "COMPOUND_C_GPU0_01"]:
        path = root / "stage3_gpu0/attempts" / name
        attempts[name] = {
            "calls": json.loads((path / "client_calls.json").read_text()),
            "metrics": json.loads((path / "official_metrics.json").read_text()),
            "receipt": json.loads((path / "controller_receipt.json").read_text()),
            "server": json.loads((path / "server_audit_summary.json").read_text()),
            "executor": json.loads((path / "executor_summary.json").read_text()),
            "executor_events": [json.loads(line) for line in (path / "executor_events.jsonl").read_text().splitlines()],
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

    policy = attempts["COMPOUND_C_GPU0_01"]["server"]["v2_policy"]
    events = policy["events"]
    candidates = [event for event in events if event["kind"] == "v2_candidate"]
    spans = [goal["deadline"] - goal["arrival"] for event in candidates for goal in event["view"]["goals"]]
    first_past = sum(goal["deadline"] <= candidates[0]["view"]["now"] for goal in candidates[0]["view"]["goals"])
    check("absolute_deadline_contract", all(span == 120.0 for span in spans) and first_past == 0,
          {"span_unique": sorted(set(spans)), "first_candidate_past": first_past},
          "all spans 120 seconds and first candidate entirely in the future")
    accepted = sum(bool(event["audit"].get("accepted")) for event in candidates)
    fallback_counts = {}
    for event in candidates:
        key = str(event["audit"].get("fallback"))
        fallback_counts[key] = fallback_counts.get(key, 0) + 1
    support_gates = [event for event in events if event["kind"] == "v2_support_gate"]
    check("support_gate_cache", len(support_gates) < 20 and policy["support_cache"],
          {"gate_events": len(support_gates), "cache": policy["support_cache"]},
          "bounded support-gate events with cached support state")
    resets = sum(event["kind"] == "v2_chain_reset" for event in events)
    holds = sum(event["kind"] == "v2_hold_same_collection" for event in events)
    check("compound_reset_and_no_hold", resets == 1 and holds == 0,
          {"resets": resets, "holds": holds}, {"resets": 1, "holds": 0})

    histograms = {}
    for name, item in attempts.items():
        histograms[name] = {}
        for event in item["executor_events"]:
            if event["kind"] == "execute_model":
                size = len(event["groups"])
                histograms[name][size] = histograms[name].get(size, 0) + 1
    check("compound_batching_restored", histograms["COMPOUND_C_GPU0_01"].get(2) == 1362
          and histograms["COMPOUND_B_GPU0_01"].get(2) == 1362,
          histograms, "B and C both have 1362 two-request batches")
    steps = {name: item["server"]["scheduler_stats"]["engine_steps"] for name, item in attempts.items()}
    check("engine_steps_conserved_within_one_prefill_chunk", abs(steps["COMPOUND_B_GPU0_01"] - steps["COMPOUND_C_GPU0_01"]) <= 1,
          steps, "engine steps differ by at most one prefill chunk")
    for name, item in attempts.items():
        backend = item["executor"]["backend"]
        check(f"kv_work_conserved_{name}", backend["actual_forward_query_tokens"] == 33770
              and backend["necessary_prefill_tokens"] == 25956 and backend["recompute_tokens"] == 0
              and backend["swap_bytes_in"] == 0 and backend["swap_bytes_out"] == 0,
              {key: backend[key] for key in ["actual_forward_query_tokens", "necessary_prefill_tokens", "recompute_tokens", "swap_bytes_in", "swap_bytes_out"]},
              "33770/25956 tokens and no recompute or swap")

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
    gained = [0] if strict["COMPOUND_C_GPU0_01"] and not strict["COMPOUND_B_GPU0_01"] else []
    lost = [0] if strict["COMPOUND_B_GPU0_01"] and not strict["COMPOUND_C_GPU0_01"] else []
    check("same_ID_ledger", not gained and not lost, {"gained": gained, "lost": lost}, "no chain gained or lost; both miss")

    b_latency = attempts["COMPOUND_B_GPU0_01"]["calls"][1]["total_latency"]
    c_latency = attempts["COMPOUND_C_GPU0_01"]["calls"][1]["total_latency"]
    relative = (c_latency - b_latency) / b_latency
    b_engine = attempts["COMPOUND_B_GPU0_01"]["server"]["scheduler_stats"]["engine_step_wall_seconds"]
    c_engine = attempts["COMPOUND_C_GPU0_01"]["server"]["scheduler_stats"]["engine_step_wall_seconds"]
    mechanism_gain = strict["COMPOUND_C_GPU0_01"] and not strict["COMPOUND_B_GPU0_01"]
    check("no_model_gain", not mechanism_gain and accepted == 0,
          {"strict_B": strict["COMPOUND_B_GPU0_01"], "strict_C": strict["COMPOUND_C_GPU0_01"],
           "C_over_B": mechanism_gain, "accepted_v2_actions": accepted},
          "no accepted model action and no C-over-B SLO gain")

    official = json.loads((root / "manifests/official_source_hashes.json").read_text())
    source_bad = []
    for rel, item in official["files"].items():
        path = root / "repo" / rel
        if not path.is_file() or sha(path) != item["sha256"]:
            source_bad.append(rel)
    check("official_source_1297", not source_bad, {"bad": source_bad[:10]}, "all 1297 match")
    launch = json.loads((root / "stage3_gpu0/manifests/launch.json").read_text())
    launch_bad = []
    for group in ["programs", "backend", "v2_src", "contracts"]:
        for item in launch.get(group, []):
            path = Path(item["path"])
            if not path.is_file() or sha(path) != item["sha256"]:
                launch_bad.append(str(path))
    check("stage3_gpu0_launch_sha", not launch_bad, {"bad": launch_bad[:10]}, "all GPU0 launch files match")
    registered = {str(process["pid"]) for item in attempts.values() for process in item["receipt"]["processes"]}
    gpu_processes = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True).strip()
    own_remaining = [line for line in gpu_processes.splitlines() if line.split(",", 1)[0].strip() in registered]
    gpu = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"], text=True).strip()
    check("GPU0_release", not own_remaining and all(item["receipt"]["gpu_released"] for item in attempts.values()),
          {"gpu": gpu, "own_remaining": own_remaining, "all_gpu_processes": gpu_processes}, "no study-owned process; foreign processes untouched")

    valid = all(item["pass"] for item in checks.values())
    result = {
        "comparison_valid": valid,
        "decision": "STOP_NO_ACCEPTED_MODEL_ACTION",
        "checks": checks,
        "strict_slo": strict,
        "same_ID_gained": gained,
        "same_ID_lost": lost,
        "native_goodput": goodput,
        "JCT": {"B": b_latency, "C": c_latency, "C_over_B_relative": relative},
        "engine_step_seconds": {"B": b_engine, "C": c_engine, "C_over_B": c_engine / b_engine},
        "request_latency_P90_P95": {
            "B": [percentile(latencies["COMPOUND_B_GPU0_01"], 90), percentile(latencies["COMPOUND_B_GPU0_01"], 95)],
            "C": [percentile(latencies["COMPOUND_C_GPU0_01"], 90), percentile(latencies["COMPOUND_C_GPU0_01"], 95)],
        },
        "v2_candidates": {"total": len(candidates), "accepted": accepted, "fallbacks": fallback_counts},
        "repair_vs_previous_C": {
            "previous_C_JCT": 167.47548812814057,
            "current_C_JCT": c_latency,
            "previous_C_over_B_relative": 0.15649832952494885,
            "current_C_over_B_relative": relative,
        },
        "formal_matrix_allowed": False,
        "HOLDOUT_allowed": False,
        "interpretation": "The P0-P2 repairs restored deadline and compound execution and removed the serialization collapse. However, no model action was accepted and there is no C-over-B SLO gain. The implementation repair is validated, but a mechanism advantage is not.",
        "next_step": "Stop and report; do not retune the predictor, expand the matrix or open HOLDOUT automatically.",
    }
    dump(root / "stage3_gpu0/receipts/COMPOUND_BC_COMPARISON.json", result)
    print(json.dumps({"comparison_valid": valid, "decision": result["decision"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "JCT": result["JCT"], "v2_candidates": result["v2_candidates"],
                      "receipt": str(root / "stage3_gpu0/receipts/COMPOUND_BC_COMPARISON.json")}, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
