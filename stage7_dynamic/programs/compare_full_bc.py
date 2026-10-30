import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    names = {"B": sys.argv[2], "C": sys.argv[3]}
    smoke = json.loads((root / "stage7_dynamic/FULL_FREEZE.json").read_text())
    attempts = {}
    for arm, name in names.items():
        path = root / "stage7_dynamic/attempts" / name
        attempts[arm] = {
            "calls": json.loads((path / "client_calls.json").read_text()),
            "metrics": json.loads((path / "official_metrics.json").read_text()),
            "receipt": json.loads((path / "controller_receipt.json").read_text()),
            "server": json.loads((path / "server_audit_summary.json").read_text()),
            "executor": json.loads((path / "executor_summary.json").read_text()),
            "executor_events": [json.loads(line) for line in
                                (path / "executor_events.jsonl").read_text().splitlines()],
        }

    checks = {}
    stage_counts = {int(key): value for key, value in smoke["selection"]["stage_counts"].items()}
    requests_per_stage = {int(key): value for key, value in
                          smoke["selection"]["requests_per_stage"].items()}
    target_arrivals = smoke["generation_and_deadline"]["target_main_arrival_offsets_seconds"]

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    def complete_call(call, collection):
        return (call.get("success") and len(call.get("stages", [])) == stage_counts[collection]
                and [len(stage["requests"]) for stage in call["stages"]] == requests_per_stage[collection]
                and all(stage["success"] for stage in call["stages"])
                and all(request["success"] for stage in call["stages"]
                        for request in stage["requests"]))

    main_by_arm = {}
    arrivals_by_arm = {}
    for arm, item in attempts.items():
        calls = item["calls"]
        warmup = [call for call in calls if call.get("phase") == "warmup"]
        main = [call for call in calls if call.get("phase") == "main"]
        ids = [call.get("collection_id") for call in main]
        main_by_arm[arm] = {call["collection_id"]: call for call in main}
        complete = (len(warmup) == 1 and warmup[0].get("collection_id") == 0
                    and complete_call(warmup[0], 0) and len(main) == 6
                    and set(ids) == set(range(6)) and len(set(ids)) == 6
                    and all(complete_call(call, call["collection_id"]) for call in main))
        check(f"complete_warmup_and_40_main_calls_{arm}", complete,
              {"calls": len(calls), "warmup": len(warmup), "main": len(main), "IDs": ids},
              "one complete chain0 warmup and six complete main chains with 40 calls")
        check(f"controller_{arm}",
              item["receipt"]["failure"] is None and item["receipt"]["client_returncode"] == 0,
              {"failure": item["receipt"]["failure"], "client": item["receipt"]["client_returncode"]},
              "success")
        check(f"backend_release_{arm}",
              item["executor"]["tracked_block_owners"] == 0
              and item["executor"]["backend"].get("evictions", 0) == 0,
              {"owners": item["executor"]["tracked_block_owners"],
               "evictions": item["executor"]["backend"].get("evictions", 0)},
              "no owners or autonomous evictions")
        arrivals = {call["collection_id"]: call.get("arrival_offset_seconds") for call in main}
        arrivals_by_arm[arm] = arrivals
        errors = [abs(arrivals[index] - target_arrivals[index]) for index in range(6)
                  if arrivals.get(index) is not None]
        check(f"arrival_contract_{arm}", len(errors) == 6 and max(errors) <= 0.5,
              {"actual": arrivals, "target": target_arrivals,
               "max_error": max(errors) if errors else None},
              "all six arrivals within 0.5 seconds of the frozen seed-42 plan")
        sizes = [len(event["groups"]) for event in item["executor_events"]
                 if event["kind"] == "execute_model"]
        check(f"dynamic_batch_accounting_{arm}", sizes and max(sizes) > 4,
              {"batches": len(sizes), "max_batch": max(sizes) if sizes else None,
               "over_four": sum(size > 4 for size in sizes)},
              "full B and C both execute real batches above the removed four-request limit")

    arrival_delta = max(abs(arrivals_by_arm["B"][index] - arrivals_by_arm["C"][index])
                        for index in range(6))
    check("matched_arrivals_across_arms", arrival_delta <= 0.5, arrival_delta,
          "same-ID arrival offsets differ by at most 0.5 seconds")

    strict = {}
    scaled = {}
    native_goodput = {}
    strict_call_goodput = {}
    strict_fraction = {}
    jct = {}
    request_latencies = {}
    generation_signatures = {"B": {}, "C": {}}
    for arm, item in attempts.items():
        strict[arm] = {}
        scaled[arm] = {}
        jct[arm] = {}
        request_latencies[arm] = []
        duration = item["metrics"]["duration"]
        native_numerator = 0
        strict_numerator = 0
        for collection, call in main_by_arm[arm].items():
            latency = call["total_latency"]
            jct[arm][collection] = latency
            strict[arm][collection] = bool(call["success"] and latency <= 20 * stage_counts[collection])
            scaled[arm][collection] = bool(
                call["success"] and latency <= 20 * (collection % 4 + 1) * stage_counts[collection])
            calls = sum(requests_per_stage[collection])
            native_numerator += calls if scaled[arm][collection] else 0
            strict_numerator += calls if strict[arm][collection] else 0
            for stage in call["stages"]:
                for request in stage["requests"]:
                    request_latencies[arm].append(request["request_latency"])
                    key = f"{collection}:{stage['stage_id']}:{request['request_id']}"
                    generation_signatures[arm][key] = [len(request["request_tbt"]), request["stop_reason"]]
        native_goodput[arm] = native_numerator / duration
        strict_call_goodput[arm] = strict_numerator / duration
        strict_fraction[arm] = sum(strict[arm].values()) / 6
        check(f"official_native_goodput_{arm}",
              item["metrics"]["slo_met_collections"] == sum(scaled[arm].values())
              and math.isclose(item["metrics"]["slo_goodput"], native_goodput[arm],
                               rel_tol=1e-9, abs_tol=1e-9),
              {"official_scaled_collections": item["metrics"]["slo_met_collections"],
               "recomputed_scaled_collections": sum(scaled[arm].values()),
               "official_goodput": item["metrics"]["slo_goodput"],
               "recomputed_goodput": native_goodput[arm]},
              "official scaled-SLO collection count and native request goodput reproduced")

    generation_mismatches = [
        {"request": key, "B": generation_signatures["B"].get(key),
         "C": generation_signatures["C"].get(key)}
        for key in sorted(set(generation_signatures["B"]) | set(generation_signatures["C"]))
        if generation_signatures["B"].get(key) != generation_signatures["C"].get(key)]
    check("same_output_length_and_stop_reason", not generation_mismatches,
          generation_mismatches[:20], "all same-ID requests have equal output-token counts and stop reasons")

    policy = attempts["C"]["server"]["v2_policy"]
    events = policy["events"]
    candidates = [event for event in events if event["kind"] == "v2_candidate"]
    deadline_bad = []
    for event in candidates:
        for goal in event["view"]["goals"]:
            collection = goal["program"]
            expected = 20 * (collection % 4 + 1) * stage_counts[collection]
            actual = goal["deadline"] - goal["arrival"]
            if not math.isclose(actual, expected, abs_tol=1e-8):
                deadline_bad.append({"collection": collection, "actual": actual, "expected": expected})
    check("C_received_SLO_and_stage_count_deadline", not deadline_bad, deadline_bad[:20],
          "each chain goal uses the received scaled request SLO times its actual stage count")

    accepted = sum(bool(event["audit"].get("accepted")) for event in candidates)
    distinguished = [event for event in candidates
                     if len(event["view"]["goals"]) >= 2
                     and event["audit"].get("candidate_discriminability") == "distinguished"]
    over_capacity = [event for event in distinguished
                     if sum(len(members) for members in event["compound_members"].values()) > 4]
    check("supported_multi_goal_discrimination", bool(distinguished),
          {"distinguished_multi_goal": len(distinguished),
           "over_capacity_distinguished": len(over_capacity)},
          "at least one distinguishable multi-goal candidate")
    check("over_capacity_candidate", bool(over_capacity), len(over_capacity),
          "at least one distinguishable candidate above four real request slots")

    acceptances = [event for event in events if event["kind"] == "v2_action_accepted"]
    schedules = [event for event in events if event["kind"] == "v2_action_scheduled"]
    executions = [event for event in events if event["kind"] == "v2_action_executed"]
    accepted_ids = {event["id"] for event in acceptances}
    scheduled_ids = {event["id"] for event in schedules}
    executed_ids = {event["id"] for event in executions}
    executed = len(accepted_ids & scheduled_ids & executed_ids)
    execution_witnesses = [{"id": event["id"], "anchor": event["anchor"],
                            "selected": event.get("selected", event.get("actual_request_ids"))}
                           for event in executions if event["id"] in accepted_ids]
    projection_matches = sum(event.get("matches_projection") is True for event in schedules)
    completed_leases = [event for event in events if event["kind"] == "v2_lease_completed"
                        and event["id"] in accepted_ids and event.get("lease_first_matches") is True]
    forced_not_scheduled = [event for event in events if event["kind"] == "v2_forced_not_scheduled"]
    check("accepted_model_action_executed",
          accepted == 0 or (accepted == len(acceptances) and executed > 0 and projection_matches > 0),
          {"accepted": accepted, "executed": executed,
           "projection_matches": projection_matches, "witnesses": execution_witnesses[:10]},
          "zero actions is a valid conservative decision; accepted actions need matching execution")
    check("complete_lease_and_no_forced_spin",
          (accepted == 0 or bool(completed_leases)) and not forced_not_scheduled,
          {"completed_leases": len(completed_leases),
           "forced_not_scheduled": len(forced_not_scheduled)},
          "accepted actions need a complete matching lease; no forced action may fail its own admission")

    resets = sum(event["kind"] == "v2_chain_reset" for event in events)
    holds = sum(event["kind"] == "v2_hold_same_collection" for event in events)
    check("compound_reset_and_no_hold", resets == 1 and holds == 0,
          {"resets": resets, "holds": holds}, {"resets": 1, "holds": 0})
    fallback_counts = {}
    for event in candidates:
        key = str(event["audit"].get("fallback"))
        fallback_counts[key] = fallback_counts.get(key, 0) + 1

    gained = [index for index in range(6) if strict["C"][index] and not strict["B"][index]]
    lost = [index for index in range(6) if strict["B"][index] and not strict["C"][index]]
    scaled_gained = [index for index in range(6) if scaled["C"][index] and not scaled["B"][index]]
    scaled_lost = [index for index in range(6) if scaled["B"][index] and not scaled["C"][index]]

    launch = json.loads((root / smoke["frozen_mechanism"]["qualification_receipt"]).read_text())
    launch_bad = [rel for rel, expected in launch["files"].items()
                  if not (root / rel).is_file() or sha(root / rel) != expected]
    check("stage7_launch_sha", not launch_bad, {"bad": launch_bad}, "all full-B/C stage7 files match")
    identities = {arm: item["server"].get("arm_identity") for arm, item in attempts.items()}
    check("full_dynamic_arm_identity",
          identities == {"B": "B-full", "C": "C-full-dynamic"}
          and all(item["server"].get("use_prediction") and item["server"].get("use_graph_matching")
                  for item in attempts.values()),
          identities,
          "B-full and C-full-dynamic both retain official prediction and graph matching")
    dynamic_observations = attempts["C"]["server"]["scheduler_stats"].get(
        "dynamic_performance_observations", 0)
    dynamic_factors = [int(key) for key in attempts["C"]["server"].get("performance_factors", {})]
    check("dynamic_performance_calibration",
          dynamic_observations > 0 and any(key > 4 for key in dynamic_factors),
          {"observations": dynamic_observations, "factor_keys": dynamic_factors},
          "C records real B>4 decode observations and uses them in the performance model")
    official = json.loads((root / "manifests/official_source_hashes.json").read_text())
    official_bad = [rel for rel, item in official["files"].items()
                    if not (root / "repo" / rel).is_file() or sha(root / "repo" / rel) != item["sha256"]]
    check("official_source_1297", not official_bad, {"bad": official_bad[:10]}, "all 1297 match")

    registered = {str(process["pid"]) for item in attempts.values()
                  for process in item["receipt"]["processes"]}
    gpu_processes = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"
    ], text=True).strip()
    own_remaining = [line for line in gpu_processes.splitlines()
                     if line.split(",", 1)[0].strip() in registered]
    gpu = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"
    ], text=True).strip()
    check("GPU0_release", not own_remaining and all(item["receipt"]["gpu_released"] for item in attempts.values()),
          {"gpu": gpu, "own_remaining": own_remaining, "all_gpu_processes": gpu_processes},
          "no study-owned process; foreign processes untouched")

    valid = all(item["pass"] for item in checks.values())
    attained = {arm: sum(strict[arm].values()) for arm in ("B", "C")}
    scaled_attained = {arm: sum(scaled[arm].values()) for arm in ("B", "C")}
    mean_jct = {arm: math.fsum(jct[arm].values()) / len(jct[arm]) for arm in ("B", "C")}
    won = scaled_attained["C"] > scaled_attained["B"]
    if not valid:
        decision = "STOP_MECHANISM_OR_ACCOUNTING_NOT_QUALIFIED"
    elif won:
        decision = "SMALL_FULL_DYNAMIC_C_WIN"
    else:
        decision = "STOP_SLO_CRITERION_NOT_MET"
    result = {
        "schema": "small-full-dynamic-bc-comparison/v3",
        "comparison_valid": valid,
        "mechanism_won": won,
        "win_criterion": {
            "official_scaled_attained": scaled_attained,
            "native_request_goodput": native_goodput,
            "mean_JCT": mean_jct,
            "rule": "C official scaled attained > B; JCT, goodput and overheads are reported without a performance threshold",
        },
        "decision": decision,
        "attempts": names,
        "checks": checks,
        "primary_unscaled_strict": {
            "attained": attained,
            "fraction": strict_fraction,
            "per_chain": strict,
            "gained": gained,
            "lost": lost,
            "call_goodput": strict_call_goodput,
        },
        "official_scaled_slo": {
            "per_chain": scaled,
            "gained": scaled_gained,
            "lost": scaled_lost,
            "native_request_goodput": native_goodput,
        },
        "JCT": {
            "per_chain": jct,
            "mean": mean_jct,
            "C_over_B_relative": (mean_jct["C"] - mean_jct["B"]) / mean_jct["B"],
            "P90": {arm: percentile(list(jct[arm].values()), 90) for arm in ("B", "C")},
            "P95": {arm: percentile(list(jct[arm].values()), 95) for arm in ("B", "C")},
        },
        "request_latency_P90_P95": {
            arm: [percentile(request_latencies[arm], 90), percentile(request_latencies[arm], 95)]
            for arm in ("B", "C")
        },
        "candidates": {
            "total": len(candidates), "accepted": accepted, "executed": executed,
            "distinguished_multi_goal": len(distinguished),
            "over_capacity_distinguished": len(over_capacity),
            "fallbacks": fallback_counts,
        },
        "engine": {arm: attempts[arm]["server"]["scheduler_stats"] for arm in ("B", "C")},
        "kv": {arm: {key: attempts[arm]["executor"]["backend"].get(key) for key in
                     ["actual_forward_query_tokens", "necessary_prefill_tokens", "productive_decode_queries",
                      "recompute_tokens", "swap_bytes_in", "swap_bytes_out"]}
               for arm in ("B", "C")},
        "scope": "One six-chain full-dynamic B/C development pair; complete official policy semantics in the shared backend, not statistical superiority evidence",
        "ablation_allowed": False,
        "formal_matrix_allowed": False,
        "holdout_allowed": False,
    }
    receipt = root / "stage7_dynamic/receipts" / f"FULL_DYNAMIC_COMPARISON_{names['B']}_{names['C']}.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"comparison_valid": valid, "decision": decision,
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "official_scaled_attained": scaled_attained,
                      "mean_JCT": mean_jct,
                      "candidates": result["candidates"], "receipt": str(receipt)}, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
