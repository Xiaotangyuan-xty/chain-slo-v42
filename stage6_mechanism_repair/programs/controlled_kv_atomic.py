import json
import os
from pathlib import Path
import sys
import traceback

from controlled_scheduler import add, make_engine, public_infos, read_batches, selector
from vllm.core.scheduler import PreemptionMode
from v2_predictor import Predictor


def long_predictor():
    records = ((0, 48, 0.0, 15.0, (10.0, 5.0)),)
    groups = tuple((key, records) for key in ((22, 0), (21, 0), (19, 0), (2, 0)))
    return Predictor(groups, 0.0, ())


def run_scenario(root, output, recompute):
    infos = public_infos(root)
    engine, scheduler = make_engine(root, output, {index: 1 for index in range(4)},
                                    "0:stage:0", 50, pred=long_predictor())
    scheduler.policy.candidate_selector = None
    details = {}
    try:
        add(engine, infos, "long-a", 0, 0, 2933, slo=120.0)
        add(engine, infos, "long-b", 1, 0, 2709, slo=120.0)
        for _ in range(20):
            engine.engine.step()
            if all(not group.is_prefill() for group in scheduler.running):
                break
        groups = {group.request_id: group for group in scheduler.running}
        if set(groups) != {"long-a", "long-b"}:
            raise RuntimeError(f"unexpected initial running state: {list(groups)}")
        for group in list(groups.values()):
            mode = scheduler._preempt(group, [], PreemptionMode.SWAP)
            if mode != PreemptionMode.SWAP:
                raise RuntimeError(f"setup could not swap {group.request_id}: {mode}")
            if group in scheduler.running:
                scheduler.running.remove(group)
            scheduler.swapped.append(group)
            engine.engine.model_executor._swap_out(group.request_id)
            engine.engine.model_executor.block_owners[group.request_id].clear()

        scheduler.policy.batch_slots = 2
        scheduler.scheduler_config.max_num_seqs = 2
        add(engine, infos, "retained-c", 2, 0, 2500, slo=120.0)
        add(engine, infos, "retained-d", 3, 0, 256, slo=120.0)
        for name in ("retained-c", "retained-d"):
            group = next(item for item in scheduler.waiting if item.request_id == name)
            scheduler.waiting.remove(group)
            scheduler._allocate_and_set_running(group)
            scheduler._add_seq_group_to_running(group)
        for _ in range(20):
            engine.engine.step()
            current = [group for group in scheduler.running if group.request_id.startswith("retained")]
            if len(current) == 2 and all(not group.is_prefill() for group in current):
                break
        groups = {group.request_id: group for group in
                  list(scheduler.running) + list(scheduler.waiting) + list(scheduler.swapped)}
        if set(groups) != {"long-a", "long-b", "retained-c", "retained-d"}:
            raise RuntimeError(f"unexpected retained state: {list(groups)}")
        if (set(scheduler.swapped) != {groups["long-a"], groups["long-b"]}
                or set(scheduler.running) != {groups["retained-c"], groups["retained-d"]}):
            raise RuntimeError("long pair must be swapped while both retained requests stay running")
        if recompute:
            scheduler.user_specified_preemption_mode = "recompute"
            scheduler.block_manager.can_swap_out = lambda group: False
            scheduler.policy.batch_slots = 2
        else:
            scheduler.policy.batch_slots = 4
        scheduler.scheduler_config.max_num_seqs = scheduler.policy.batch_slots
        ordered = [groups[name] for name in ("long-a", "long-b", "retained-c", "retained-d")]
        original_dispatch = scheduler.policy.dispatch_order
        scheduler.policy.dispatch_order = lambda candidates: list(ordered)
        scheduler.policy.candidate_selector = selector("0:stage:0")
        scheduler.policy.remaining = 0.0
        initial_free = scheduler.block_manager.get_num_free_gpu_blocks()
        ledger_free = scheduler.v2_ledger.view().free
        details.update(initial_free_blocks=initial_free, initial_backend_free_tokens=ledger_free,
                       block_size=scheduler.cache_config.block_size)
        exception = None
        try:
            engine.engine.step()
        except Exception as error:
            exception = f"{type(error).__name__}: {error}"
        details["exception"] = exception
        details["plan"] = scheduler.compound_events[-1] if scheduler.compound_events else None
        details["batch"] = read_batches(output)[-1]
        details["backend_free_after"] = scheduler.v2_ledger.view().free
        scheduler.policy.dispatch_order = original_dispatch
        scheduler.policy.candidate_selector = None
        scheduler.user_specified_preemption_mode = None
        scheduler.policy.batch_slots = 4
        scheduler.scheduler_config.max_num_seqs = 4
        scheduler.policy.remaining = 0.0
        steps = 0
        while engine.engine.has_unfinished_requests() and steps < 4000:
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        details.update(completion_steps=steps,
                       unfinished=engine.engine.has_unfinished_requests(),
                       batches=read_batches(output),
                       backend_events=[event for event in engine.engine.model_executor.events
                                       if event["kind"] == "backend_release"])
    finally:
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()
    details["summary"] = json.loads((output / "executor_summary.json").read_text())
    return details


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only KV atomicity gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    for name, recompute in (("swapped_long", False), ("recompute_long", True)):
        try:
            result = run_scenario(root, root / "stage6_mechanism_repair/controlled" / attempt / name,
                                  recompute)
            details[name] = result
            check(f"{name}_no_kv_ledger_divergence", result["exception"] is None,
                  result["exception"], "no scheduler/backend KV divergence")
            check(f"{name}_long_pair_not_jointly_overadmitted",
                  not {"long-a", "long-b"} <= set(result["batch"]),
                  {"batch": result["batch"], "initial_free_blocks": result["initial_free_blocks"]},
                  "two long swapped prefills are not jointly restored when their sum exceeds free KV")
            check(f"{name}_all_complete_and_released",
                  not result["unfinished"] and result["completion_steps"] < 4000
                  and result["summary"]["tracked_block_owners"] == 0,
                  {"unfinished": result["unfinished"], "steps": result["completion_steps"],
                   "owners": result["summary"]["tracked_block_owners"]},
                  "deferred work eventually completes without KV leaks")
            if recompute:
                check("recompute_invalidates_backend_before_restore",
                      any(event["reason"] == "scheduler_recompute"
                          for event in result["backend_events"]),
                      result["backend_events"],
                      "RECOMPUTE synchronously releases backend state before later restoration")
            else:
                plan = result["plan"] or {}
                check("swapped_long_plan_accounts_real_free_and_retained",
                      "initial_available_blocks" in plan and "remaining_available_blocks" in plan
                      and plan["remaining_available_blocks"] >= 0
                      and any("long-b" in item[0] and item[1] == "kv"
                              for item in plan.get("deferred", [])),
                      plan,
                      "the admission plan, not only the backend guard, accounts for retained state and defers long-b for KV")
        except Exception as error:
            details[name] = {"exception": traceback.format_exc(limit=10)}
            check(f"{name}_controlled_completion", False,
                  f"{type(error).__name__}: {error}", "scenario completed")

    passed = all(item["pass"] for item in checks.values())
    result = {"schema": "stage9-kv-atomic-gate/v1", "attempt": attempt,
              "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
              "gpu_used": False, "next_gpu_consideration_allowed": passed}
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_KV_ATOMIC_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
