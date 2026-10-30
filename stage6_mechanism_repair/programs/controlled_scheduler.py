import asyncio
import copy
import functools
import json
import os
from pathlib import Path
import sys
import time
import traceback

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams

from fake_backend import FakeBackend
from harness_executor import HarnessAsyncExecutor, HarnessAsyncLLMEngine
from v2_predictor import Predictor
from v2_slo_scheduler import install_v2_slo


class ToyPerformance:
    decode_seconds = 0.01

    def iteration(self, batch_slots, context):
        return self.decode_seconds * max(1, batch_slots)

    def prefill(self, new, cached=0):
        return 0.0

    def recovery(self, entry, needed):
        return 0.0


def predictor():
    records = ((0, 48, 0.0, 15.0, (10.0, 5.0)),)
    groups = tuple((key, records) for key in ((2, 0), (2, 1), (4, 0), (4, 1)))
    groups += (((3, 0), ((0, 80, 0.0, 15.0, (10.0, 5.0)),)),)
    return Predictor(groups, 0.0, ())


def selector(anchor):
    def choose(view, instance, eligible, pred, perf, owners, **kwargs):
        key = anchor if anchor in eligible else sorted(eligible)[0]
        return (key, 4 * perf.decode_seconds), {
            "accepted": True, "fallback": None, "actions": [], "seconds": 0.0,
        }
    return choose


def make_engine(root, output, stage_counts, anchor, schedule_interval, num_blocks=512,
                perf=None, pred=None):
    output.mkdir(parents=True, exist_ok=False)
    prepare = json.loads((root / "PREPARE.json").read_text())
    HarnessAsyncExecutor.configure(output, FakeBackend)
    args = AsyncEngineArgs(
        model=prepare["model"], tokenizer=prepare["model"], device="cpu", dtype="bfloat16",
        scheduling_policy="jitserve", max_num_seqs=4,
        max_num_batched_tokens=min(2048, num_blocks * 16),
        max_model_len=min(8192, num_blocks * 16), num_gpu_blocks_override=num_blocks, swap_space=1,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        enforce_eager=True, disable_async_output_proc=True, disable_log_stats=True,
        search_strategy="sliding_window", top_k_selection=3, max_swaps_per_iter=3,
        seed=42,
    )
    engine = HarnessAsyncLLMEngine.from_engine_args(args, start_engine_loop=False)
    scheduler = install_v2_slo(engine, perf or ToyPerformance(), pred or predictor(),
                               stage_counts, selector(anchor))
    scheduler.policy.schedule_interval = schedule_interval
    original_execute = engine.engine.model_executor.execute_model

    @functools.wraps(original_execute)
    def timed_execute(execute_model_req):
        started = time.perf_counter()
        result = original_execute(execute_model_req=execute_model_req)
        scheduler.policy.after_model_step(time.perf_counter() - started, time.perf_counter())
        return result

    engine.engine.model_executor.execute_model = timed_execute
    return engine, scheduler


def add(engine, infos, request_id, collection, stage, prompt, deadline=120.0, slo=20.0):
    from jitserve.request_info import RequestType
    info = copy.deepcopy(infos[0])
    info.request_type = RequestType.COLLECTIVE
    info.collection_id = collection
    info.stage_id = stage
    info.deadline = deadline
    info.input_len = prompt
    info.slo_constraint = (2.0, 0.1, slo)
    info.output_len = 4
    asyncio.run(engine.engine.add_request_async(
        request_id, prompt=TokensPrompt(prompt_token_ids=[42] * prompt),
        params=SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens=4, seed=42),
        request_info=info, arrival_time=time.time(),
    ))


def public_infos(root):
    from trace.trace_deepresearch import DeepResearchTrace
    return DeepResearchTrace.load_trace(
        str(root / "repo/traces/deepresearch_filter.jsonl"))[0].to_request_infos()


def read_batches(output):
    events = [json.loads(line) for line in (output / "executor_events.jsonl").read_text().splitlines()]
    return [[group["request_id"] for group in event["groups"]]
            for event in events if event["kind"] == "execute_model"]


def run_basic(root, output, schedule_interval):
    infos = public_infos(root)
    stage_counts = {index: 1 for index in range(4)}
    engine, scheduler = make_engine(root, output, stage_counts, "1:stage:0", schedule_interval)
    try:
        for request_id, collection, prompt in [
            ("single-0", 0, 256), ("compound-1a", 1, 256), ("compound-1b", 1, 256),
            ("single-2", 2, 256), ("single-3", 3, 256),
        ]:
            add(engine, infos, request_id, collection, 0, prompt)
        steps = 0
        if schedule_interval != 50:
            scheduler.policy.schedule_interval = 50
            engine.engine.step()
            steps = 1
            if scheduler.waiting:
                group = scheduler.waiting.popleft()
                scheduler._allocate_and_set_running(group)
                scheduler._add_seq_group_to_running(group)
            scheduler.policy.schedule_interval = schedule_interval
        while engine.engine.has_unfinished_requests() and steps < 300:
            if schedule_interval != 50 and steps == 3:
                scheduler.policy.schedule_interval = 50
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        result = {
            "batches": read_batches(output),
            "policy": scheduler.policy.snapshot(),
            "steps": steps,
            "unfinished": engine.engine.has_unfinished_requests(),
            "compound_events": scheduler.compound_events,
            "queues": {name: [group.request_id for group in getattr(scheduler, name)]
                       for name in ("running", "waiting", "swapped")},
        }
    finally:
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()
    result["summary"] = json.loads((output / "executor_summary.json").read_text())
    return result


def run_forced(root, output):
    infos = public_infos(root)
    stage_counts = {index: 1 for index in range(4)}
    engine, scheduler = make_engine(root, output, stage_counts, "1:stage:0", 50)
    try:
        for request_id, collection in [("single-0", 0), ("compound-1a", 1),
                                       ("compound-1b", 1), ("single-2", 2)]:
            add(engine, infos, request_id, collection, 0, 256)
        engine.engine.step()
        add(engine, infos, "single-3", 3, 0, 256)
        scheduler.policy.current["single-3"]["ready_since"] -= 10.0
        scheduler.policy.schedule_interval = 1
        engine.engine.step()
        forced_batch = read_batches(output)[-1]
        forced_value = scheduler.policy.forced
        scheduler.policy.schedule_interval = 50
        steps = 2
        while engine.engine.has_unfinished_requests() and steps < 100:
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        result = {
            "forced_batch": forced_batch,
            "forced": forced_value,
            "batches": read_batches(output),
            "unfinished": engine.engine.has_unfinished_requests(),
            "summary": None,
        }
    finally:
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()
    result["summary"] = json.loads((output / "executor_summary.json").read_text())
    return result


def run_swap(root, output):
    infos = public_infos(root)
    stage_counts = {index: 1 for index in range(4)}
    engine, scheduler = make_engine(root, output, stage_counts, "1:stage:0", 50, num_blocks=40)
    preemptions = []
    original_preempt = scheduler._preempt

    @functools.wraps(original_preempt)
    def audited_preempt(group, blocks_to_swap_out, preemption_mode):
        result = original_preempt(group, blocks_to_swap_out, preemption_mode)
        preemptions.append({"request_id": group.request_id, "mode": str(result)})
        return result

    scheduler._preempt = audited_preempt
    try:
        for request_id, collection, prompt in [
            ("single-0", 0, 256), ("compound-1a", 1, 256), ("compound-1b", 1, 256),
            ("single-2", 2, 256), ("single-3", 3, 256),
        ]:
            add(engine, infos, request_id, collection, 0, prompt)
        engine.engine.step()
        duplicate = scheduler.running[0]
        scheduler.swapped.append(duplicate)
        scheduler._normalize_queues()
        dedup_witness = (scheduler.running.count(duplicate) == 1
                         and scheduler.swapped.count(duplicate) == 0)
        scheduler.policy.candidate_selector = selector("0:stage:0")
        scheduler.policy.remaining = 0.0
        scheduler.policy.schedule_interval = 1
        steps = 1
        while engine.engine.has_unfinished_requests() and steps < 100:
            if steps == 3:
                scheduler.policy.schedule_interval = 50
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        result = {
            "batches": read_batches(output),
            "preemptions": preemptions,
            "dedup_witness": dedup_witness,
            "compound_events": scheduler.compound_events,
            "unfinished": engine.engine.has_unfinished_requests(),
            "queues": {name: [group.request_id for group in getattr(scheduler, name)]
                       for name in ("running", "waiting", "swapped")},
        }
    finally:
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()
    result["summary"] = json.loads((output / "executor_summary.json").read_text())
    return result


def run_compound_preemption(root, output):
    infos = public_infos(root)
    engine, scheduler = make_engine(root, output, {0: 1, 1: 1}, "1:stage:0", 50,
                                    num_blocks=64)
    preemptions = []
    original_preempt = scheduler._preempt

    @functools.wraps(original_preempt)
    def audited_preempt(group, blocks_to_swap_out, preemption_mode):
        result = original_preempt(group, blocks_to_swap_out, preemption_mode)
        preemptions.append({"request_id": group.request_id, "mode": str(result)})
        return result

    scheduler._preempt = audited_preempt
    try:
        for request_id, collection in [("target-0", 0), ("compound-1a", 1), ("compound-1b", 1)]:
            add(engine, infos, request_id, collection, 0, 256)
        engine.engine.step()
        initial = read_batches(output)[-1]
        scheduler.policy.batch_slots = 2
        scheduler.policy.candidate_selector = selector("0:stage:0")
        scheduler.policy.remaining = 0.0
        engine.engine.step()
        replacement = read_batches(output)[-1]
        queues = {name: [group.request_id for group in getattr(scheduler, name)]
                  for name in ("running", "waiting", "swapped")}
        scheduler.policy.batch_slots = 4
        scheduler.policy.candidate_selector = selector("1:stage:0")
        scheduler.policy.remaining = 0.0
        steps = 2
        while engine.engine.has_unfinished_requests() and steps < 300:
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        result = {
            "initial": initial, "replacement": replacement, "queues": queues,
            "preemptions": preemptions, "batches": read_batches(output),
            "steps": steps, "unfinished": engine.engine.has_unfinished_requests(),
        }
    finally:
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()
    result["summary"] = json.loads((output / "executor_summary.json").read_text())
    return result


def run_state(root, output):
    infos = public_infos(root)
    engine, scheduler = make_engine(root, output, {0: 2}, "0:stage:0", 1)
    try:
        add(engine, infos, "state-0", 0, 0, 256)
        scheduler.policy.schedule_interval = 50
        engine.engine.step()
        scheduler.policy.schedule_interval = 1
        steps = 1
        while engine.engine.has_unfinished_requests() and steps < 100:
            if steps == 3:
                scheduler.policy.schedule_interval = 50
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        virtual = scheduler.policy.chain_meta[0]["virtual"]
        away = scheduler.policy.state.away_goals(time.perf_counter() + 1.0, set())
        add(engine, infos, "state-1", 0, 1, 256)
        inherited = scheduler.policy.chain_meta[0]["virtual"]
        while engine.engine.has_unfinished_requests() and steps < 200:
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        add(engine, infos, "state-reset", 0, 0, 256)
        reset_virtual = scheduler.policy.chain_meta[0]["virtual"]
        while engine.engine.has_unfinished_requests() and steps < 300:
            engine.engine.step()
            steps += 1
        for _ in range(2):
            engine.engine.step()
        result = {
            "virtual_after_stage0": virtual,
            "virtual_after_stage1_admit": inherited,
            "virtual_after_new_chain_admit": reset_virtual,
            "away": [goal.to_dict() if hasattr(goal, "to_dict") else {
                "state": goal.state, "tool_entered": goal.tool_entered,
                "tool_elapsed": goal.tool_elapsed,
            } for goal in away],
            "policy": scheduler.policy.snapshot(),
            "steps": steps,
            "unfinished": engine.engine.has_unfinished_requests(),
        }
    finally:
        engine.shutdown_background_loop()
        engine.engine.model_executor.finalize()
    result["summary"] = json.loads((output / "executor_summary.json").read_text())
    return result


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        frame = run_basic(root, root / "stage6_mechanism_repair/controlled" / attempt / "frame", 1)
        details["frame"] = frame
        batches = frame["batches"]
        events = frame["policy"]["events"]
        compound_partial = [batch for batch in batches
                            if ("compound-1a" in batch) != ("compound-1b" in batch)]
        check("frame_anchor_and_bounded_batch_executed",
              batches and "compound-1a" in batches[0] and "compound-1b" in batches[0]
              and 0 < max(map(len, batches)) <= 4,
              {"first_batch": batches[0] if batches else None,
               "max_batch": max(map(len, batches)) if batches else None},
              "accepted compound anchor executes without exceeding the four-request envelope")
        check("frame_compound_atomic", not compound_partial, compound_partial,
              "no partial compound batch")
        after = [event for event in events if event["kind"] == "v2_after_step"]
        check("compound_lease_decrement_once_per_group_step",
              after and abs(after[0]["remaining"] - 0.03) < 1e-9,
              {"first_remaining": after[0]["remaining"] if after else None},
              "two-member compound consumes one of four decode-equivalent lease units per step")
        check("accepted_does_not_poison_support",
              all(entry["unsupported_from"] is None for entry in frame["policy"]["support_cache"].values()),
              frame["policy"]["support_cache"], "accepted action preserves support")
        check("frame_complete_and_release", not frame["unfinished"]
              and frame["summary"]["tracked_block_owners"] == 0,
              {"unfinished": frame["unfinished"], "owners": frame["summary"]["tracked_block_owners"]},
              "all calls complete and KV owners are released")

        nonframe = run_basic(root, root / "stage6_mechanism_repair/controlled" / attempt / "nonframe", 50)
        details["nonframe"] = nonframe
        batches = nonframe["batches"]
        partial = [batch for batch in batches
                   if ("compound-1a" in batch) != ("compound-1b" in batch)]
        check("nonframe_compound_atomic_and_anchor",
              batches and not partial and "compound-1a" in batches[0] and "compound-1b" in batches[0],
              {"first_batch": batches[0] if batches else None, "partial": partial},
              "the native non-frame path also preserves compound and executes anchor")

        forced = run_forced(root, root / "stage6_mechanism_repair/controlled" / attempt / "forced")
        details["forced"] = forced
        check("forced_waiting_request_executes",
              "single-3" in forced["forced_batch"] and forced["forced"] == "3:stage:0"
              and not forced["unfinished"],
              {"forced_batch": forced["forced_batch"], "forced": forced["forced"],
               "unfinished": forced["unfinished"]},
              "a forced waiting request enters the actual frame and all calls complete")

        swap = run_swap(root, root / "stage6_mechanism_repair/controlled" / attempt / "swap")
        details["swap"] = swap
        partial = [batch for batch in swap["batches"]
                   if ("compound-1a" in batch) != ("compound-1b" in batch)]
        check("swap_compound_atomic_and_release",
              not partial and not swap["unfinished"]
              and swap["summary"]["tracked_block_owners"] == 0,
              {"preemptions": len(swap["preemptions"]), "partial": partial,
               "unfinished": swap["unfinished"], "queues": swap["queues"]},
              "KV-constrained compound and waiting work complete atomically without leaked state")
        duplicates = [batch for batch in swap["batches"] if len(batch) != len(set(batch))]
        check("queue_dedup_and_unique_execution",
              swap["dedup_witness"] and not duplicates,
              {"dedup_witness": swap["dedup_witness"], "duplicate_batches": duplicates},
              "cross-queue duplicate is removed and no request executes twice in a batch")

        preempted = run_compound_preemption(
            root, root / "stage6_mechanism_repair/controlled" / attempt / "compound_preemption")
        details["compound_preemption"] = preempted
        partial = [batch for batch in preempted["batches"]
                   if ("compound-1a" in batch) != ("compound-1b" in batch)]
        evicted = {item["request_id"] for item in preempted["preemptions"]}
        check("running_compound_preempted_and_restored_atomically",
              set(preempted["initial"]) == {"target-0", "compound-1a", "compound-1b"}
              and preempted["replacement"] == ["target-0"]
              and evicted == {"compound-1a", "compound-1b"}
              and set(preempted["queues"]["swapped"]) == {"compound-1a", "compound-1b"}
              and not partial and not preempted["unfinished"]
              and preempted["summary"]["tracked_block_owners"] == 0,
              {"initial": preempted["initial"], "replacement": preempted["replacement"],
               "preemptions": preempted["preemptions"], "partial": partial,
               "unfinished": preempted["unfinished"]},
              "both running compound members swap out together and later resume without duplication or leaks")

        state = run_state(root, root / "stage6_mechanism_repair/controlled" / attempt / "state")
        details["state"] = state
        check("chain_virtual_survives_stage_and_resets_only_new_chain",
              state["virtual_after_stage0"] > 0
              and state["virtual_after_stage1_admit"] == state["virtual_after_stage0"]
              and state["virtual_after_new_chain_admit"] == 0,
              {key: state[key] for key in ("virtual_after_stage0", "virtual_after_stage1_admit",
                                           "virtual_after_new_chain_admit")},
              "virtual service persists across stages and resets only for a new chain instance")
        away = state["away"]
        check("current_away_goal_present", len(away) == 1 and away[0]["state"] == "AWAY"
              and away[0]["tool_entered"] is not None and away[0]["tool_elapsed"] > 0,
              away, "one current AWAY goal with an independent elapsed timer")
        check("state_complete_and_release", not state["unfinished"]
              and state["summary"]["tracked_block_owners"] == 0,
              {"unfinished": state["unfinished"], "owners": state["summary"]["tracked_block_owners"]},
              "all state transitions complete without KV leaks")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "all controlled checks completed")

    passed = all(item["pass"] for item in checks.values())
    result = {"schema": "step2-execution-gate/v1", "attempt": attempt,
              "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
              "gpu_used": False, "next_step_allowed": passed}
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_EXECUTION_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
