import functools
import json
import os
from pathlib import Path
import sys
import traceback

from controlled_scheduler import add, make_engine, public_infos, read_batches, selector


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only residency gate")
    output = root / "stage6_mechanism_repair/controlled" / attempt / "residency"
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        infos = public_infos(root)
        engine, scheduler = make_engine(root, output, {0: 1, 1: 1}, "1:stage:0", 50,
                                        num_blocks=64)
        preemptions = []
        original = scheduler._preempt

        @functools.wraps(original)
        def audited(group, blocks, mode):
            result = original(group, blocks, mode)
            preemptions.append({"request_id": group.request_id, "mode": str(result)})
            return result

        scheduler._preempt = audited
        try:
            for request_id, collection in (("target-0", 0), ("compound-1a", 1), ("compound-1b", 1)):
                add(engine, infos, request_id, collection, 0, 256)
                for group in scheduler.policy.seq_group_dict[collection]:
                    group.sampling_params.max_tokens = 8
                    group.request_info.output_len = 8
                    group.predict_output_length = 8
            scheduler.policy.batch_slots = 3
            engine.engine.step()
            initial = read_batches(output)[-1]
            scheduler.policy.batch_slots = 2
            scheduler.policy.candidate_selector = selector("0:stage:0")
            scheduler.policy.remaining = 0.0
            engine.engine.step()
            first_eviction = read_batches(output)[-1]
            first_count = len(preemptions)

            scheduler.policy.candidate_selector = selector("1:stage:0")
            scheduler.policy.remaining = 0.0
            engine.engine.step()
            resumed = read_batches(output)[-1]
            scheduler.policy.candidate_selector = selector("0:stage:0")
            scheduler.policy.remaining = 0.0
            protected_batches = []
            progress = []
            while len(preemptions) < 4 and len(protected_batches) < 12:
                progress.append(scheduler.policy.decode_progress.get("compound-1a", 0))
                engine.engine.step()
                protected_batches.append(read_batches(output)[-1])
            after_guard_count = len(preemptions)
            guard_batches = protected_batches[:-1] if after_guard_count >= 4 else protected_batches
            second_eviction = protected_batches[-1] if after_guard_count >= 4 else None
            second_count = after_guard_count

            scheduler.policy.batch_slots = 4
            scheduler.policy.candidate_selector = selector("1:stage:0")
            scheduler.policy.remaining = 0.0
            steps = 0
            while engine.engine.has_unfinished_requests() and steps < 100:
                engine.engine.step()
                steps += 1
            for _ in range(2):
                engine.engine.step()
            details.update(initial=initial, first_eviction=first_eviction,
                           resumed=resumed, protected_batches=protected_batches,
                           progress=progress, second_eviction=second_eviction,
                           first_count=first_count, after_guard_count=after_guard_count,
                           second_count=second_count, preemptions=preemptions,
                           policy_events=scheduler.policy.audit_events,
                           unfinished=engine.engine.has_unfinished_requests())
        finally:
            engine.shutdown_background_loop()
            engine.engine.model_executor.finalize()
        summary = json.loads((output / "executor_summary.json").read_text())
        details["summary"] = summary
        check("compound_has_no_immediate_repeated_swap",
              first_count == 2 and guard_batches
              and all(batch == ["compound-1a", "compound-1b"] for batch in guard_batches),
              {"first_count": first_count, "after_guard_count": after_guard_count,
               "progress": progress, "guard_batches": guard_batches},
              "after restoration the whole compound makes progress without another swap")
        compound_preemptions = sum(item["request_id"] in ("compound-1a", "compound-1b")
                                   for item in preemptions)
        check("second_swap_requires_four_decode_steps",
              compound_preemptions == 4 and progress and max(progress) <= 4
              and second_eviction == ["target-0"],
              {"progress": progress, "compound_preemptions": compound_preemptions,
               "all_preemptions": preemptions, "second_eviction": second_eviction},
              "the compound is evicted exactly twice, with the second allowed after four decode steps")
        check("residency_completes_without_leaks",
              not details["unfinished"] and summary["tracked_block_owners"] == 0,
              {"unfinished": details["unfinished"], "owners": summary["tracked_block_owners"]},
              "protected requests resume and finish without deadlock or KV leaks")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "residency gate completed")

    passed = all(item["pass"] for item in checks.values())
    result = {"schema": "stage10-residency-gate/v1", "attempt": attempt,
              "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
              "gpu_used": False}
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_RESIDENCY_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
