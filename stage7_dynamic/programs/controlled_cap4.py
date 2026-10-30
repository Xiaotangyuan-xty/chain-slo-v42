import json
import os
from pathlib import Path
import sys
import traceback

from controlled_scheduler import add, make_engine, public_infos, read_batches
from cap4_slo_scheduler import Cap4SLOScheduler


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only B-cap4 gate")
    output = root / "stage6_mechanism_repair/controlled" / attempt / "cap4"
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        infos = public_infos(root)
        engine, unused = make_engine(root, output, {index: 1 for index in range(5)},
                                     "0:stage:0", 50)
        old = engine.engine.scheduler[0]
        scheduler = Cap4SLOScheduler(old.scheduler_config, old.cache_config,
                                     old.lora_config, batch_slots=4)
        engine.engine.scheduler[0] = scheduler
        frames = []
        original = scheduler._schedule_slo

        def audited_frame():
            frames.append(scheduler.policy.num_schedule_count)
            return original()

        scheduler._schedule_slo = audited_frame
        try:
            for index in range(5):
                add(engine, infos, f"cap4-{index}", index, 0, 256, slo=120.0)
                group = scheduler.policy.seq_group_dict[index][0]
                group.sampling_params.max_tokens = 64
                group.request_info.output_len = 64
                group.predict_output_length = 64
            steps = 0
            while engine.engine.has_unfinished_requests() and steps < 400:
                engine.engine.step()
                steps += 1
            for _ in range(2):
                engine.engine.step()
            batches = read_batches(output)
            details.update(steps=steps, batches=batches, frames=frames,
                           unfinished=engine.engine.has_unfinished_requests(),
                           schedule_count=scheduler.policy.num_schedule_count,
                           scheduler_config_max_num_seqs=scheduler.scheduler_config.max_num_seqs)
        finally:
            engine.shutdown_background_loop()
            engine.engine.model_executor.finalize()
        summary = json.loads((output / "executor_summary.json").read_text())
        details["summary"] = summary
        check("cap4_real_batches",
              batches and max(map(len, batches)) <= 4
              and details["scheduler_config_max_num_seqs"] == 4,
              {"max_batch": max(map(len, batches), default=0),
               "config_max_num_seqs": details["scheduler_config_max_num_seqs"]},
              "B-cap4 never expands the real request batch or scheduler budget beyond four")
        check("official_slo_frames_still_execute",
              len(frames) >= 1 and details["schedule_count"] >= 50,
              {"frames": frames, "schedule_count": details["schedule_count"]},
              "the official _schedule_slo path continues to run at its frame cadence")
        check("cap4_completes_and_releases",
              not details["unfinished"] and steps < 400
              and summary["tracked_block_owners"] == 0,
              {"unfinished": details["unfinished"], "steps": steps,
               "owners": summary["tracked_block_owners"]},
              "all longer requests complete without starvation or leaked KV owners")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "B-cap4 gate completed")

    passed = all(item["pass"] for item in checks.values())
    result = {"schema": "stage8-cap4-gate/v1", "attempt": attempt,
              "gate": "PASS" if passed else "FAIL", "checks": checks, "details": details,
              "gpu_used": False, "gpu_qualification_allowed": passed}
    receipt = root / "stage6_mechanism_repair/receipts" / f"{attempt}_CAP4_GATE.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps({"gate": result["gate"],
                      "failed_checks": [name for name, item in checks.items() if not item["pass"]],
                      "receipt": str(receipt)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
