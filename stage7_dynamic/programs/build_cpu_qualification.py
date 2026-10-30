import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    qualification_name = Path(sys.argv[2]).name if len(sys.argv) > 2 else "CPU_QUALIFICATION.json"
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only stage7 qualification")
    stage = root / "stage7_dynamic"
    receipts = stage / "receipts"
    freeze = json.loads((stage / "FULL_FREEZE.json").read_text())
    runtime_receipt = receipts / (Path(sys.argv[3]).name if len(sys.argv) > 3
                                  else "V3_CPU_RUNTIME_45_RUNTIME_INTEGRATION.json")
    runtime = json.loads(runtime_receipt.read_text())
    tests = {}
    commands = {
        "dynamic_batch": ("stage7_dynamic/programs/controlled_dynamic_batch.py", 5),
        "scheduler": ("stage7_dynamic/programs/controlled_stage7_scheduler.py", 44),
        "rollout": ("stage7_dynamic/programs/controlled_stage7_rollout.py", 30),
    }
    for name, (relative, expected) in commands.items():
        completed = subprocess.run(
            [sys.executable, str(root / relative)],
            cwd=root,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log = receipts / f"CPU_{name.upper()}_{Path(qualification_name).stem}.log"
        with log.open("x") as handle:
            handle.write(completed.stdout)
        tests[name] = {
            "gate": "PASS" if completed.returncode == 0 else "FAIL",
            "expected_checks": expected,
            "command": relative,
            "log": str(log.relative_to(root)),
            "log_sha256": sha(log),
        }
    tests["real_vllm_runtime"] = {
        "gate": runtime["gate"],
        "receipt": str(runtime_receipt.relative_to(root)),
        "receipt_sha256": sha(runtime_receipt),
        "checks": {name: value["pass"] for name, value in runtime["checks"].items()},
        "max_batch": runtime["details"]["max_batch"],
        "accepted": runtime["details"]["accepted"],
        "scheduled": runtime["details"]["scheduled"],
        "executed": runtime["details"]["executed"],
        "controlled_selector": True,
        "natural_performance_claim": False,
    }
    files = {}
    for group in ("programs", "backend_src", "v2_src"):
        for path in sorted((stage / group).glob("*.py")):
            compile(path.read_text(), str(path), "exec")
            files[str(path.relative_to(root))] = sha(path)
    for path in sorted((stage / "assets").glob("*.json")):
        files[str(path.relative_to(root))] = sha(path)
    for name in ("FULL_FREEZE.json", "GPU0_AUTHORIZATION.json"):
        path = stage / name
        files[str(path.relative_to(root))] = sha(path)
    passed = all(item["gate"] == "PASS" for item in tests.values())
    result = {
        "schema": "stage7-dynamic-cpu-qualification/v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gate": "PASS" if passed else "FAIL",
        "mechanism_version": freeze["frozen_mechanism"]["version"],
        "gpu_used": False,
        "tests": tests,
        "files": files,
        "source_file_count": len(files),
        "dynamic_batch_limit_removed": True,
        "official_dynamic_sliding_window": True,
        "gpu_launch_requires_new_idle_preflight": passed,
        "win_criterion": "C official scaled attained > B; JCT, goodput and overheads are reported without a performance threshold",
        "not_proven": [
            "No stage7 GPU performance result exists yet",
            "The runtime action uses a declared controlled selector, not a natural performance gain",
            "Real model generation and scheduling must still be checked on the full public smoke pair",
        ],
        "formal_matrix_allowed": False,
        "holdout_allowed": False,
    }
    path = receipts / qualification_name
    with path.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"gate": result["gate"], "tests": {k: v["gate"] for k, v in tests.items()},
                      "source_file_count": len(files), "sha256": sha(path),
                      "receipt": str(path)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
