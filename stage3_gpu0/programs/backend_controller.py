import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def gpu_state():
    gpu = subprocess.check_output(["nvidia-smi", "-i", "2", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"], text=True).strip()
    processes = subprocess.check_output(["nvidia-smi", "-i", "2", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True).strip()
    return {"gpu": gpu, "processes": processes}


def identity(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    ticks = int(stat.rsplit(")", 1)[1].split()[19])
    created = subprocess.check_output(["ps", "-o", "lstart=", "-p", str(pid)], text=True).strip()
    return {"pid": pid, "pgid": os.getpgid(pid), "create_time": created, "proc_start_ticks": ticks}


def same_process(record):
    try:
        return identity(record["pid"])["proc_start_ticks"] == record["proc_start_ticks"]
    except (FileNotFoundError, ProcessLookupError, subprocess.CalledProcessError):
        return False


def main():
    root = Path(sys.argv[1]).resolve()
    arm = sys.argv[2]
    attempt_name = sys.argv[3]
    freeze = json.loads((root / "stage3/BACKEND_FREEZE.json").read_text())
    base = json.loads((root / "FREEZE.json").read_text())
    prepare = json.loads((root / "PREPARE.json").read_text())
    initial = gpu_state()
    if not initial["gpu"].startswith(f'2, {freeze["gpu_uuid"]},') or initial["processes"]:
        raise RuntimeError(f"GPU2 is unavailable: {initial}")
    attempt = root / "stage3/attempts" / attempt_name
    attempt.mkdir(parents=True, exist_ok=False)
    registry = []
    process = None
    handles = []
    started = time.monotonic()
    failure = None
    returncode = None
    try:
        library = json.loads(Path(base["runtime"]["library_contract"]).read_text())
        env = os.environ.copy()
        for key in base["runtime"]["reuse_only"]:
            env[key] = library[key]
        for key, value in base["runtime"]["new_environment_variables"].items():
            env[key] = value
        paths = [str(root / "stage3/programs"), str(root / "stage3/backend_src")]
        paths.extend(entry if os.path.isabs(entry) else str(root / entry)
                     for entry in base["runtime"]["pythonpath_entries"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["CUDA_VISIBLE_DEVICES"] = "2"
        env["STUDY_GPU"] = "2"
        stdout = (attempt / "worker.stdout.log").open("x")
        stderr = (attempt / "worker.stderr.log").open("x")
        handles.extend([stdout, stderr])
        command = [prepare["python"], str(root / "stage3/programs/backend_worker.py"), str(root), arm, str(attempt)]
        process = subprocess.Popen(command, cwd=root / "runtime", env=env,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        record = identity(process.pid)
        record.update({"command": command, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        registry.append(record)
        dump(attempt / "processes.json", registry)
        returncode = process.wait(timeout=freeze["execution"]["max_attempt_wall_seconds"])
        if returncode != 0:
            raise RuntimeError(f"Backend worker exited with {returncode}")
    except BaseException as error:
        failure = {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        dump(attempt / "CONTROLLER_FAILURE.json", failure)
    finally:
        if process is not None and process.poll() is None:
            record = registry[0]
            if same_process(record) and record["pgid"] == record["pid"]:
                os.killpg(record["pgid"], signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    if same_process(record):
                        os.killpg(record["pgid"], signal.SIGTERM)
                    process.wait(timeout=10)
        for handle in handles:
            handle.close()
        final = gpu_state()
        own_remaining = [line for line in final["processes"].splitlines()
                         if line.split(",", 1)[0].strip() in {str(item["pid"]) for item in registry}]
        receipt = {
            "attempt": attempt_name,
            "arm": arm,
            "failure": failure,
            "returncode": returncode if process is None else process.poll(),
            "processes": [{**item, "returncode": process.poll() if process else None} for item in registry],
            "gpu_released": not own_remaining,
            "own_gpu_processes_remaining": own_remaining,
            "final_gpu2": final,
            "elapsed_seconds": time.monotonic() - started,
            "scope": "Small fixed backend calibration only; no C or formal matrix",
        }
        dump(attempt / "controller_receipt.json", receipt)
        dump(root / "stage3/receipts" / f"{attempt_name}.json", receipt)
    if failure is not None:
        raise RuntimeError(failure["error"])


if __name__ == "__main__":
    main()
