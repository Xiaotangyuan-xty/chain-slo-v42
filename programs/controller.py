import datetime
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def process_identity(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    start_ticks = stat.rsplit(")", 1)[1].split()[19]
    create_time = subprocess.check_output(["ps", "-o", "lstart=", "-p", str(pid)], text=True).strip()
    return {"pid": pid, "pgid": os.getpgid(pid), "create_time": create_time, "proc_start_ticks": int(start_ticks)}


def same_process(identity):
    try:
        return process_identity(identity["pid"])["proc_start_ticks"] == identity["proc_start_ticks"]
    except (FileNotFoundError, ProcessLookupError, subprocess.CalledProcessError):
        return False


def gpu_state(index):
    gpu = subprocess.check_output([
        "nvidia-smi", "-i", str(index), "--query-gpu=index,uuid,memory.used",
        "--format=csv,noheader",
    ], text=True).strip()
    processes = subprocess.check_output([
        "nvidia-smi", "-i", str(index), "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader",
    ], text=True).strip()
    return {"gpu": gpu, "processes": processes}


def main():
    root = Path(sys.argv[1]).resolve()
    attempt_name = sys.argv[2]
    freeze = json.loads((root / "FREEZE.json").read_text())
    prepare = json.loads((root / "PREPARE.json").read_text())
    attempt = root / "attempts" / attempt_name
    attempt.mkdir(exist_ok=False)
    status_path = attempt / "controller_status.json"
    registry_path = attempt / "processes.json"
    registry = []
    processes = {}
    log_files = []
    started = time.monotonic()
    failure = None
    client_returncode = None
    gpu_released = False

    def status(phase, **fields):
        dump(status_path, {
            "phase": phase,
            "utc": utcnow(),
            "elapsed_seconds": time.monotonic() - started,
            "controller_pid": os.getpid(),
            "attempt": attempt_name,
            **fields,
        })

    def remaining():
        value = freeze["resources"]["max_attempt_wall_seconds"] - (time.monotonic() - started)
        if value <= 0:
            raise TimeoutError("Frozen attempt wall-clock budget exhausted")
        return value

    def start_process(name, command, cuda_visible):
        env = os.environ.copy()
        library = json.loads(Path(freeze["runtime"]["library_contract"]).read_text())
        for key in freeze["runtime"]["reuse_only"]:
            env[key] = library[key]
        for key, value in freeze["runtime"]["new_environment_variables"].items():
            env[key] = value
        pythonpath = [
            entry if os.path.isabs(entry) else str(root / entry)
            for entry in freeze["runtime"]["pythonpath_entries"]
        ]
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        stdout = (attempt / f"{name}.stdout.log").open("x")
        stderr = (attempt / f"{name}.stderr.log").open("x")
        log_files.extend([stdout, stderr])
        process = subprocess.Popen(
            command,
            cwd=root / "runtime",
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        identity = process_identity(process.pid)
        identity.update({"name": name, "command": command, "cuda_visible_devices": cuda_visible, "utc": utcnow()})
        registry.append(identity)
        dump(registry_path, registry)
        processes[name] = process
        return process

    def stop_process(name, grace_signal):
        process = processes.get(name)
        if process is None or process.poll() is not None:
            return
        identity = next(item for item in registry if item["name"] == name)
        if not same_process(identity):
            raise RuntimeError(f"Refusing to signal changed process identity for {name}")
        if identity["pgid"] != identity["pid"]:
            raise RuntimeError(f"Unexpected process group for {name}")
        os.killpg(identity["pgid"], grace_signal)
        try:
            process.wait(timeout=25)
            return
        except subprocess.TimeoutExpired:
            pass
        if not same_process(identity):
            return
        os.killpg(identity["pgid"], signal.SIGTERM)
        try:
            process.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            pass
        if same_process(identity):
            os.killpg(identity["pgid"], signal.SIGKILL)
        process.wait(timeout=10)

    try:
        status("PREFLIGHT")
        gpu = gpu_state(freeze["resources"]["physical_gpu_index"])
        expected = f'{freeze["resources"]["physical_gpu_index"]}, {freeze["resources"]["gpu_uuid"]},'
        if not gpu["gpu"].startswith(expected):
            raise RuntimeError(f"GPU2 identity mismatch: {gpu['gpu']}")
        if gpu["processes"]:
            raise RuntimeError(f"GPU2 already has compute processes: {gpu['processes']}")
        for port in [freeze["server"]["port"], freeze["server"]["prediction_port"]]:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((freeze["server"]["host"], port))
        training_link = root / "runtime/benchmarks/dataset/trace/deepresearch_training.json"
        training_link.parent.mkdir(parents=True, exist_ok=True)
        training_target = root / freeze["input"]["training_graphs"]
        if training_link.is_symlink():
            if training_link.resolve() != training_target.resolve():
                raise RuntimeError("Unexpected existing training-graph link")
        elif not training_link.exists():
            training_link.symlink_to(training_target)
        else:
            raise RuntimeError("Unexpected existing training-graph path")

        status("PREDICTOR_START")
        predictor = start_process(
            "predictor",
            [prepare["python"], str(root / "programs/predictor_server.py"), str(root)],
            freeze["resources"]["predictor_and_client_cuda_visible_devices"],
        )
        deadline = time.monotonic() + min(300, remaining())
        while True:
            if predictor.poll() is not None:
                raise RuntimeError(f"Predictor exited during startup with {predictor.returncode}")
            try:
                with socket.create_connection((freeze["server"]["prediction_host"], freeze["server"]["prediction_port"]), timeout=1):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Predictor listening timeout")
                time.sleep(0.25)
        status("PREDICTOR_READY")

        server_args = [
            prepare["python"],
            str(root / "programs/server_audit.py"),
            str(root),
            str(attempt),
            "--host", freeze["server"]["host"],
            "--port", str(freeze["server"]["port"]),
            "--model", prepare["model"],
            "--prediction-model-path", prepare["qrf_model"],
            "--prediction-tokenizer-path", prepare["qrf_vectorizer"],
            "--scheduling-policy", freeze["server"]["scheduling_policy"],
            "--graph-structure-type", freeze["server"]["graph_structure_type"],
            "--graph-matching-mode", freeze["server"]["graph_matching_mode"],
            "--stage-ratio-method", freeze["server"]["stage_ratio_method"],
            "--enable-chunked-prefill", str(freeze["server"]["enable_chunked_prefill"]),
            "--penalty-factor", str(freeze["server"]["penalty_factor"]),
            "--max-num-seqs", str(freeze["server"]["max_num_seqs"]),
            "--gpu-memory-utilization", str(freeze["server"]["gpu_memory_utilization"]),
        ]
        status("SERVER_START")
        server = start_process("server", server_args, freeze["resources"]["server_cuda_visible_devices"])
        deadline = time.monotonic() + min(600, remaining())
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited during startup with {server.returncode}")
            try:
                connection = http.client.HTTPConnection(freeze["server"]["host"], freeze["server"]["port"], timeout=2)
                connection.request("GET", "/health")
                response = connection.getresponse()
                response.read()
                connection.close()
                if response.status == 200:
                    break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Server health timeout")
            time.sleep(0.5)
        status("SERVER_READY", gpu=gpu_state(freeze["resources"]["physical_gpu_index"]))

        client = start_process(
            "client",
            [prepare["python"], str(root / "programs/client_audit.py"), str(root), str(attempt)],
            freeze["resources"]["predictor_and_client_cuda_visible_devices"],
        )
        status("CLIENT_RUNNING")
        client_returncode = client.wait(timeout=remaining())
        if client_returncode != 0:
            raise RuntimeError(f"Official client exited with {client_returncode}")
        time.sleep(3)
        status("CLIENT_COMPLETE", returncode=client_returncode)
    except BaseException as error:
        failure = {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        dump(attempt / "CONTROLLER_FAILURE.json", failure)
    finally:
        status("CLEANUP")
        for name, grace in [("client", signal.SIGTERM), ("server", signal.SIGINT), ("predictor", signal.SIGTERM)]:
            try:
                stop_process(name, grace)
            except BaseException as error:
                failure = failure or {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
                dump(attempt / f"CLEANUP_FAILURE_{name}.json", failure)
        for handle in log_files:
            handle.close()
        release_deadline = time.monotonic() + 30
        final_gpu = None
        while time.monotonic() < release_deadline:
            final_gpu = gpu_state(freeze["resources"]["physical_gpu_index"])
            if not final_gpu["processes"]:
                gpu_released = True
                break
            time.sleep(1)
        receipt = {
            "study_id": freeze["study_id"],
            "attempt": attempt_name,
            "logical_id": "A_public_deepresearch_chain0_native_smoke",
            "client_returncode": client_returncode,
            "failure": failure,
            "processes": [
                {**identity, "returncode": processes[identity["name"]].poll()}
                for identity in registry
            ],
            "gpu_released": gpu_released,
            "final_gpu2": final_gpu,
            "elapsed_seconds": time.monotonic() - started,
            "utc": utcnow(),
            "scope": "A native qualification smoke; no B/C performance run",
        }
        dump(root / "RECEIPT" / f"{attempt_name}.json", receipt)
        dump(attempt / "controller_receipt.json", receipt)
        status("CLEANUP_COMPLETE" if failure is None else "FAILED_PRESERVED", gpu_released=gpu_released)
    if failure is not None:
        raise RuntimeError(failure["error"])


if __name__ == "__main__":
    main()
