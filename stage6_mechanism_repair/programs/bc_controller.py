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
    ticks = int(stat.rsplit(")", 1)[1].split()[19])
    created = subprocess.check_output(["ps", "-o", "lstart=", "-p", str(pid)], text=True).strip()
    return {"pid": pid, "pgid": os.getpgid(pid), "create_time": created, "proc_start_ticks": ticks}


def same_process(identity):
    try:
        return process_identity(identity["pid"])["proc_start_ticks"] == identity["proc_start_ticks"]
    except (FileNotFoundError, ProcessLookupError, subprocess.CalledProcessError):
        return False


def gpu_state():
    gpu = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"
    ], text=True).strip()
    processes = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"
    ], text=True).strip()
    return {"gpu": gpu, "processes": processes}


def main():
    root = Path(sys.argv[1]).resolve()
    arm = sys.argv[2]
    attempt_name = sys.argv[3]
    required_gate = sys.argv[4] if len(sys.argv) > 4 else None
    if arm not in {"B", "C"}:
        raise ValueError("Only arm B or C is supported")
    if required_gate:
        gate = json.loads((root / required_gate).read_text())
        if gate.get("gate") != "PASS" and gate.get("final_gate") != "PASS":
            raise RuntimeError(f"Required gate is not PASS: {required_gate}")
    smoke = json.loads((root / "stage6_mechanism_repair/SMOKE_FREEZE.json").read_text())
    authorization = json.loads((root / "stage6_mechanism_repair/GPU0_AUTHORIZATION.json").read_text())
    base = json.loads((root / "FREEZE.json").read_text())
    prepare = json.loads((root / "PREPARE.json").read_text())
    resources = dict(smoke["resources"])
    resources["scheduling_policy"] = "jitserve"
    resources["physical_gpu_index"] = 0
    resources["gpu_uuid"] = authorization["gpu_uuid"]
    attempt = root / "stage6_mechanism_repair/attempts" / attempt_name
    attempt.mkdir(parents=True, exist_ok=False)
    status_path = attempt / "controller_status.json"
    registry_path = attempt / "processes.json"
    registry = []
    processes = {}
    handles = []
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
            "arm": arm,
            **fields,
        })

    def remaining():
        value = authorization["max_attempt_wall_seconds"] - (time.monotonic() - started)
        if value <= 0:
            raise TimeoutError("Stage6 attempt wall-clock budget exhausted")
        return value

    def start_process(name, command, cuda_visible):
        env = os.environ.copy()
        library = json.loads(Path(base["runtime"]["library_contract"]).read_text())
        for key in base["runtime"]["reuse_only"]:
            env[key] = library[key]
        for key, value in base["runtime"]["new_environment_variables"].items():
            env[key] = value
        paths = [str(root / "stage6_mechanism_repair/programs"),
                 str(root / "stage3_gpu0/programs"), str(root / "stage3_gpu0/backend_src"),
                 str(root / "stage3_gpu0/v2_src")]
        paths.extend(entry if os.path.isabs(entry) else str(root / entry)
                     for entry in base["runtime"]["pythonpath_entries"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        env["STUDY_GPU"] = "0"
        stdout = (attempt / f"{name}.stdout.log").open("x")
        stderr = (attempt / f"{name}.stderr.log").open("x")
        handles.extend([stdout, stderr])
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
        if not same_process(identity) or identity["pgid"] != identity["pid"]:
            raise RuntimeError(f"Refusing to signal changed identity for {name}")
        os.killpg(identity["pgid"], grace_signal)
        try:
            process.wait(timeout=25)
            return
        except subprocess.TimeoutExpired:
            pass
        if same_process(identity):
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
        gpu = gpu_state()
        if not gpu["gpu"].startswith(f'0, {resources["gpu_uuid"]},'):
            raise RuntimeError(f"GPU0 identity mismatch: {gpu['gpu']}")
        if gpu["processes"]:
            raise RuntimeError(f"GPU0 already has compute processes: {gpu['processes']}")
        for port in [18080, 65433]:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
        training_link = root / "runtime/benchmarks/dataset/trace/deepresearch_training.json"
        target = root / "repo/traces/deepresearch_training.json"
        if not training_link.is_symlink() or training_link.resolve() != target.resolve():
            raise RuntimeError("Official training graph link is missing or incorrect")

        if arm == "B":
            status("PREDICTOR_START")
            predictor = start_process(
                "predictor",
                [prepare["python"], str(root / "programs/predictor_server.py"), str(root)], "")
            deadline = time.monotonic() + min(900, remaining())
            while True:
                if predictor.poll() is not None:
                    raise RuntimeError(f"Predictor exited during startup with {predictor.returncode}")
                try:
                    with socket.create_connection(("127.0.0.1", 65433), timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Predictor listening timeout")
                    time.sleep(0.25)
            status("PREDICTOR_READY")
        else:
            status("NATIVE_V2_PREDICTOR_SELECTED")

        server_script = root / ("stage3_gpu0/programs/bc_b_server_audit.py" if arm == "B" else "stage6_mechanism_repair/programs/c_server_audit.py")
        server_args = [
            prepare["python"], str(server_script), str(root), str(attempt),
            "--host", "127.0.0.1",
            "--port", "18080",
            "--model", prepare["model"],
            "--prediction-model-path", prepare["qrf_model"],
            "--prediction-tokenizer-path", prepare["qrf_vectorizer"],
            "--scheduling-policy", resources["scheduling_policy"],
            "--graph-structure-type", "deepresearch",
            "--graph-matching-mode", "online",
            "--stage-ratio-method", "execution_time",
            "--enable-chunked-prefill", "True",
            "--disable-async-output-proc",
            "--enforce-eager",
            "--penalty-factor", str(resources["penalty_factor"]),
            "--max-num-seqs", str(resources["max_num_seqs"]),
            "--max-num-batched-tokens", str(resources["max_num_batched_tokens"]),
            "--num-gpu-blocks-override", str(resources["num_gpu_blocks_override"]),
            "--swap-space", str(resources["swap_space_gib"]),
            "--gpu-memory-utilization", str(resources["gpu_memory_utilization"]),
            "--top-k-selection", str(resources["top_k_selection"]),
            "--search-strategy", resources["search_strategy"],
            "--max-swaps-per-iter", str(resources["max_swaps_per_iter"]),
        ]
        if arm == "C":
            server_args.extend(["--disable-prediction", "--disable-graph-matching", "--graph-matching-mode", "none"])
        cuda_visible = "0"
        status("SERVER_START")
        server = start_process("server", server_args, cuda_visible)
        deadline = time.monotonic() + min(600, remaining())
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited during startup with {server.returncode}")
            try:
                connection = http.client.HTTPConnection("127.0.0.1", 18080, timeout=2)
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
        status("SERVER_READY", gpu=gpu_state())
        client = start_process(
            "client",
            [prepare["python"], str(root / "stage6_mechanism_repair/programs/smoke_client.py"), str(root), str(attempt)], "")
        status("CLIENT_RUNNING")
        client_returncode = client.wait(timeout=remaining())
        if client_returncode != 0:
            raise RuntimeError(f"Client exited with {client_returncode}")
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
        for handle in handles:
            handle.close()
        final_gpu = None
        own_pids = {str(identity["pid"]) for identity in registry}
        own_remaining = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            final_gpu = gpu_state()
            own_remaining = [
                line for line in final_gpu["processes"].splitlines()
                if line.split(",", 1)[0].strip() in own_pids
            ]
            if not own_remaining:
                gpu_released = True
                break
            time.sleep(1)
        receipt = {
            "attempt": attempt_name,
            "arm": arm,
            "client_returncode": client_returncode,
            "failure": failure,
            "processes": [
                {**identity, "returncode": processes[identity["name"]].poll()}
                for identity in registry
            ],
            "gpu_released": gpu_released,
            "own_gpu_processes_remaining": own_remaining,
            "final_gpu0": final_gpu,
            "elapsed_seconds": time.monotonic() - started,
            "utc": utcnow(),
            "scope": "Stage6 fixed-output six-chain B/C mechanism qualification only; no A, ablation or formal matrix",
        }
        dump(root / "stage6_mechanism_repair/receipts" / f"{attempt_name}.json", receipt)
        dump(attempt / "controller_receipt.json", receipt)
        status("CLEANUP_COMPLETE" if failure is None else "FAILED_PRESERVED", gpu_released=gpu_released)
    if failure is not None:
        raise RuntimeError(failure["error"])


if __name__ == "__main__":
    main()
