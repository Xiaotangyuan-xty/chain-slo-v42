import datetime
import http.client
import hashlib
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


def gpu_state(physical):
    physical = str(physical)
    gpu = subprocess.check_output([
        "nvidia-smi", "-i", physical, "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"
    ], text=True).strip()
    processes = subprocess.check_output([
        "nvidia-smi", "-i", physical, "--query-compute-apps=pid,used_memory", "--format=csv,noheader"
    ], text=True).strip()
    utilization=int(subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip())
    free=int(subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    return {"gpu": gpu, "processes": processes, 'utilization':utilization,'memory_free_MiB':free}


def main():
    root = Path(sys.argv[1]).resolve()
    stage = Path(sys.argv[2]).resolve()
    arm, attempt_name, mode = sys.argv[3:6]
    if arm not in {"A", "C"} or mode != "full":
        raise ValueError("Invalid arm/workload")
    protocol = json.loads((stage / "PROTOCOL.json").read_text())
    assert protocol['runtime_arm']==arm and protocol['allowed_arms']==[arm]
    smoke = json.loads((root / "stage7_dynamic/FULL_FREEZE.json").read_text())
    device = protocol["arm_devices"][arm]
    authorization = json.loads((stage / device["authorization_file"]).read_text())
    physical = authorization["physical_gpu_index"]
    assert physical==0 and protocol['workload_kind']=='mixed' and protocol['role']=='formal_nativeToT_mixed600'
    approval=json.loads((stage/'LONG_EXECUTION_APPROVAL.json').read_text())
    if not (approval.get('authorized') is True and approval.get('long_experiment_authorized') is True):raise PermissionError('New long-experiment approval required')
    if hashlib.sha256((stage/'LONG_EXECUTION_APPROVAL.json').read_bytes()).hexdigest()!=protocol['long_approval_sha256']:raise PermissionError('Long approval hash mismatch')
    if not any(d['physical_gpu_index']==physical and d['gpu_uuid']==authorization['gpu_uuid'] for d in approval['devices']):raise PermissionError('Assigned device not approved')
    if authorization['allowed_physical_gpu_indices'] != [physical]:raise PermissionError('Device authorization is not exclusive')
    if (device["physical_gpu_index"], device["gpu_uuid"]) != (physical, authorization["gpu_uuid"]):
        raise RuntimeError("Protocol and arm GPU authorization disagree")
    qualification = json.loads((stage / "receipts/CPU_QUALIFICATION.json").read_text())
    if qualification.get("gate") != "PASS":
        raise RuntimeError("Stage18 CPU qualification is not PASS")
    for prefix, hashes in [(root, qualification["base_files"]),
                           (stage, qualification["stage_files"])]:
        changed = [name for name, expected in hashes.items() if not (prefix / name).is_file()
                   or hashlib.sha256((prefix / name).read_bytes()).hexdigest() != expected]
        if changed:
            raise RuntimeError(f"Qualified source mismatch: {changed}")
    if mode == "compat" and arm != "C":
        raise ValueError("Compatibility is for the new C only")
    if mode != 'full' or protocol.get('study_mode') != 'stage74_v47max_A_Astar_mixed600_gpu0':raise RuntimeError('Only the frozen bound long protocol is allowed')
    base = json.loads((root / "FREEZE.json").read_text())
    from model_contract import prepare as bound_prepare
    prepare = bound_prepare(root,stage)
    from model_contract import resources as bound_resources, verify_external
    asset_receipt=verify_external(stage)
    resources = bound_resources(smoke["resources"],stage)
    resources["scheduling_policy"] = "jitserve"
    resources["physical_gpu_index"] = physical
    resources["gpu_uuid"] = authorization["gpu_uuid"]
    attempt = stage / "attempts" / attempt_name
    attempt.mkdir(parents=True, exist_ok=False)
    status_path = attempt / "controller_status.json"
    registry_path = attempt / "processes.json"
    registry = []
    processes = {}
    http_port=protocol["http_ports"][arm]
    prediction_port=protocol["prediction_ports"][arm]
    affinity=set(protocol["cpu_affinity"][arm])
    if not affinity or not affinity <= os.sched_getaffinity(0):raise RuntimeError("Invalid CPU affinity")
    os.sched_setaffinity(0,affinity)
    handles = []
    listeners = {}
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
            raise TimeoutError("Stage18 attempt wall-clock budget exhausted")
        return value

    def start_process(name, command, cuda_visible):
        env = os.environ.copy()
        library = json.loads(Path(base["runtime"]["library_contract"]).read_text())
        for key in base["runtime"]["reuse_only"]:
            env[key] = library[key]
        for key, value in base["runtime"]["new_environment_variables"].items():
            env[key] = value
        paths = [str(root / "stage7_dynamic/programs"),
                 str(root / "stage7_dynamic/backend_src"), str(root / "stage7_dynamic/v2_src"),
                 str(root / "stage3_gpu0/programs")]
        paths.extend(entry if os.path.isabs(entry) else str(root / entry)
                     for entry in base["runtime"]["pythonpath_entries"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        env["STUDY_GPU"] = str(physical)
        env["STUDY_ARM"] = arm
        env["STUDY_OUTPUT_DIR"] = str(attempt)
        inherited = listeners[name].fileno() if name in listeners else None
        if inherited is not None:env["STUDY_PREDICTOR_FD" if name=="predictor" else "STUDY_HTTP_FD"] = str(inherited)
        env["STUDY_C_VERSION"] = protocol["C_version"]
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
            pass_fds=(() if inherited is None else (inherited,)),
        )
        identity = process_identity(process.pid)
        identity.update({"name": name, "command": command, "cuda_visible_devices": cuda_visible, "utc": utcnow()})
        registry.append(identity)
        dump(registry_path, registry)
        processes[name] = process
        if name in listeners:listeners.pop(name).close()
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
            process.wait(timeout=300 if name == "server" else 25)
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
        gpu = gpu_state(physical)
        if not gpu["gpu"].startswith(f'{physical}, {resources["gpu_uuid"]},'):
            raise RuntimeError(f"GPU{physical} identity mismatch: {gpu['gpu']}")
        if gpu['memory_free_MiB']<protocol['minimum_free_memory_MiB_before_load']:
            raise RuntimeError('Insufficient GPU0 free memory for the fixed shared capacity; preserve without external signals: '+repr(gpu))
        status("PREFLIGHT_EXTERNAL_OBSERVED",gpu=gpu,external_coexistence_allowed=True)
        from port_reservation import reserve
        # Retain both listeners through exec. No check-close-bind interval.
        while True:
            try:
                listeners['predictor']=reserve(prediction_port)
                listeners['server']=reserve(http_port)
                break
            except OSError:
                for listener in listeners.values():listener.close()
                listeners.clear()
                status('WAIT_PORT_RESERVATION')
                remaining()
                time.sleep(5)
        dump(attempt/'LISTENER_RESERVATIONS.json',{'http':http_port,'prediction':prediction_port,'inherited_FD':True})
        training_link = root / "runtime/benchmarks/dataset/trace/deepresearch_training.json"
        target = root / "repo/traces/deepresearch_training.json"
        if not training_link.is_symlink() or training_link.resolve() != target.resolve():
            raise RuntimeError("Official training graph link is missing or incorrect")

        status("PREDICTOR_START")
        predictor = start_process(
            "predictor",
            [prepare["python"], str(stage / "programs/predictor_server.py"), str(root),str(stage),arm], "")
        deadline = time.monotonic() + min(900, remaining())
        while True:
            if predictor.poll() is not None:
                raise RuntimeError(f"Predictor exited during startup with {predictor.returncode}")
            try:
                if not (attempt/'PREDICTOR_READY.json').exists():raise OSError('Predictor model not ready')
                with socket.create_connection(("127.0.0.1",prediction_port), timeout=1):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Predictor listening timeout")
                time.sleep(0.25)
        status("PREDICTOR_READY")

        server_script = stage / "programs" / f"{arm.lower()}_server.py"
        server_args = [
            prepare["python"], str(server_script), str(root), str(stage), str(attempt),
            "--host", "127.0.0.1",
            "--port", str(http_port),
            "--model", prepare["model"],
            "--prediction-model-path", prepare["qrf_model"],
            "--prediction-tokenizer-path", prepare["qrf_vectorizer"],
            "--scheduling-policy", resources["scheduling_policy"],
            "--graph-structure-type", "tot",
            "--graph-matching-mode", "online",
            "--stage-ratio-method", "execution_time",
            "--enable-chunked-prefill", "True",
            "--disable-async-output-proc",
            "--enforce-eager",
            "--penalty-factor", str(resources["penalty_factor"]),
            "--dtype", "bfloat16",
            "--max-model-len", str(resources["max_model_len"]),
            "--seed", "0",
            "--max-num-seqs", str(resources["max_num_seqs"]),
            "--max-num-batched-tokens", str(resources["max_num_batched_tokens"]),
            "--swap-space", str(resources["swap_space_gib"]),
            "--gpu-memory-utilization", str(resources["gpu_memory_utilization"]),
            "--top-k-selection", str(resources["top_k_selection"]),
            "--search-strategy", resources["search_strategy"],
            "--max-swaps-per-iter", str(resources["max_swaps_per_iter"]),
        ]
        if resources.get("num_gpu_blocks_override") is not None:
            server_args.extend(["--num-gpu-blocks-override",str(resources["num_gpu_blocks_override"])])
        cuda_visible = authorization["gpu_uuid"]
        status("SERVER_START")
        server = start_process("server", server_args, cuda_visible)
        deadline = time.monotonic() + min(600, remaining())
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited during startup with {server.returncode}")
            try:
                connection = http.client.HTTPConnection("127.0.0.1",http_port, timeout=2)
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
        status("SERVER_READY", gpu=gpu_state(physical))
        if protocol.get('role') == 'kv_capacity_profile':
            assert physical == 0 and protocol['candidate_version'] == 'A'
            status('CAPACITY_READY_NO_REQUESTS')
            dump(attempt / 'CAPACITY_READY.json', {'utc': utcnow(), 'requests_sent': 0, 'model': prepare['model']})
        else:
            client = start_process(
                "client",
                [prepare["python"], str(stage / "programs/client.py"), str(root), str(stage), str(attempt), mode], "")
            status("CLIENT_RUNNING")
            client_returncode = client.wait(timeout=remaining())
            if client_returncode != 0:
                raise RuntimeError(f"Client exited with {client_returncode}")
            time.sleep(3)
            server_returncode = server.poll()
            if server_returncode not in (None, 0):
                raise RuntimeError(f"Server exited with {server_returncode}")
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
        if failure is None and "server" in processes:
            server_returncode = processes["server"].poll()
            if server_returncode not in (None, 0, -2):
                failure = {"type": "ServerExit", "error": f"Server exited with {server_returncode}",
                           "traceback": ""}
                dump(attempt / "CONTROLLER_FAILURE.json", failure)
        for listener in listeners.values():listener.close()
        listeners.clear()
        for handle in handles:
            handle.close()
        final_gpu = None
        own_pids = {str(identity["pid"]) for identity in registry}
        own_remaining = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            final_gpu = gpu_state(physical)
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
            "mode": mode,
            "stage": str(stage),
            "cpu_qualification_sha256": hashlib.sha256((stage / "receipts/CPU_QUALIFICATION.json").read_bytes()).hexdigest(),
            "client_returncode": client_returncode,
            "failure": failure,
            "processes": [
                {**identity, "returncode": processes[identity["name"]].poll()}
                for identity in registry
            ],
            "gpu_released": gpu_released,
            "own_gpu_processes_remaining": own_remaining,
            "final_gpu": final_gpu,
            "physical_gpu_index": physical,
            "gpu_uuid": authorization["gpu_uuid"],
            "elapsed_seconds": time.monotonic() - started,
            "utc": utcnow(),
            "scope": "Stage74 GPU0 five frozen mixed600 ratios: v47max/A/Astar fixed KV",
            "model_asset_verification":asset_receipt,
            "http_port":http_port,"prediction_port":prediction_port,"cpu_affinity":sorted(affinity),
        }
        dump(stage / "receipts" / f"{attempt_name}.json", receipt)
        dump(attempt / "controller_receipt.json", receipt)
        status("CLEANUP_COMPLETE" if failure is None else "FAILED_PRESERVED", gpu_released=gpu_released)
    if failure is not None:
        raise RuntimeError(failure["error"])


if __name__ == "__main__":
    main()
