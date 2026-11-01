import argparse
import asyncio
from dataclasses import asdict
from contextvars import ContextVar
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


class CallJSONLWriter:
    _STOP = object()

    def __init__(self, path):
        self.path = Path(path)
        self._queue = queue.Queue()
        self._ready = threading.Event()
        self._error = None
        self._exception = None
        self._finished = False
        self._thread = threading.Thread(target=self._run, name="client-calls-jsonl", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            self._thread.join()
            self._raise_error()

    @property
    def error(self):
        return self._error

    def _capture_error(self, error):
        if self._error is None:
            self._error = {
                "type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            self._exception = error

    def _raise_error(self):
        if self._error is not None:
            raise RuntimeError(f"client_calls JSONL writer failed: {self._error['error']}") from self._exception

    def _run(self):
        try:
            handle = self.path.open("x")
        except BaseException as error:
            self._capture_error(error)
            self._ready.set()
            return
        self._ready.set()
        try:
            while True:
                value = self._queue.get()
                try:
                    if value is self._STOP:
                        break
                    if self._error is None:
                        handle.write(json.dumps(value, allow_nan=False, default=str) + "\n")
                        handle.flush()
                except BaseException as error:
                    self._capture_error(error)
                finally:
                    self._queue.task_done()
        finally:
            try:
                handle.close()
            except BaseException as error:
                self._capture_error(error)

    def record(self, value):
        if self._finished:
            raise RuntimeError("Cannot record after JSONL writer finish")
        self._queue.put(value)

    def drain(self):
        self._queue.join()
        self._raise_error()

    def finish(self):
        if not self._finished:
            self._finished = True
            self._queue.put(self._STOP)
            self._queue.join()
            self._thread.join()
        self._raise_error()


REQUEST_RECORDS = {}
PHASE = ContextVar("audit_phase", default="warmup")

def request_summary(output, phase):
    return {
        **REQUEST_RECORDS[(phase, output.collection_id, output.stage_id, output.request_id)],
        "collection_id": output.collection_id,
        "stage_id": output.stage_id,
        "request_id": output.request_id,
        "state": output.state,
        "success": output.success,
        "error": output.error,
        "stop_reason": output.stop_reason,
        "finish_before_ddl": output.finish_before_ddl,
        "request_ttft": output.request_ttft,
        "request_tbt": output.request_tbt,
        "request_service_gain": output.request_service_gain,
        "request_latency": output.request_latency,
        "request_finish_time": output.request_finish_time,
    }


def collective_summary(output, phase):
    return {
        "phase": phase,
        "collection_id": output.collection_id,
        "success": output.success,
        "total_latency": output.total_latency,
        "stages": [
            {
                "stage_id": stage.stage_id,
                "success": stage.success,
                "stage_latency": stage.stage_latency,
                "requests": [request_summary(request, phase) for request in stage.stage_outputs],
            }
            for stage in output.stage_outputs
        ],
    }


def apply_response_data(output, data):
    output.request_tbt = data.get("tbt", [])
    output.request_service_gain = data.get("service_gain", 0)
    output.stop_reason = data.get("finish_reason", "")
    generated_text = data.get("text", [""])[0]
    if output.request_ttft == 0.0 and data.get("ttft") is not None:
        output.request_ttft = data["ttft"]
    return generated_text


def parse_response_buffer(buffer, is_stream, request_prompt, output):
    generated_text = ""
    if is_stream:
        for part in buffer.split("\0")[:-1]:
            generated_text = apply_response_data(output, json.loads(part))
    else:
        generated_text = apply_response_data(output, json.loads(buffer))
    output.request_output = generated_text[len(request_prompt):]
    return output


def make_optimized_single_request(client_module):
    async def optimized_single_request(
        request_info,
        model_name,
        client_deadline=200,
        is_stream=True,
        served_time=0.0,
    ):
        api_url = request_info.api_url
        prompt_format = client_module.get_prompt_format_func(model_name)
        request_prompt = prompt_format(request_info.request.prompt)

        output = client_module.RequestOutput()
        output.request_type = request_info.request.request_type.value
        output.request_input = request_prompt
        output.collection_id = request_info.request.collection_id
        output.stage_id = request_info.request.stage_id
        output.request_id = request_info.request.request_id
        output.state = request_info.request.state

        timeout = client_module.aiohttp.ClientTimeout(total=client_deadline)
        async with client_module.aiohttp.ClientSession(timeout=timeout) as session:
            payload = {
                "request_info": request_info.request.to_dict(),
                "slo_constraint": request_info.slo_constraint,
                "sampling_params": request_info.sampling_params.to_dict(),
                "client_id": request_info.client_id,
                "stream": is_stream,
                "target_output_length": request_info.request.output_len,
                "num_stages": request_info.num_stages,
                "requests_per_stage": request_info.requests_per_stage,
                "accumulate_stage_ratio": request_info.accumulate_stage_ratio,
                "served_time": served_time,
            }
            payload["request_info"]["prompt"] = request_prompt
            st = time.perf_counter()
            try:
                async with session.post(url=api_url, json=payload) as response:
                    if response.status == 200:
                        chunks = []
                        async for chunk_bytes in response.content.iter_any():
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue
                            chunks.append(chunk_bytes.decode("utf-8"))
                        buffer = "".join(chunks)
                        parse_response_buffer(buffer, is_stream, request_prompt, output)
                        output.request_latency = time.perf_counter() - st
                        output.success = True
                    else:
                        output.success = False
                        output.error = f"Failed response, status: {response.status}"
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
        output.request_finish_time = time.perf_counter()
        return output

    return optimized_single_request


def transport_call(function, request_info, args, kwargs, timeout):
    bound = inspect.signature(function).bind(request_info, *args, **kwargs)
    bound.arguments['client_deadline'] = timeout
    return bound


async def absolute_arrivals(requests, targets, origin):
    if len(requests) != len(targets):
        raise RuntimeError('Full trace and arrival plan lengths differ')
    for request, offset in zip(requests, targets):
        await asyncio.sleep(max(0.0, origin + offset - time.perf_counter()))
        yield request


def main():
    root = Path(sys.argv[1]).resolve()
    stage = Path(sys.argv[2]).resolve()
    output_dir = Path(sys.argv[3]).resolve()
    mode = sys.argv[4]
    protocol = json.loads((stage / "PROTOCOL.json").read_text())
    workload = protocol["workloads"][mode]
    smoke = json.loads((root / "stage7_dynamic/FULL_FREEZE.json").read_text())
    prepare = json.loads((root / "PREPARE.json").read_text())
    repo = Path(prepare["repo"])
    sys.path[:0] = [
        str(repo / "benchmark"),
        str(repo),
        str(repo / "third_party/vllm"),
        prepare["sklearn_deps"],
        prepare["support_deps"],
    ]
    from trace import client_deepresearch as client_module
    import benchmark_scheduler_deepresearch as benchmark_module

    optimized_single_request = make_optimized_single_request(client_module)

    async def audited_single(request_info, *args, **kwargs):
        phase = PHASE.get()
        request = request_info.request
        start = time.perf_counter()
        bound = transport_call(optimized_single_request, request_info, args, kwargs, protocol['http_request_timeout_seconds'])
        result = await optimized_single_request(*bound.args, **bound.kwargs)
        end = time.perf_counter()
        params = request_info.sampling_params
        REQUEST_RECORDS[(phase, request.collection_id, request.stage_id, request.request_id)] = {
            "client_request_start_monotonic": start,
            "client_request_end_monotonic": end,
            "reported_request_finish_monotonic": result.request_finish_time,
            "input_prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest(),
            "target_output_tokens": request.output_len,
            "received_slo": list(request_info.slo_constraint),
            "generation": {key: getattr(params, key, None) for key in
                           ["temperature", "top_p", "top_k", "seed", "ignore_eos", "max_tokens", "n"]},
        }
        return result
    client_module.send_single_request = audited_single

    calls = []
    log_errors = []
    arrival = {"origin": None}
    original_send = client_module.send_deepresearch_collective_request

    async def audited_send(*args, **kwargs):
        phase = "warmup" if not calls else "main"
        started = time.perf_counter()
        token = PHASE.set(phase)
        try:
            try:
                result = await original_send(*args, **kwargs)
            except BaseException as error:
                summary = {"phase": phase, "exception": repr(error), "traceback": traceback.format_exc()}
                calls.append(summary)
                try:
                    writer.record(summary)
                except BaseException as log_error:
                    log_errors.append({"type": type(log_error).__name__, "error": str(log_error),
                                       "traceback": traceback.format_exc()})
                raise
            summary = collective_summary(result, phase)
            summary["client_chain_start_monotonic"] = started
            summary["client_chain_end_monotonic"] = time.perf_counter()
            summary["arrival_offset_seconds"] = (
                None if arrival["origin"] is None else started - arrival["origin"])
            calls.append(summary)
            writer.record(summary)
            if phase == "warmup" and not summary.get("success"):
                raise RuntimeError("Warmup failed; full workload will not start")
            return result
        finally:
            PHASE.reset(token)

    client_module.send_deepresearch_collective_request = audited_send
    original_simulator = benchmark_module.deepresearch_client_simulator

    async def audited_simulator(*args, **kwargs):
        arrival["origin"] = time.perf_counter()
        return await original_simulator(*args, **kwargs)

    benchmark_module.deepresearch_client_simulator = audited_simulator
    async def planned_generator(collective_requests, poisson_lambda=1000.0, burst=False):
        if burst or poisson_lambda != 1000:
            raise RuntimeError('Full trace must retain the frozen 1 chain/s arrival rule')
        async for request in absolute_arrivals(collective_requests, protocol['target_arrivals'], arrival['origin']):
            yield request
    client_module.get_request_generator = planned_generator
    original_metrics = benchmark_module.calculate_deepresearch_metrics

    def audited_metrics(collective_outputs, *args, **kwargs):
        metrics = original_metrics(collective_outputs, *args, **kwargs)
        dump(output_dir / "official_metrics.json", asdict(metrics))
        return metrics

    benchmark_module.calculate_deepresearch_metrics = audited_metrics
    resources = smoke["resources"]
    generation = smoke["generation_and_deadline"]
    args = argparse.Namespace(
        base_url=None,
        host="127.0.0.1",
        port=protocol["http_port"],
        endpoint="/generate",
        trace_path=str(root / smoke["input"]["trace"]),
        num_prompts=workload["main_chain_count"],
        arrival_rate=generation["arrival_rate"],
        burst=generation["burst"],
        slo_constraint=",".join(str(value) for value in generation["slo_constraint_seconds"]),
        penalty_factor=resources["penalty_factor"],
        max_concurrency=None,
        policy="jitserve",
        model=prepare["model"],
        tokenizer=None,
        n=1,
        best_of=1,
        logprobs=None,
        max_output_len=generation["max_output_len"],
        seed=generation["seed"],
        trust_remote_code=False,
        disable_tqdm=True,
        ignore_eos=generation["ignore_eos"],
        metric_percentiles="50,90,95,99",
        show_id_length=False,
        is_stream=generation["is_stream"],
    )
    dump(output_dir / "CLIENT_CONTRACT.json", {"mode": mode, "workload": workload,
         "generation": generation, "trace_sha256": hashlib.sha256((root / smoke["input"]["trace"]).read_bytes()).hexdigest(),
         "http_request_timeout_seconds": protocol['http_request_timeout_seconds'],
         "measurement": "Official send/stage/chain semantics; shared absolute arrival plan and transport-only timeout extension; actual SLO unchanged"})
    writer = CallJSONLWriter(output_dir / "client_calls.jsonl")

    def finish_logging():
        errors = list(log_errors)
        try:
            writer.finish()
        except BaseException as error:
            errors.append({"type": type(error).__name__, "error": str(error),
                           "traceback": traceback.format_exc(), "writer_error": writer.error})
        try:
            dump(output_dir / "client_calls.json", calls)
        except BaseException as error:
            errors.append({"type": type(error).__name__, "error": str(error),
                           "traceback": traceback.format_exc()})
        return errors

    try:
        # Avoid cyclic-GC pauses during finite replay; process exit reclaims memory.
        gc.disable()
        benchmark_module.main(args)
    except BaseException as error:
        failure = {"error": repr(error), "traceback": traceback.format_exc()}
        errors = finish_logging()
        if errors:
            failure["logging_errors"] = errors
        try:
            dump(output_dir / "CLIENT_FAILURE.json", failure)
        except BaseException:
            pass
        raise
    errors = finish_logging()
    if errors:
        failure = {"error": "Client call logging failed", "logging_errors": errors}
        try:
            dump(output_dir / "CLIENT_FAILURE.json", failure)
        except BaseException:
            pass
        raise RuntimeError(f"Client call logging failed: {errors[0]['error']}")
    dump(output_dir / "CLIENT_DONE.json", {"complete": True, "calls": len(calls), "wall": time.time()})


if __name__ == "__main__":
    main()
