import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "stage7_dynamic/programs"),
    str(ROOT / "stage7_dynamic/backend_src"),
    str(ROOT / "stage3_gpu0/programs"),
]


class ExecutorAsyncBase:
    pass


class AsyncLLMEngine:
    pass


class SamplerOutput:
    def __init__(self, outputs, model_forward_time, model_execute_time):
        self.outputs = outputs
        self.model_forward_time = model_forward_time
        self.model_execute_time = model_execute_time


class CompletionSequenceGroupOutput:
    def __init__(self, samples, prompt_logprobs):
        self.samples = samples
        self.prompt_logprobs = prompt_logprobs


class Logprob:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class SequenceGroupMetadataDelta:
    pass


class SequenceOutput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class CacheEngine:
    pass


def install_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module


def install_harness_stubs():
    for name in (
        "vllm", "vllm.engine", "vllm.executor", "vllm.model_executor",
        "vllm.model_executor.layers", "vllm.worker", "transformers",
    ):
        install_module(name)
    install_module("vllm.engine.async_llm_engine", AsyncLLMEngine=AsyncLLMEngine)
    install_module("vllm.executor.executor_base", ExecutorAsyncBase=ExecutorAsyncBase)
    install_module("vllm.model_executor.layers.sampler", SamplerOutput=SamplerOutput)
    install_module(
        "vllm.sequence", CompletionSequenceGroupOutput=CompletionSequenceGroupOutput,
        Logprob=Logprob, SequenceGroupMetadataDelta=SequenceGroupMetadataDelta,
        SequenceOutput=SequenceOutput)
    install_module("vllm.worker.cache_engine", CacheEngine=CacheEngine)
    install_module("vllm.worker.worker", raise_if_cache_size_invalid=lambda *args: None)


class DynamicCache:
    def __init__(self, legacy=()):
        self.legacy = legacy

    @classmethod
    def from_legacy_cache(cls, legacy):
        return cls(legacy)

    def to_legacy_cache(self):
        return self.legacy


install_harness_stubs()
install_module("transformers.cache_utils", DynamicCache=DynamicCache)

import torch

from contracts import CacheLedger
from fake_backend import FakeBackend
from harness_executor import HarnessAsyncExecutor
from pool_backend import BatchedBackend


class RecordingFakeBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.batch_calls = []

    def step_batch(self, items):
        self.batch_calls.append(list(items))
        logits = {}
        for item in items:
            logits.update(super().step_batch([item]))
        return logits


class Data:
    def __init__(self, ids, prompt_len):
        self.ids = ids
        self.prompt_len = prompt_len

    def get_token_ids(self):
        return self.ids

    def get_prompt_len(self):
        return self.prompt_len

    def get_num_computed_tokens(self):
        return self.prompt_len


def sampling_params():
    return SimpleNamespace(
        n=1, best_of=None, temperature=0.0, top_p=1.0, top_k=-1,
        min_p=0.0, min_tokens=0, presence_penalty=0.0,
        frequency_penalty=0.0, repetition_penalty=1.0,
        logprobs=None, prompt_logprobs=None)


def decode_metadata(index):
    ids = [10 + index, 20 + index, 30 + index, 40 + index]
    return SimpleNamespace(
        request_id=f"executor-{index}", seq_data={0: Data(ids, 3)},
        token_chunk_size=1, is_prompt=False, do_sample=True,
        computed_block_nums=[], block_tables={0: [index]},
        sampling_params=sampling_params())


def executor_batch():
    metadata = [decode_metadata(index) for index in range(6)]
    request = SimpleNamespace(
        seq_group_metadata_list=metadata, finished_requests_ids=[],
        blocks_to_copy=[], blocks_to_swap_out=[], blocks_to_swap_in=[])
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        executor = HarnessAsyncExecutor.__new__(HarnessAsyncExecutor)
        executor.backend = RecordingFakeBackend()
        executor.backend.reset(8192, 8192)
        executor.block_owners = {}
        executor.computed_by_request = {}
        executor.events = []
        executor.finalized = False
        executor.cache_block_bytes = 0
        executor.output_dir = output
        executor.cache_config = SimpleNamespace(num_gpu_blocks=512, num_cpu_blocks=0)
        executor.event_handle = (output / "executor_events.jsonl").open("x")
        try:
            result = executor.execute_model(request)
        finally:
            executor.event_handle.close()
        calls = [[item[0] for item in call] for call in executor.backend.batch_calls]
        samples = [len(output.samples) for output in result[0].outputs]
        computed = dict(executor.computed_by_request)
    return calls, samples, computed


class FakeEvent:
    def record(self):
        return None

    def synchronize(self):
        return None

    def elapsed_time(self, other):
        return 0.0


class CPUTorch:
    def __init__(self):
        self.long = torch.long
        self.nn = torch.nn
        self.cuda = SimpleNamespace(Event=lambda enable_timing: FakeEvent())

    def tensor(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        return torch.tensor(*args, **kwargs)

    def cat(self, *args, **kwargs):
        return torch.cat(*args, **kwargs)

    def inference_mode(self):
        return torch.inference_mode()


class FakeBackbone:
    def __call__(self, input_ids, past_key_values, **kwargs):
        produced = []
        for pair in past_key_values.to_legacy_cache():
            row = []
            for value in pair:
                update = input_ids.to(value.dtype)[:, :, None, None].expand(
                    value.shape[0], value.shape[1], 1, value.shape[-1])
                row.append(torch.cat((value, update), dim=-2))
            produced.append(tuple(row))
        return SimpleNamespace(
            last_hidden_state=input_ids.to(torch.float32).unsqueeze(-1),
            past_key_values=DynamicCache(tuple(produced)))


class FakeModel:
    def __init__(self):
        self.model = FakeBackbone()

    def lm_head(self, hidden):
        return torch.cat((hidden, -hidden), dim=-1)


def make_backend(items, capacity=1024):
    backend = BatchedBackend.__new__(BatchedBackend)
    backend.torch = CPUTorch()
    backend.model = FakeModel()
    backend.gpu = {}
    backend.host = {}
    backend.pending = {}
    backend.tokens = {}
    backend.gpu_tokens = {}
    backend.host_tokens = {}
    backend.ledger = CacheLedger(capacity, 1024)
    backend.gpu_capacity = capacity
    backend.host_capacity = 1024
    backend.stats = {
        "productive_decode_queries": 0, "logical_outputs": 0,
        "actual_forward_query_tokens": 0, "forward_wall_seconds": 0.0,
    }
    backend.events = []
    for index, (key, prompt, emitted) in enumerate(items):
        full = tuple(prompt) + tuple(emitted[:-1])
        value = torch.full((1, 1, len(full), 1), float(index))
        backend.gpu[key] = ((value, value.clone()),)
        backend.gpu_tokens[key] = full
        backend.tokens[key] = full
        backend.ledger.commit("gpu_produced", key, len(full))
    return backend


def caught_error(call):
    try:
        call()
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only dynamic batch gate")
    checks = {}
    details = {}

    def check(name, passed, actual, expected):
        checks[name] = {"pass": bool(passed), "actual": actual, "expected": expected}

    try:
        calls, samples, computed = executor_batch()
        expected_ids = [f"executor-{index}" for index in range(6)]
        details["executor"] = {
            "backend_batch_calls": calls, "samples_per_group": samples,
            "computed_requests": sorted(computed),
        }
        check("executor_passes_complete_metadata_once",
              calls == [expected_ids] and samples == [1] * 6
              and sorted(computed) == expected_ids,
              details["executor"],
              "six scheduler groups produce one complete backend.step_batch call")

        items = [
            (f"backend-{index}", [10 + index] * (index % 3 + 1),
             [100 + index, 200 + index])
            for index in range(6)
        ]
        backend = make_backend(items)
        logits = backend.step_batch(items)
        expected_tokens = {
            key: tuple(prompt) + tuple(emitted) for key, prompt, emitted in items
        }
        kv_lengths = {key: backend.gpu[key][0][0].shape[-2] for key, _, _ in items}
        details["backend"] = {
            "logits_requests": sorted(logits),
            "batch_records": backend.stats.get("native_decode_batches", []),
            "kv_lengths": kv_lengths,
            "decode_queries": backend.stats["productive_decode_queries"],
        }
        check("backend_executes_ragged_batch_above_four",
              set(logits) == {key for key, _, _ in items}
              and backend.gpu_tokens == expected_tokens
              and all(kv_lengths[key] == len(expected_tokens[key]) for key in kv_lengths)
              and backend.stats["native_decode_batches"][0]["B"] == 6
              and backend.stats["productive_decode_queries"] == 6
              and len(backend.events) == 6
              and all(event["batch_size"] == 6 for event in backend.events),
              details["backend"], "one six-request ragged decode preserves tokens, KV and outputs")

        duplicate_error = caught_error(lambda: backend.step_batch([
            ("duplicate", [1], [2]), ("duplicate", [1], [2])]))
        details["duplicate_error"] = duplicate_error
        check("duplicate_request_ids_remain_rejected",
              duplicate_error == "ValueError: batch boundary", duplicate_error,
              "ValueError: batch boundary")

        overflow_items = [(f"overflow-{index}", [index], [100 + index])
                          for index in range(5)]
        overflow = make_backend(overflow_items, capacity=5)
        overflow_error = caught_error(lambda: overflow.step_batch(overflow_items))
        details["overflow_error"] = overflow_error
        check("batch_kv_quota_remains_enforced",
              overflow_error == "RuntimeError: batch cannot fit common KV quota",
              overflow_error, "RuntimeError: batch cannot fit common KV quota")

        prefix_items = [("prefix", [1], [2, 3])]
        prefix = make_backend(prefix_items)
        prefix_error = caught_error(lambda: prefix.step_batch([
            ("prefix", [9], [2, 3])]))
        details["prefix_error"] = prefix_error
        check("prefix_identity_remains_enforced",
              prefix_error == "RuntimeError: GPU prefix witness mismatch",
              prefix_error, "RuntimeError: GPU prefix witness mismatch")
    except Exception as error:
        details["exception"] = traceback.format_exc(limit=10)
        check("controlled_completion", False, f"{type(error).__name__}: {error}",
              "all dynamic batch checks completed")

    passed = bool(checks) and all(item["pass"] for item in checks.values())
    result = {
        "schema": "stage7-dynamic-batch-gate/v1",
        "gate": "PASS" if passed else "FAIL",
        "checks": checks,
        "details": details,
        "gpu_used": False,
        "real_model_generation_equivalence_proven": False,
    }
    print(json.dumps(result, indent=2, allow_nan=False, default=str))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
