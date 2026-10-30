import hashlib
import json
import os
from pathlib import Path
import time

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (CompletionSequenceGroupOutput, Logprob,
                           SequenceGroupMetadataDelta, SequenceOutput)
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.worker import raise_if_cache_size_invalid


class HarnessAsyncExecutor(ExecutorAsyncBase):
    uses_ray = False
    output_dir = None
    backend_factory = None

    @classmethod
    def configure(cls, output_dir, backend_factory=None):
        cls.output_dir = Path(output_dir).resolve()
        cls.backend_factory = backend_factory

    def _init_executor(self):
        if self.output_dir is None:
            raise RuntimeError("Harness executor output directory is not configured")
        self.output_dir = Path(self.output_dir).resolve()
        self.backend_factory = type(self).backend_factory
        if self.backend_factory is None:
            from scheduler_backend import SchedulerManagedBackend
            self.backend = SchedulerManagedBackend(8192, 8192)
        else:
            self.backend = self.backend_factory()
        self.block_owners = {}
        self.computed_by_request = {}
        self.events = []
        self.finalized = False
        self.event_path = self.output_dir / "executor_events.jsonl"
        self.event_handle = self.event_path.open("x")
        self.cache_block_bytes = CacheEngine.get_cache_block_size(
            self.cache_config, self.model_config, self.parallel_config)
        self.record("executor_init", cache_block_bytes=self.cache_block_bytes)

    def record(self, kind, **fields):
        event = {"n": len(self.events), "kind": kind, "wall": time.time(), **fields}
        self.events.append(event)
        self.event_handle.write(json.dumps(event, allow_nan=False, default=str) + "\n")
        self.event_handle.flush()

    def determine_num_available_blocks(self):
        gpu_blocks = 512
        cpu_blocks = int(self.cache_config.swap_space_bytes // self.cache_block_bytes)
        self.record("determine_blocks", gpu_blocks=gpu_blocks, cpu_blocks=cpu_blocks)
        return gpu_blocks, cpu_blocks

    def initialize_cache(self, num_gpu_blocks, num_cpu_blocks):
        raise_if_cache_size_invalid(
            num_gpu_blocks, self.cache_config.block_size,
            self.cache_config.is_attention_free, self.model_config.max_model_len)
        if num_cpu_blocks < 1:
            raise RuntimeError("SLOScheduler requires at least one CPU KV block")
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        self.backend.reset(num_gpu_blocks * self.cache_config.block_size,
                           num_cpu_blocks * self.cache_config.block_size)
        self.record("initialize_cache", gpu_blocks=num_gpu_blocks, cpu_blocks=num_cpu_blocks,
                    gpu_tokens=num_gpu_blocks * self.cache_config.block_size,
                    cpu_tokens=num_cpu_blocks * self.cache_config.block_size)

    def _prompt_hash(self, tokens):
        return hashlib.sha256(json.dumps(list(tokens)).encode()).hexdigest()

    def _swap_out(self, request_id):
        backend = self.backend
        backend.resolve(request_id)
        cache = backend.gpu[request_id]
        copied, start, end = backend._copy(cache, True)
        begin = time.perf_counter()
        end.synchronize()
        backend._record_copy_wait(time.perf_counter() - begin)
        backend.stats["copy_cuda_span_seconds"] += start.elapsed_time(end) / 1000
        backend.stats["swap_bytes_out"] += backend.nbytes(cache)
        backend.host[request_id] = copied
        backend.host_tokens[request_id] = backend.gpu_tokens[request_id]
        token_count = len(backend.host_tokens[request_id])
        backend.gpu.pop(request_id)
        backend.gpu_tokens.pop(request_id)
        backend.ledger.commit("swap_out", request_id, token_count)
        backend.events.append({"event": "official_swap_out", "program": request_id, "tokens": token_count})
        self.record("backend_swap_out", request_id=request_id, tokens=token_count)

    def _apply_swaps(self, request):
        if request.blocks_to_copy:
            raise RuntimeError("blocks_to_copy is not supported by the single-sequence harness executor")
        owners_to_swap = set()
        pairs = []
        for source, destination in request.blocks_to_swap_out:
            owners = [request_id for request_id, blocks in self.block_owners.items() if source in blocks]
            if len(owners) != 1:
                raise RuntimeError(f"Cannot identify swap-out owner for block {source}: {owners}")
            owners_to_swap.add(owners[0])
            pairs.append((owners[0], source, destination))
        for request_id in sorted(owners_to_swap):
            self._swap_out(request_id)
        for request_id, source, destination in pairs:
            self.block_owners[request_id].discard(source)
            self.record("official_swap_out_block", request_id=request_id, source=source, destination=destination)
        if request.blocks_to_swap_in:
            self.record("official_swap_in_blocks", pairs=request.blocks_to_swap_in)

    def _claim_blocks(self, request_id, blocks):
        claimed = set(blocks)
        for owner in list(self.block_owners):
            self.block_owners[owner].difference_update(claimed)
        self.block_owners[request_id] = claimed

    def _validate_sampling(self, metadata):
        params = metadata.sampling_params
        if params is None:
            raise RuntimeError("SamplingParams are required")
        checks = {
            "n": params.n == 1,
            "best_of": params.best_of is None or params.best_of == params.n,
            "temperature": params.temperature == 0.0,
            "top_p": params.top_p == 1.0,
            "top_k": params.top_k == -1,
            "min_p": params.min_p == 0.0,
            "min_tokens": params.min_tokens == 0,
            "presence_penalty": params.presence_penalty == 0.0,
            "frequency_penalty": params.frequency_penalty == 0.0,
            "repetition_penalty": params.repetition_penalty == 1.0,
            "use_beam_search": not getattr(params, "use_beam_search", False),
            "logprobs": params.logprobs is None,
            "prompt_logprobs": params.prompt_logprobs is None,
            "guided_decoding": getattr(params, "guided_decoding", None) is None,
            "logits_processors": not getattr(params, "logits_processors", None),
            "guided_logits_processors": not getattr(params, "guided_logits_processors", None),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise NotImplementedError(f"Unsupported sampling semantics: {failed}")

    def _release_request(self, request_id, reason):
        self.backend.release(request_id)
        self.block_owners.pop(request_id, None)
        self.computed_by_request.pop(request_id, None)
        self.record("backend_release", request_id=request_id, reason=reason)

    def execute_model(self, execute_model_req):
        started = time.perf_counter()
        self._apply_swaps(execute_model_req)
        for request_id in execute_model_req.finished_requests_ids:
            self._release_request(request_id, "finished")
        items = []
        descriptions = []
        targets = {}
        for metadata in execute_model_req.seq_group_metadata_list:
            if isinstance(metadata, SequenceGroupMetadataDelta):
                raise NotImplementedError("SequenceGroupMetadataDelta is not supported")
            if len(metadata.seq_data) != 1:
                raise RuntimeError("The harness executor accepts exactly one sequence per group")
            self._validate_sampling(metadata)
            if getattr(metadata, "computed_block_nums", None):
                raise NotImplementedError("Prefix-cached computed blocks are not supported")
            seq_id = next(iter(metadata.seq_data))
            data = metadata.seq_data[seq_id]
            ids = list(data.get_token_ids())
            prompt_len = data.get_prompt_len()
            computed = data.get_num_computed_tokens()
            chunk = metadata.token_chunk_size or 0
            previous_computed = self.computed_by_request.get(metadata.request_id)
            if metadata.is_prompt and (computed == 0 or
                                       (previous_computed is not None and computed < previous_computed)):
                self._release_request(metadata.request_id, "recompute_invalidate")
            target = min(len(ids), computed + chunk) if metadata.is_prompt else len(ids)
            prompt_end = min(prompt_len, target)
            full_prompt = ids[:prompt_len]
            if metadata.is_prompt:
                items.append((metadata.request_id, ids[:target], []))
            else:
                emitted = ids[prompt_len:target]
                if not emitted:
                    raise RuntimeError("Decode metadata has no emitted token")
                items.append((metadata.request_id, full_prompt, emitted))
            targets[metadata.request_id] = target
            blocks = [block for table in metadata.block_tables.values() for block in table]
            self._claim_blocks(metadata.request_id, blocks)
            descriptions.append({
                "request_id": metadata.request_id,
                "seq_id": seq_id,
                "prompt_tokens": prompt_len,
                "prompt_sha256": self._prompt_hash(full_prompt),
                "visible_prompt_tokens": prompt_end,
                "emitted_tokens": target - prompt_end,
                "computed_tokens_before": computed,
                "computed_tokens_after": target,
                "token_chunk_size": chunk,
                "is_prompt": metadata.is_prompt,
                "do_sample": metadata.do_sample,
                "blocks": blocks,
            })
        forward_before = self.backend.stats.get("forward_wall_seconds", 0.0)
        logits = {}
        for begin in range(0, len(items), 4):
            logits.update(self.backend.step_batch(items[begin:begin + 4]))
        outputs = []
        sampled = []
        for metadata in execute_model_req.seq_group_metadata_list:
            samples = []
            if metadata.do_sample:
                value = logits[metadata.request_id]
                token = int(value.argmax(dim=-1).item())
                seq_id = next(iter(metadata.seq_data))
                samples.append(SequenceOutput(
                    parent_seq_id=seq_id,
                    output_token=token,
                    logprobs={token: Logprob(logprob=float("inf"), rank=0, decoded_token=None)},
                ))
                sampled.append({"request_id": metadata.request_id, "token": token})
            outputs.append(CompletionSequenceGroupOutput(samples=samples, prompt_logprobs=None))
            self.computed_by_request[metadata.request_id] = targets[metadata.request_id]
        elapsed = time.perf_counter() - started
        forward_elapsed = self.backend.stats.get("forward_wall_seconds", 0.0) - forward_before
        self.record("execute_model", groups=descriptions, sampled=sampled,
                    elapsed_seconds=elapsed, forward_wall_seconds=forward_elapsed,
                    backend_stats={key: value for key, value in self.backend.stats.items()
                                   if key != "native_decode_batches"})
        return [SamplerOutput(outputs=outputs, model_forward_time=forward_elapsed,
                              model_execute_time=elapsed)]

    async def execute_model_async(self, execute_model_req):
        result = self.execute_model(execute_model_req)
        if execute_model_req.async_callback is not None:
            execute_model_req.async_callback()
        return result

    def add_lora(self, lora_request):
        return False

    def remove_lora(self, lora_id):
        return False

    def pin_lora(self, lora_id):
        return False

    def list_loras(self):
        return set()

    def add_prompt_adapter(self, prompt_adapter_request):
        return False

    def remove_prompt_adapter(self, prompt_adapter_id):
        return False

    def pin_prompt_adapter(self, prompt_adapter_id):
        return False

    def list_prompt_adapters(self):
        return set()

    def check_health(self):
        return None

    def _release_all(self, reason):
        request_ids = set(self.block_owners) | set(self.backend.ledger.entries)
        for request_id in sorted(request_ids):
            self._release_request(request_id, reason)

    def stop_remote_worker_execution_loop(self):
        self._release_all("idle_stop")

    async def stop_remote_worker_execution_loop_async(self):
        self._release_all("async_idle_stop")

    def finalize(self):
        if self.finalized:
            return
        self._release_all("finalize")
        summary = {
            "cache_block_bytes": self.cache_block_bytes,
            "gpu_blocks": self.cache_config.num_gpu_blocks,
            "cpu_blocks": self.cache_config.num_cpu_blocks,
            "backend": self.backend.summary(),
            "event_counts": {kind: sum(event["kind"] == kind for event in self.events)
                             for kind in sorted({event["kind"] for event in self.events})},
            "tracked_block_owners": len(self.block_owners),
        }
        temporary = self.output_dir / "executor_summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False, default=str) + "\n")
        os.replace(temporary, self.output_dir / "executor_summary.json")
        self.finalized = True
        self.event_handle.close()

    def shutdown(self):
        self.finalize()


class HarnessAsyncLLMEngine(AsyncLLMEngine):

    @classmethod
    def _get_executor_cls(cls, engine_config):
        return HarnessAsyncExecutor
