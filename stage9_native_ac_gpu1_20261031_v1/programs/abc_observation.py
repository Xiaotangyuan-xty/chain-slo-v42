"""Read-only engine/output observations shared by A, B, and frozen C."""
import atexit
from collections import Counter
import functools
import hashlib
import inspect
import json
from pathlib import Path
import time


class Observation:
    def __init__(self, output_dir, arm, root, device):
        self.path, self.arm, self.root = Path(output_dir), arm, Path(root)
        self.device = device
        self.handle = (self.path / 'common_events.jsonl').open('x')
        self.requests = {}
        self.stats = Counter()
        self.resources = None
        self.engine = None
        self.zero_starts = 0
        self.main = False
        self.closed = False
        self.record('device_identity', **device)
        atexit.register(self.finalize)

    def record(self, kind, **fields):
        self.handle.write(json.dumps({'kind': kind, 'monotonic': time.perf_counter(),
                                      'wall': time.time(), **fields}, default=str, allow_nan=False) + '\n')
        self.handle.flush()

    def engine_ready(self, engine):
        self.engine = engine
        llm = engine.engine
        executor = llm.model_executor
        scheduler = llm.scheduler[0]
        policy = scheduler.policy
        cache = llm.cache_config
        sc = llm.scheduler_config
        model = llm.model_config
        native = type(executor).__module__.startswith('vllm.executor.')
        if self.arm == 'A':
            assert type(executor).__module__.startswith('vllm.executor.'), type(executor)
            assert type(scheduler).__name__ == 'SLOScheduler', type(scheduler)
            assert type(policy).__name__ == 'SLOPolicy', type(policy)
            assert 'harness_executor' not in type(executor).__module__
        elif self.arm == 'B':
            assert type(scheduler).__name__ == 'SLOScheduler'
        else:
            assert native and getattr(policy, 'version', None) == 'v43_native_chain_deadline_reservation'
        self.resources = {
            'arm': self.arm, 'native_executor': native,
            'executor_type': f'{type(executor).__module__}.{type(executor).__qualname__}',
            'executor_source': inspect.getfile(type(executor)),
            'scheduler_type': f'{type(scheduler).__module__}.{type(scheduler).__qualname__}',
            'scheduler_source': inspect.getfile(type(scheduler)),
            'policy_type': f'{type(policy).__module__}.{type(policy).__qualname__}',
            'policy_source': inspect.getfile(type(policy)),
            'mechanism_version': getattr(policy, 'version', 'official'),
            'gpu_blocks': cache.num_gpu_blocks, 'cpu_blocks': cache.num_cpu_blocks,
            'block_size': cache.block_size, 'swap_space_bytes': cache.swap_space_bytes,
            'max_num_seqs': sc.max_num_seqs, 'max_num_batched_tokens': sc.max_num_batched_tokens,
            'max_model_len': model.max_model_len, 'dtype': str(model.dtype),
            'enforce_eager': model.enforce_eager,
            'single_engine': len(llm.scheduler) == 1,
            'schedule_interval': policy.schedule_interval,
            'search_strategy': scheduler.search_strategy,
            'max_swaps_per_iter': scheduler.max_swaps_per_iter,
            'enable_chunked_prefill': sc.chunked_prefill_enabled,
            'enable_prefix_caching': cache.enable_prefix_caching,
        }
        self.record('engine_ready', **self.resources)
        original = executor.execute_model_async
        @functools.wraps(original)
        async def execute(execute_model_req, *args, **kwargs):
            groups = execute_model_req.seq_group_metadata_list if execute_model_req is not None else []
            start = time.perf_counter()
            result = await original(execute_model_req, *args, **kwargs)
            self.stats['execute_calls'] += 1
            self.stats['executed_groups'] += len(groups)
            self.stats['max_executed_batch'] = max(self.stats['max_executed_batch'], len(groups))
            self.stats['over_four_executions'] += int(len(groups) > 4)
            self.record('model_execute', request_ids=[g.request_id for g in groups],
                        batch_size=len(groups), elapsed_seconds=time.perf_counter() - start,
                        swap_in_blocks=len(execute_model_req.blocks_to_swap_in or []) if execute_model_req else 0,
                        swap_out_blocks=len(execute_model_req.blocks_to_swap_out or []) if execute_model_req else 0)
            return result
        executor.execute_model_async = execute

    def finalize(self):
        if self.closed:
            return
        data = {'arm': self.arm, 'device': self.device, 'resources': self.resources,
                'stats': dict(self.stats), 'requests': list(self.requests.values()),
                'observation_only': True, 'scheduling_changes': False if self.arm == 'A' else True}
        if self.engine is not None:
            manager = self.engine.engine.scheduler[0].block_manager
            scheduler = self.engine.engine.scheduler[0]
            if hasattr(scheduler, 'reservation_events'):
                events = scheduler.reservation_events
                with (self.path / 'reservation_events.jsonl').open('x') as f:
                    for e in events:
                        f.write(json.dumps(e) + '\n')
                data['reservation_events'] = len(events)
            data['final_kv_free'] = {'gpu_blocks': manager.get_num_free_gpu_blocks(),
                                     'cpu_blocks': manager.get_num_free_cpu_blocks()}
        p = self.path / 'common_audit_summary.json'
        p.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + '\n')
        self.handle.close()
        self.closed = True


def install_observation(server, engine_cls, output_dir, arm, root, device):
    observation = Observation(output_dir, arm, root, device)
    original = engine_cls.generate
    @functools.wraps(original)
    async def generate(self, prompt, request_info, sampling_params, request_id, *args, **kwargs):
        c, s, member = (request_info.collection_id, request_info.stage_id, request_info.request_id)
        if c == 0 and s == 0:
            observation.zero_starts += 1
            if observation.zero_starts >= 2:
                observation.main = True
        phase = 'main' if observation.main else 'warmup'
        generation = {k: getattr(sampling_params, k, None) for k in
                      ['temperature', 'top_p', 'top_k', 'seed', 'ignore_eos', 'max_tokens',
                       'min_tokens', 'target_output_length', 'n']}
        row = {'request_uuid': request_id, 'phase': phase, 'collection_id': c,
               'stage_id': s, 'request_id': member,
               'prompt_sha256': hashlib.sha256(str(prompt).encode()).hexdigest(),
               'generation': generation, 'received_slo': list(request_info.slo_constraint),
               'start_monotonic': time.perf_counter(), 'finished': False}
        observation.requests[request_id] = row
        observation.record('request_start', **row)
        async for output in original(self, prompt, request_info, sampling_params, request_id, *args, **kwargs):
            if output.finished:
                row.update(finished=True, finish_monotonic=time.perf_counter(),
                           prompt_tokens=len(output.prompt_token_ids),
                           output_tokens=sum(len(x.token_ids) for x in output.outputs),
                           finish_reasons=[x.finish_reason for x in output.outputs],
                           stop_reasons=[x.stop_reason for x in output.outputs],
                           token_ids_sha256=hashlib.sha256(json.dumps(
                               [list(x.token_ids) for x in output.outputs]).encode()).hexdigest())
                observation.record('request_finish', **row)
            yield output
    engine_cls.generate = generate
    return observation
