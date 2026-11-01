"""CPU runtime checks using the real pinned vLLM scheduler/block manager."""
import time
import unittest
from types import SimpleNamespace as S

from deadline_reservation import Member, reserve


class Reservations(unittest.TestCase):
    def member(self, request, chain=1, deadline=100., prompt=16, **kw):
        return Member(request, chain, deadline, kw.get('arrival', 1.), prompt,
                      kw.get('generated', 0), kw.get('prediction', 16),
                      kw.get('max_tokens', 64))

    def test_whole_chain_edf_not_member_reward_or_id(self):
        members = [self.member('a', chain=999, deadline=20),
                   self.member('b', chain=-3, deadline=80),
                   self.member('c', chain=-3, deadline=80)]
        chosen, _, _ = reserve(members, 10, 5, 32)
        self.assertEqual(chosen, {'a'})

    def test_compound_barrier_and_cap(self):
        members = [self.member(str(i), chain=i//3) for i in range(6)]
        chosen, _, used = reserve(members, 10, 15, 32)
        self.assertEqual(chosen, {'0','1','2'})
        self.assertEqual(used, 15)

    def test_late_work_drains_after_on_time(self):
        late = self.member('late', deadline=1)
        ready = self.member('ready', chain=2, deadline=100)
        self.assertEqual(reserve([late, ready], 10, 100, 32)[0], {'ready'})
        self.assertEqual(reserve([late], 10, 100, 32)[0], {'late'})

    def test_progress_overrides_underprediction(self):
        item = self.member('x', prompt=100, generated=500, prediction=1, max_tokens=1024)
        self.assertEqual(item.blocks(), 41)

    def test_oversized_compound_makes_progress(self):
        items = [self.member(str(i), prompt=64) for i in range(3)]
        selected, _, _ = reserve(items, 10, 8, 32)
        self.assertEqual(selected, {'0'})


class NativeRuntime(unittest.TestCase):
    def make_scheduler(self):
        from vllm.config import CacheConfig, SchedulerConfig
        from native_chain import NativeChainScheduler
        config = SchedulerConfig('generate', 64, 32, 256,
                                 enable_chunked_prefill=True, policy='jitserve',
                                 top_k_selection=3, max_swaps_per_iter=3)
        cache = CacheConfig(16, .9, 1, 'auto')
        cache.num_gpu_blocks, cache.num_cpu_blocks = 16, 32
        return NativeChainScheduler(config, cache, None)

    def group(self, rid, chain, deadline, prompt=64):
        from jitserve.request_info import RequestInfo, RequestType
        from vllm.sequence import Sequence, SequenceGroup
        from vllm import SamplingParams
        info = RequestInfo(RequestType.COLLECTIVE, (2., .1, 20.), 1,
                           chain, 120, prompt, 16)
        info.chain_deadline = deadline
        seq = Sequence(int(rid), inputs={'prompt_token_ids': list(range(prompt)),
                                        'prompt': 'cpu'}, block_size=16)
        return SequenceGroup(rid, [seq], time.time(), info,
                             SamplingParams(temperature=0, max_tokens=64))

    def advance(self, metadata, outputs):
        from vllm.sequence import Logprob
        for item in outputs.scheduled_seq_groups:
            item.seq_group.update_num_computed_tokens(item.token_chunk_size)
            if not item.seq_group.is_prefill():
                item.seq_group.first_seq.append_token_id(1, {1: Logprob(0.)})

    def test_real_chunked_prefill_decode_swap_resume_and_release(self):
        from vllm.sequence import SequenceStatus
        scheduler = self.make_scheduler()
        late = self.group('1', 101, time.perf_counter()+1000, prompt=160)
        scheduler.add_seq_group(late)
        for _ in range(3):
            metadata, output, _ = scheduler.schedule()
            self.advance(metadata, output)
        self.assertFalse(late.is_prefill())
        urgent = self.group('2', 202, time.perf_counter()+100, prompt=160)
        scheduler.add_seq_group(urgent)
        metadata, output, _ = scheduler.schedule()
        self.assertTrue(output.blocks_to_swap_out)
        self.assertFalse(output.blocks_to_swap_in)
        self.assertFalse(output.scheduled_seq_groups)
        self.assertEqual(late.first_seq.status, SequenceStatus.SWAPPED)
        for _ in range(3):
            metadata, output, _ = scheduler.schedule()
            self.advance(metadata, output)
        self.assertFalse(urgent.is_prefill())
        scheduler.abort_seq_group('2')
        metadata, output, _ = scheduler.schedule()
        self.assertTrue(output.blocks_to_swap_in)
        self.assertEqual(late.first_seq.status, SequenceStatus.RUNNING)
        self.advance(metadata, output)
        scheduler.abort_seq_group('1')
        self.assertEqual(scheduler.block_manager.get_num_free_gpu_blocks(), 16)
        self.assertEqual(scheduler.block_manager.get_num_free_cpu_blocks(), 32)

    def test_deadline_capture_uses_received_metadata_only(self):
        from native_chain import install
        server = S(_handle_collective_request=lambda info, data, prompt: (True, info.collection_id))
        install(server)
        info = S(collection_id=999, slo_constraint=(2., .1, 40.))
        now = time.perf_counter()
        server._handle_collective_request(info, {'num_stages': 4, 'served_time': 60}, 'prompt')
        self.assertAlmostEqual(info.chain_deadline-now, 100, delta=.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
