"""Native scheduler/block-manager regression for reversible expired-root draining."""
import time
import unittest
from unittest.mock import patch
from vllm.config import CacheConfig,SchedulerConfig
from vllm.sequence import SequenceStatus
from jitserve.request_info import RequestType
from mixed_joint import MixedScheduler
import native_chain
import controlled_native as reference


class ExpiredDrain(unittest.TestCase):
    def setUp(self):
        self.fixture=reference.NativeRuntime();self.model=reference.DummyModel()
        self.p=patch.object(native_chain,'MODEL',self.model);self.p.start();self.addCleanup(self.p.stop)

    def scheduler(self):
        cfg=SchedulerConfig('generate',64,2,512,enable_chunked_prefill=True,
                            policy='jitserve',top_k_selection=3,max_swaps_per_iter=3)
        cache=CacheConfig(16,.9,1,'auto');cache.num_gpu_blocks=128;cache.num_cpu_blocks=32
        return MixedScheduler(cfg,cache,None)

    def group(self,rid,deadline=900.,prompt=64):
        return self.fixture.group(str(rid),deadline,prompt)

    def test_expired_backlog_bypasses_forecast_and_expands_only_within_pool(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler()
            for rid in range(1,13):s.add_seq_group(self.group(rid))
            out=self.fixture.step(s)
            self.assertTrue(s.fast_expired_drain);self.assertEqual(self.model.calls,0)
            self.assertGreater(s.scheduler_config.max_num_seqs,2)
            batches=[len(out.scheduled_seq_groups)]
            for _ in range(20):batches.append(len(self.fixture.step(s).scheduled_seq_groups))
            self.assertGreater(max(batches),2)
            self.assertEqual(s.cache_config.num_gpu_blocks,128)
            self.assertLessEqual(s.scheduler_config.max_num_seqs,64)
            s.abort_seq_group([str(i)for i in range(1,13)])
            self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)

    def test_new_live_root_exits_fast_drain_and_restores_prior_limit(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler()
            for rid in range(1,5):s.add_seq_group(self.group(rid))
            self.fixture.step(s);self.assertGreater(s.scheduler_config.max_num_seqs,2)
            s.add_seq_group(self.group(99,2000.));self.fixture.step(s)
            self.assertFalse(s.fast_expired_drain)
            self.assertEqual(s.scheduler_config.max_num_seqs,2)
            self.assertGreater(self.model.calls,0)
            s.abort_seq_group([str(i)for i in [1,2,3,4,99]])
            self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)

    def test_unsubmitted_next_stage_keeps_root_credit_live(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();g=self.group(1,1100.)
            g.request_info.chain_stage_count=2;g.request_info.chain_stage_widths=[1,1]
            s.add_seq_group(g);self.fixture.step(s)
            g.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED;s.free_finished_seq_groups()
            self.assertFalse(s.running or s.waiting or s.swapped)
            self.assertTrue(s.policy.has_live_credit(1000.))
            s.add_seq_group(self.group(2,900.));self.fixture.step(s)
            self.assertFalse(s.fast_expired_drain)
            with patch('time.perf_counter',return_value=1101.):self.fixture.step(s)
            self.assertTrue(s.fast_expired_drain)
            s.abort_seq_group('2')

    def test_zero_model_probability_is_not_proof_of_expiry(self):
        self.model.drain=True
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();s.add_seq_group(self.group(1,2000.))
            for _ in range(10):self.fixture.step(s)
            self.assertTrue(s.policy.draining);self.assertFalse(s.fast_expired_drain)
            self.assertEqual(s.scheduler_config.max_num_seqs,2)
            s.abort_seq_group('1')

    def test_new_foreground_restores_limits_and_remains_executable(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler()
            for rid in range(1,5):s.add_seq_group(self.group(rid))
            self.fixture.step(s)
            fg=self.group(99,2000.);fg.request_type=RequestType.LATENCY
            fg.request_info.request_type=RequestType.LATENCY;s.add_seq_group(fg)
            seen=False
            for _ in range(10):
                out=self.fixture.step(s)
                seen |= any(item.seq_group.request_id=='99'for item in out.scheduled_seq_groups)
            self.assertFalse(s.fast_expired_drain);self.assertEqual(s.scheduler_config.max_num_seqs,2)
            self.assertTrue(seen)
            s.abort_seq_group(['1','2','3','4','99'])

    def test_all_expired_calls_finish_without_losing_emitted_prefix_or_kv(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();groups=[self.group(rid,prompt=64+(rid%3)*128)for rid in range(1,21)]
            for g in groups:s.add_seq_group(g)
            previous={g.request_id:0 for g in groups}
            for tick in range(1000):
                self.fixture.step(s)
                for g in groups:
                    n=g.first_seq.get_output_len();self.assertGreaterEqual(n,previous[g.request_id]);previous[g.request_id]=n
                    if n>=20:g.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED
                s.free_finished_seq_groups()
                if not(s.running or s.waiting or s.swapped):break
            self.assertLess(tick,999);self.assertTrue(all(n==20 for n in previous.values()))
            self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)
            self.assertEqual(s.block_manager.get_num_free_cpu_blocks(),32)
            self.assertEqual(self.model.calls,0)

    def test_reused_root_id_cannot_reactivate_stale_deadline_heap_entry(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();g=self.group(1,2000.);s.add_seq_group(g);self.fixture.step(s)
            g.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED;s.free_finished_seq_groups()
            h=self.group(1,900.);s.add_seq_group(h)
            self.assertFalse(s.policy.has_live_credit(1000.))
            s.abort_seq_group('1')


if __name__=='__main__':unittest.main(verbosity=2)
