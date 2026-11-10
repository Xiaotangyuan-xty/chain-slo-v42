"""Pure golden traces and actual native mixed continuity/resource contracts."""
from copy import copy
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest

from test_pure_core_restore import PureCoreTraceTests
from mixed_joint import MixedScheduler as Original
from mixed_max import MixedScheduler, marginal_foreground
from vllm.config import CacheConfig
from vllm.sequence import SequenceStatus


class MaxPureTraceTests(PureCoreTraceTests):
    def scheduler(self, blocks=64, slots=4):
        original = super().scheduler(blocks, slots)
        # Use the exact same config and native block manager configuration.
        cache = copy(original.cache_config)
        return MixedScheduler(self.shared, cache, None)

    def reference(self, blocks=64, slots=4):
        cache = CacheConfig(16, .9, 1, 'auto')
        cache.num_gpu_blocks = blocks; cache.num_cpu_blocks = 64
        return Original(copy(self.shared), cache, None)

    def test_pure_inherited_helpers_are_original_functions(self):
        for name in ['_preempt', '_chunked', '_clear_plan', '_adjust_batch_limits',
                     '_expired_drain', '_restore_drain_limits', 'add_seq_group']:
            self.assertIs(getattr(MixedScheduler, name), getattr(Original, name), name)


class MaxMixedTests(MaxPureTraceTests):
    def test_plan_change_cannot_evict_a_running_decode(self):
        s = self.scheduler(blocks=128)
        fg = self.group(91, kind=0, prompt=16)
        s.add_seq_group(fg)
        for _ in range(3):self.step(s)
        self.assertIn(fg, s.running)
        before = fg.first_seq.get_output_len()
        with patch.object(s, '_preempt', wraps=s._preempt) as preempt:
            out = s._max_execute(set(), s.base_max_num_seqs)
            self.assertIn(fg, s.running)
            self.assertIn(fg.request_id, s.last_recorded_selected)
            self.assertEqual(preempt.call_count, 0)
            self.assertTrue(out.scheduled_seq_groups)
        self.assertEqual(fg.first_seq.get_output_len(), before)
        self.cleanup(s)

    def test_foreground_only_uses_no_chain_forecast_and_keeps_progress(self):
        s = self.scheduler(blocks=128)
        groups = [self.group(90+i, kind=i % 2, prompt=16) for i in range(4)]
        for g in groups:s.add_seq_group(g)
        with patch('mixed_max.MODEL.estimate', side_effect=AssertionError('No chain exists')):
            for _ in range(12):self.step(s)
        self.assertTrue(all(g.first_seq.get_output_len() >= 8 for g in groups))
        self.assertEqual(s.v47_counts.get('all_preemptions', 0), 0)
        self.cleanup(s)

    def test_marginal_reward_excludes_settled_latency_credit(self):
        g = NS(request_type=NS(value=0), first_seq=NS(get_output_len=lambda:10,
            get_prompt_len=lambda:16), request_info=NS(output_len=20),
            concord_metrics=NS(service_gain=900.), is_prefill=lambda:False,
            arrival_time=100., TTFT_constraint=2., TBT_constraint=.1)
        policy = NS(physical_running=[], physical_swapped=[])
        service = NS(recovery=lambda view:0., last_round=.05)
        with patch('mixed_max.SLOPolicy.reward_estimation', return_value=(1000.,5.)):
            self.assertAlmostEqual(marginal_foreground(policy,g,service,100.)['score'],20.)

    def test_prediction_revision_invalidates_admission_without_evicting(self):
        s = self.scheduler(blocks=128)
        g = self.group(92, kind=0, prompt=16);s.add_seq_group(g)
        for _ in range(3):self.step(s)
        before=s.max_counts['mixed_plans']
        g.request_info.prediction_revision=1;g.request_info.output_len+=20
        self.step(s)
        self.assertGreater(s.max_counts['mixed_plans'],before)
        self.assertIn(g,s.running)
        self.assertEqual(s.v47_counts.get('all_preemptions',0),0)
        self.cleanup(s)

    def test_mixed_queues_are_restored_and_all_KV_is_freed(self):
        s=self.scheduler(blocks=20,slots=2)
        groups=[self.group(100+i,kind=i%2,prompt=128)for i in range(6)]
        for g in groups:s.add_seq_group(g)
        previous={g.request_id:0 for g in groups}
        for _ in range(220):
            self.step(s)
            for g in groups:
                value=g.first_seq.get_output_len()
                self.assertGreaterEqual(value,previous[g.request_id]);previous[g.request_id]=value
                if value>=8:g.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED
            s.free_finished_seq_groups()
            if all(g.is_finished()for g in groups):break
        self.assertTrue(all(g.first_seq.get_output_len()==8 for g in groups))
        self.assertEqual(s.v47_counts.get('all_preemptions',0),0)
        self.cleanup(s)


if __name__=='__main__':unittest.main(verbosity=2)
