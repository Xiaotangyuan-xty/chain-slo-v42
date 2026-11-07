import unittest
from unittest.mock import patch
from vllm.config import CacheConfig,SchedulerConfig
from vllm.sequence import SequenceStatus
from vllm import SamplingParams
from jitserve.request_info import RequestType
from jitserve.scheduler.slo_scheduler import SLOScheduler
import native_chain
import controlled_native as reference
from mixed_joint import MixedScheduler


class MixedRuntime(unittest.TestCase):
    def setUp(self):
        self.model=reference.DummyModel()
        p=patch.object(native_chain,'MODEL',self.model);p.start();self.addCleanup(p.stop)
        self.fixture=reference.NativeRuntime()

    def scheduler(self,cpu=32,gpu=128,cls=MixedScheduler):
        cfg=SchedulerConfig('generate',64,32,512,enable_chunked_prefill=True,
                            policy='jitserve',top_k_selection=3,max_swaps_per_iter=3)
        cache=CacheConfig(16,.9,1,'auto');cache.num_gpu_blocks=gpu;cache.num_cpu_blocks=cpu
        return cls(cfg,cache,None)

    def group(self,rid,kind=2,prompt=64,deadline=2000.):
        group=self.fixture.group(str(rid),deadline,prompt)
        group.request_type=RequestType(kind);group.request_info.request_type=RequestType(kind)
        return group

    def test_new_foreground_is_eligible_before_50ticks(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();s.add_seq_group(self.group(1,0));s.add_seq_group(self.group(2))
            for _ in range(8):self.fixture.step(s)
            old=s.last_plan;s.add_seq_group(self.group(3,0));out=self.fixture.step(s)
            self.assertGreater(s.last_plan,old)
            self.assertIn('new_request_or_stage',s.last_joint_summary['reasons'])
            self.assertIn('3',[x.seq_group.request_id for x in out.scheduled_seq_groups])
            s.abort_seq_group(['1','2','3']);self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)

    def test_next_stage_arrives_while_other_selected_work_is_live(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();fg=self.group(1,0);first=self.group(2)
            first.request_info.chain_stage_count=2;first.request_info.chain_stage_widths=[1,1]
            s.add_seq_group(fg);s.add_seq_group(first)
            for _ in range(5):self.fixture.step(s)
            first.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED;s.free_finished_seq_groups()
            new=self.group(3);new.collection_id=2;new.request_info.collection_id=2
            new.request_info.stage_id=1;new.request_info.chain_stage_count=2;new.request_info.chain_stage_widths=[1,1]
            s.add_seq_group(new);old=s.last_plan;out=self.fixture.step(s)
            self.assertGreater(s.last_plan,old);self.assertIn('3',[x.seq_group.request_id for x in out.scheduled_seq_groups])
            s.abort_seq_group(['1','3']);self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)

    def test_prediction_revision_triggers_plan_without_membership_change(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();fg=self.group(1,0);s.add_seq_group(fg);s.add_seq_group(self.group(2))
            for _ in range(5):self.fixture.step(s)
            old=s.last_plan;fg.request_info.prediction_revision=1;self.fixture.step(s)
            self.assertGreater(s.last_plan,old);self.assertIn('prediction_revision',s.last_joint_summary['reasons'])
            s.abort_seq_group(['1','2'])

    def test_zero_probability_chain_backfills_with_positive_chain_and_foreground(self):
        def estimate(meta,stage,views,slack,unit):
            p=0. if slack>500 else 1.
            return {'probability':p,'score':10*p,'remaining_work':100.,'critical_mean_seconds':1.,'current_remaining_tokens':50.,'donors':1}
        with patch.object(self.model,'estimate',side_effect=estimate),patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();groups=[self.group(1,0),self.group(2,deadline=1100.),self.group(3,deadline=2000.)]
            for g in groups:s.add_seq_group(g)
            for _ in range(20):self.fixture.step(s)
            self.assertTrue(all(g.first_seq.get_output_len()>0 for g in groups))
            self.assertGreater(s.mixed_counts['backfill_members'],0)
            s.abort_seq_group(['1','2','3']);self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)

    def exercise_pressure(self,cpu):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler(cpu=cpu,gpu=20);fg=self.group(1,1,prompt=160);s.add_seq_group(fg)
            for _ in range(4):self.fixture.step(s)
            prefix=fg.first_seq.get_output_len();self.assertGreater(prefix,0)
            chain=self.group(2,prompt=160,deadline=1100.);s.add_seq_group(chain)
            # Foreground has missed its20s deadline and may yield real capacity.
            with patch('time.time',return_value=130.):
                for _ in range(6):
                    out=self.fixture.step(s)
                    if fg.first_seq.status!=SequenceStatus.RUNNING:break
            self.assertEqual(fg.first_seq.get_output_len(),prefix)
            self.assertIn(fg.first_seq.status,[SequenceStatus.SWAPPED,SequenceStatus.WAITING])
            if cpu==1:self.assertFalse(out.blocks_to_swap_out)
            else:self.assertTrue(out.blocks_to_swap_out)
            s.abort_seq_group('2')
            for _ in range(12):
                self.fixture.step(s)
                if fg.first_seq.get_output_len()>prefix:break
            self.assertGreater(fg.first_seq.get_output_len(),prefix)
            s.abort_seq_group('1')
            self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),20)
            self.assertEqual(s.block_manager.get_num_free_cpu_blocks(),cpu)

    def test_mixed_pressure_native_swap_resumes_prefix(self):self.exercise_pressure(32)
    def test_mixed_pressure_native_recompute_resumes_prefix(self):self.exercise_pressure(1)

    def test_pure_chain_trace_matches_unchanged_v47_loop(self):
        def trace(cls):
            s=self.scheduler(cls=cls)
            self.model.calls=0
            for rid in [1,2,3]:s.add_seq_group(self.group(rid))
            rows=[]
            for _ in range(140):
                out=self.fixture.step(s)
                rows.append((tuple(x.seq_group.request_id for x in out.scheduled_seq_groups),tuple(out.blocks_to_swap_out),tuple(out.blocks_to_swap_in),tuple(sorted(s.selected))if s.selected is not None else None,s.policy.draining,s.last_plan,dict(s.leases)))
            return rows,self.model.calls
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            self.assertEqual(trace(native_chain.NativeChainScheduler),trace(MixedScheduler))

    def test_foreground_only_stays_native_and_all_abort_frees_registries(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();s.add_seq_group(self.group(1,0))
            with patch.object(SLOScheduler,'_schedule',autospec=True,return_value='native')as call:
                self.assertEqual(s._schedule(),'native');call.assert_called_once_with(s)
            s.add_seq_group(self.group(2));self.fixture.step(s);s.abort_seq_group(['1','2'])
            self.assertFalse(s.running or s.waiting or s.swapped)
            self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)

    def test_mixed_to_chain_only_replans_immediately_and_retains_prefix(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();fg=self.group(1,0);chain=self.group(2)
            s.add_seq_group(fg);s.add_seq_group(chain)
            for _ in range(5):self.fixture.step(s)
            old_calls=self.model.calls;prefix=chain.first_seq.get_output_len()
            fg.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED;s.free_finished_seq_groups()
            self.fixture.step(s)
            self.assertGreater(self.model.calls,old_calls)
            self.assertGreaterEqual(chain.first_seq.get_output_len(),prefix)
            self.assertEqual(s.last_plan,s.policy.num_schedule_count)
            self.assertFalse(s.policy.joint_mode)
            s.abort_seq_group('2');self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)


if __name__=='__main__':unittest.main(verbosity=2)
