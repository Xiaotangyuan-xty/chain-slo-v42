import unittest
from unittest.mock import patch
from vllm.config import CacheConfig,SchedulerConfig
from vllm.sequence import SequenceStatus
from jitserve.request_info import RequestType
from jitserve.scheduler.slo_scheduler import SLOScheduler
from chain_forecast import StageForecaster
import native_chain
import controlled_native as reference
from mixed_joint import MixedScheduler


def model():
    return StageForecaster({'neighbors':1,'prefill_tokens_per_decode_unit':256,
          'interstage_gap_decode_units':1,'forecast_guard_decode_units':0,
          'service_safety_factor':1.,'records':[{'id':1,'stages':[
              {'inputs':[64],'outputs':[64]},{'inputs':[64],'outputs':[64]}]}]})


class Runtime(unittest.TestCase):
    def setUp(self):
        self.fixture=reference.NativeRuntime();self.model=model()
        p=patch.object(native_chain,'MODEL',self.model);p.start();self.addCleanup(p.stop)

    def scheduler(self,cpu=32,gpu=128,cls=MixedScheduler):
        cfg=SchedulerConfig('generate',64,32,512,enable_chunked_prefill=True,
                            policy='jitserve',top_k_selection=3,max_swaps_per_iter=3)
        cache=CacheConfig(16,.9,1,'auto');cache.num_gpu_blocks=gpu;cache.num_cpu_blocks=cpu
        return cls(cfg,cache,None)

    def group(self,rid,kind=2,prompt=64,deadline=1100.):
        g=self.fixture.group(str(rid),deadline,prompt)
        g.request_type=RequestType(kind);g.request_info.request_type=RequestType(kind)
        return g

    def finish(self,g):g.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED

    def test_new_request_and_revision_replan_before_periodic_frame(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();s.add_seq_group(self.group(1,0));s.add_seq_group(self.group(2))
            for _ in range(6):self.fixture.step(s)
            last=s.last_plan;s.add_seq_group(self.group(3,0));self.fixture.step(s)
            self.assertGreater(s.last_plan,last);self.assertIn('new_request_or_stage',s.last_joint_summary['reasons'])
            groups=list(s.running)+list(s.waiting)+list(s.swapped);groups[0].request_info.prediction_revision=1
            last=s.last_plan;self.fixture.step(s);self.assertGreater(s.last_plan,last)
            self.assertIn('prediction_revision',s.last_joint_summary['reasons'])
            s.abort_seq_group(['1','2','3']);self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)

    def test_free_KV_does_not_force_expired_work_into_decode_batch(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();live=self.group(1,0);expired=self.group(2,deadline=999.)
            s.add_seq_group(live);s.add_seq_group(expired)
            for _ in range(8):out=self.fixture.step(s)
            self.assertTrue(s.selected is None or '2' not in s.selected)
            self.assertTrue(s.last_joint_summary['detail']['reason'] in ['native_required_work_fallback','anytime_stage_window_token_credit'])
            self.assertLess(s.last_joint_summary['detail']['selected_charge'],128)
            self.finish(live);s.free_finished_seq_groups()
            for _ in range(8):self.fixture.step(s)
            self.assertGreater(expired.first_seq.get_output_len(),0)
            s.abort_seq_group('2');self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)

    def exercise_compute_swap(self,cpu):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler(cpu=cpu,gpu=128);old=self.group(1,deadline=1100.)
            s.add_seq_group(old)
            for _ in range(5):self.fixture.step(s)
            prefix=old.first_seq.get_output_len();self.assertGreater(prefix,0)
            # Model expiry is a real root deadline, independent of a benchmark end.
            s.policy.chains[1]['deadline']=999.;old.request_info.chain_deadline=999.
            live=self.group(2,0);s.add_seq_group(live)
            with patch('mixed_joint.plan',return_value=({'2'},{'backfill':[],'evicted':['1'],'actions':[]})):
                out=self.fixture.step(s)
            self.assertNotIn('1',s.selected);self.assertIn('2',s.selected)
            self.assertEqual(old.first_seq.get_output_len(),prefix)
            self.assertIn(old.first_seq.status,[SequenceStatus.SWAPPED,SequenceStatus.WAITING])
            if cpu==1:self.assertFalse(out.blocks_to_swap_out)
            else:self.assertTrue(out.blocks_to_swap_out)
            self.assertFalse(out.blocks_to_swap_in)
            for _ in range(6):self.fixture.step(s)
            self.assertGreater(live.first_seq.get_output_len(),0)
            self.finish(live);s.free_finished_seq_groups()
            for _ in range(10):
                self.fixture.step(s)
                if old.first_seq.get_output_len()>prefix:break
            self.assertGreater(old.first_seq.get_output_len(),prefix)
            s.abort_seq_group('1');self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)
            self.assertEqual(s.block_manager.get_num_free_cpu_blocks(),cpu)

    def test_compute_interference_can_swap_with_free_KV_and_resume(self):self.exercise_compute_swap(32)
    def test_compute_interference_can_recompute_when_CPU_swap_full(self):self.exercise_compute_swap(1)

    def test_multi_member_focus_preserves_prefix_and_releases_both_pools(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler(cpu=4,gpu=256);groups=[self.group(i,deadline=1100.)for i in range(1,9)]
            for g in groups:s.add_seq_group(g)
            for _ in range(5):self.fixture.step(s)
            prior={g.request_id:g.first_seq.get_output_len()for g in groups}
            for g in groups:s.policy.chains[g.collection_id]['deadline']=999.;g.request_info.chain_deadline=999.
            fg=self.group(99,0);s.add_seq_group(fg)
            with patch('mixed_joint.plan',return_value=({'99'},{'backfill':[],'evicted':[g.request_id for g in groups],'actions':[]})):
                for _ in range(12):self.fixture.step(s)
            for g in groups:self.assertEqual(g.first_seq.get_output_len(),prior[g.request_id])
            self.assertGreater(fg.first_seq.get_output_len(),0)
            self.finish(fg);s.free_finished_seq_groups()
            for _ in range(100):
                self.fixture.step(s)
                if all(g.first_seq.get_output_len()>prior[g.request_id]for g in groups):break
            self.assertTrue(all(g.first_seq.get_output_len()>prior[g.request_id]for g in groups))
            s.abort_seq_group([g.request_id for g in groups]);self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),256);self.assertEqual(s.block_manager.get_num_free_cpu_blocks(),4)

    def test_next_stage_event_keeps_native_barrier_metadata(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();fg=self.group(1,0);first=self.group(2)
            first.request_info.chain_stage_count=2;first.request_info.chain_stage_widths=[1,1]
            s.add_seq_group(fg);s.add_seq_group(first)
            for _ in range(4):self.fixture.step(s)
            self.finish(first);s.free_finished_seq_groups()
            next_stage=self.group(3);next_stage.collection_id=2;next_stage.request_info.collection_id=2
            next_stage.request_info.stage_id=1;next_stage.request_info.chain_stage_count=2;next_stage.request_info.chain_stage_widths=[1,1]
            s.add_seq_group(next_stage);last=s.last_plan;self.fixture.step(s)
            self.assertGreater(s.last_plan,last);self.assertIn('new_request_or_stage',s.last_joint_summary['reasons'])
            self.assertIn(0,s.policy.chains[2]['observed'])
            s.abort_seq_group(['1','3']);self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)

    def test_real_deadline_crossing_replans_without_waiting_50_ticks(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();s.add_seq_group(self.group(1,0));s.add_seq_group(self.group(2,deadline=1001.))
            for _ in range(5):self.fixture.step(s)
            old=s.last_plan
            with patch('time.perf_counter',return_value=1001.1):self.fixture.step(s)
            self.assertGreater(s.last_plan,old)
            self.assertIn('deadline_or_membership_change',s.last_joint_summary['reasons'])
            self.assertTrue(s.selected is None or '2'not in s.selected)
            s.abort_seq_group(['1','2'])

    def test_scheduler_intervals_are_never_mixed_into_executor_cost_samples(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();s.add_seq_group(self.group(1,0));s.add_seq_group(self.group(2))
            for _ in range(5):self.fixture.step(s)
            self.assertFalse(s.cost_model.samples)
            self.assertIn('scheduler_seconds',s._execution_flags)
            s.abort_seq_group(['1','2'])

    def test_selected_members_only_and_no_work_disappears(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();groups=[self.group(1,0),self.group(2),self.group(3,deadline=999.)]
            for g in groups:s.add_seq_group(g)
            for _ in range(30):
                out=self.fixture.step(s)
                if not s.policy.draining:self.assertTrue({x.seq_group.request_id for x in out.scheduled_seq_groups}<=s.last_recorded_selected)
                self.assertEqual({g.request_id for g in list(s.running)+list(s.waiting)+list(s.swapped)},{'1','2','3'})
            s.abort_seq_group(['1','2','3']);self.assertFalse(s.running or s.waiting or s.swapped)

    def test_required_mixed_work_finishes_without_drops_or_lost_prefix(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();groups=[self.group(1,0),self.group(2),self.group(3,deadline=999.)]
            for g in groups:s.add_seq_group(g)
            previous={g.request_id:0 for g in groups}
            for tick in range(500):
                out=self.fixture.step(s)
                for item in out.scheduled_seq_groups:
                    g=item.seq_group;n=g.first_seq.get_output_len()
                    self.assertGreaterEqual(n,previous[g.request_id]);previous[g.request_id]=n
                    if n>=20:self.finish(g)
                s.free_finished_seq_groups()
                if not(s.running or s.waiting or s.swapped):break
            self.assertLess(tick,499);self.assertTrue(all(n==20 for n in previous.values()))
            self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)
            self.assertEqual(s.block_manager.get_num_free_cpu_blocks(),32)

    def test_pure_DR_decisions_still_match_original_v47(self):
        def trace(cls):
            s=self.scheduler(cls=cls)
            for rid in [1,2,3]:s.add_seq_group(self.group(rid))
            result=[]
            for _ in range(120):
                out=self.fixture.step(s);result.append((tuple(x.seq_group.request_id for x in out.scheduled_seq_groups),
                    tuple(out.blocks_to_swap_out),tuple(out.blocks_to_swap_in),
                    tuple(sorted(s.selected))if s.selected is not None else None,s.policy.draining,s.last_plan,dict(s.leases)))
            s.abort_seq_group(['1','2','3']);return result
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            self.assertEqual(trace(native_chain.NativeChainScheduler),trace(MixedScheduler))

    def test_mixed_to_pure_replans_and_foreground_only_delegates(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();fg=self.group(1,0);chain=self.group(2)
            s.add_seq_group(fg)
            with patch.object(SLOScheduler,'_schedule',autospec=True,return_value=None):
                # Exercise dispatch only: production _schedule always receives SchedulerOutputs.
                self.assertIsNone(s._dispatch())
            s.add_seq_group(chain)
            for _ in range(4):self.fixture.step(s)
            s.abort_seq_group('1');self.fixture.step(s)
            self.assertFalse(s.policy.joint_mode);self.assertEqual(s.last_plan,s.policy.num_schedule_count)
            s.abort_seq_group('2');self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)


if __name__=='__main__':unittest.main(verbosity=2)
