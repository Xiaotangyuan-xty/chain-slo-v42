"""Native scheduler/block-manager CPU fixtures; no model or CUDA execution."""
from pathlib import Path
import unittest
from unittest.mock import patch

from vllm.config import CacheConfig, SchedulerConfig
from vllm.sequence import SequenceStatus, Sequence, SequenceGroup
from vllm import SamplingParams
from jitserve.request_info import RequestInfo, RequestType
from jitserve.scheduler.slo_scheduler import SLOScheduler
import native_chain
import controlled_native as reference
from mixed_joint import MixedScheduler


class MixedRuntime(unittest.TestCase):
    def setUp(self):
        self.model=reference.DummyModel()
        self.patch=patch.object(native_chain,'MODEL',self.model);self.patch.start()
        self.addCleanup(self.patch.stop)
        self.fixture=reference.NativeRuntime()

    def scheduler(self,cls=MixedScheduler,cpu=32):
        cfg=SchedulerConfig('generate',64,32,512,enable_chunked_prefill=True,
                            policy='jitserve',top_k_selection=3,max_swaps_per_iter=3)
        cache=CacheConfig(16,.9,1,'auto');cache.num_gpu_blocks=20;cache.num_cpu_blocks=cpu
        return cls(cfg,cache,None)

    def group(self,rid,kind=2,prompt=64):
        info=RequestInfo(RequestType(kind),(2.,.1,20.),0,rid,120,prompt,64)
        if kind==2:
            info.chain_deadline=2000.;info.chain_stage_count=1;info.chain_stage_widths=[1]
        seq=Sequence(rid,inputs={'prompt_token_ids':list(range(prompt)),'prompt':'CPU'},block_size=16)
        return SequenceGroup(str(rid),[seq],100.,info,SamplingParams(temperature=0,max_tokens=256))

    def finish(self,scheduler,out):
        for item in out.scheduled_seq_groups:
            if item.seq_group.first_seq.get_output_len()>=64:
                item.seq_group.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED
        scheduler.free_finished_seq_groups()

    def test_joint_model_execution_serves_foreground_and_chains_and_frees_kv(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();seen=set()
            for rid,kind in [(1,0),(2,1),(3,2),(4,2)]:s.add_seq_group(self.group(rid,kind))
            for tick in range(1500):
                out=self.fixture.step(s)
                seen.update(item.seq_group.request_type.value for item in out.scheduled_seq_groups)
                self.finish(s,out)
                if not(s.running or s.waiting or s.swapped):break
            self.assertLess(tick,1499);self.assertEqual(seen,{0,1,2})
            self.assertGreater(s.mixed_counts['joint_plans_selecting_both_types'],0)
            self.assertGreater(self.model.calls,0)
            self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),20)
            self.assertEqual(s.block_manager.get_num_free_cpu_blocks(),32)

    def test_pure_chain_delegates_to_current_chain_loop(self):
        def trace(cls):
            s=self.scheduler(cls);self.model.calls=0
            for rid in [1,2,3]:s.add_seq_group(self.group(rid))
            rows=[]
            for tick in range(140):
                out=self.fixture.step(s)
                rows.append((tuple(x.seq_group.request_id for x in out.scheduled_seq_groups),
                             tuple(out.blocks_to_swap_out),tuple(out.blocks_to_swap_in),
                             tuple(sorted(s.selected)) if s.selected is not None else None,
                             s.policy.draining,s.last_plan,dict(s.leases)))
            return rows,self.model.calls
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            self.assertEqual(trace(native_chain.NativeChainScheduler),trace(MixedScheduler))

    def test_foreground_only_uses_original_native_loop(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();s.add_seq_group(self.group(1,0))
            with patch.object(SLOScheduler,'_schedule',autospec=True,return_value='native') as call:
                self.assertEqual(s._schedule(),'native');call.assert_called_once_with(s)
            self.assertEqual(self.model.calls,0)

    def test_no_wall_clock_lease_expiry_or_status_trigger_in_mixed_mode(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();s.add_seq_group(self.group(1,0));chain=self.group(2);s.add_seq_group(chain)
            for _ in range(8):self.fixture.step(s)
            calls=self.model.calls;plan=s.last_plan
            # Ten wall seconds later, the unchanged v45 generated-token lease
            # still applies. Status changes did not add a per-tick replan.
            with patch('time.perf_counter',return_value=1010.):
                self.fixture.step(s)
            self.assertEqual(self.model.calls,calls);self.assertEqual(s.last_plan,plan)
            self.assertIn(chain.request_id,s.leases)
            self.assertLess(chain.first_seq.get_output_len()-s.leases[chain.request_id],50)

    def test_zero_probability_mixed_work_drains_without_active_preemption(self):
        self.model.drain=True
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();s.add_seq_group(self.group(1,0));s.add_seq_group(self.group(2))
            for _ in range(20):
                out=self.fixture.step(s)
                self.assertFalse(out.blocks_to_swap_out)
            self.assertTrue(s.policy.draining);self.assertEqual(self.model.calls,1)
            self.assertTrue(all(g.first_seq.get_output_len()>0 for g in s.running))

    def test_mixed_does_not_call_global_native_scheduler(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();s.add_seq_group(self.group(1,0));s.add_seq_group(self.group(2))
            with patch.object(SLOScheduler,'_schedule',side_effect=AssertionError('global bypass')):
                self.fixture.step(s)
            self.assertGreater(self.model.calls,0)

    def test_joint_transition_cancels_only_unexecuted_foreground_bin_plans(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();fg=self.group(1,0);s.add_seq_group(fg)
            self.fixture.step(s)
            prefix=fg.first_seq.get_output_len()
            s._pending_swap_out_queue.append(fg)
            s._current_bin_queue.append(fg)
            s.add_seq_group(self.group(2))
            with patch.object(SLOScheduler,'_schedule',side_effect=AssertionError('global bypass')):
                self.fixture.step(s)
            self.assertFalse(s._pending_swap_out_queue)
            self.assertFalse(s._current_bin_queue)
            self.assertGreaterEqual(fg.first_seq.get_output_len(),prefix)
            self.assertGreater(self.model.calls,0)

    def test_aborting_mixed_work_cleans_both_request_registries(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();s.add_seq_group(self.group(1,0));s.add_seq_group(self.group(2))
            for _ in range(3):self.fixture.step(s)
            s.abort_seq_group(['1','2'])
            self.assertFalse(s.running or s.waiting or s.swapped)
            self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),20)
            self.assertEqual(s.block_manager.get_num_free_cpu_blocks(),32)


if __name__ == '__main__':
    unittest.main()
