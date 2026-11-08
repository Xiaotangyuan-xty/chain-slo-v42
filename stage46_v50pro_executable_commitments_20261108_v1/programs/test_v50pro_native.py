"""Actual native scheduler/block-manager fixtures, no model/device execution."""
import unittest
from unittest.mock import patch
from vllm.config import CacheConfig,SchedulerConfig
from vllm.sequence import SequenceStatus
from jitserve.request_info import RequestType
from jitserve.scheduler.policy import SLOPolicy
from jitserve.scheduler.slo_scheduler import SLOScheduler
from chain_forecast import StageForecaster
import controlled_native as reference
import native_chain
from mixed_joint import MixedScheduler

def model():return StageForecaster({'neighbors':1,'prefill_tokens_per_decode_unit':256,'interstage_gap_decode_units':1,'forecast_guard_decode_units':0,'service_safety_factor':1.,'records':[{'id':1,'stages':[{'inputs':[64],'outputs':[64]}]}]})
class Native(unittest.TestCase):
 def setUp(self):
  self.fixture=reference.NativeRuntime();p=patch.object(native_chain,'MODEL',model());p.start();self.addCleanup(p.stop)
 def scheduler(self,cls=MixedScheduler):
  cfg=SchedulerConfig('generate',64,32,512,enable_chunked_prefill=True,policy='jitserve',top_k_selection=3,max_swaps_per_iter=3)
  cache=CacheConfig(16,.9,1,'auto');cache.num_gpu_blocks=128;cache.num_cpu_blocks=32
  return cls(cfg,cache,None)
 def group(self,rid,kind=2):
  g=self.fixture.group(str(rid),1100.,64);g.request_type=RequestType(kind);g.request_info.request_type=RequestType(kind);return g
 def test_foreground_and_mixed_paths_do_not_call_baseline_scores_or_planning(self):
  with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.),patch.object(SLOPolicy,'get_priority',side_effect=AssertionError('baseline score called')),patch.object(SLOScheduler,'_schedule_slo',side_effect=AssertionError('baseline plan called')):
   s=self.scheduler();s.add_seq_group(self.group(1,0))
   for _ in range(8):self.fixture.step(s)
   s.add_seq_group(self.group(2))
   for _ in range(12):self.fixture.step(s)
   self.assertTrue(s._mixed_ever);self.assertGreater(s.mixed_counts['compute_admission_calls'],0)
   s.abort_seq_group(['1','2']);self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128)
 def test_expired_mixed_tasks_complete_without_drops_and_release_pools(self):
  with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
   s=self.scheduler();groups=[self.group(1,0),self.group(2)]
   for g in groups:
    g.arrival_time=0.;s.add_seq_group(g)
   s.policy.chains[2]['deadline']=999.
   for tick in range(500):
    out=self.fixture.step(s)
    for item in out.scheduled_seq_groups:
     if item.seq_group.first_seq.get_output_len()>=20:item.seq_group.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED
    s.free_finished_seq_groups()
    if not(s.running or s.waiting or s.swapped):break
   self.assertLess(tick,499);self.assertTrue(all(g.first_seq.get_output_len()==20 for g in groups));self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),128);self.assertEqual(s.block_manager.get_num_free_cpu_blocks(),32)
 def test_projection_budget_cannot_leave_all_waiters_hidden_on_idle_gpu(self):
  with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.),patch('mixed_joint.plan',return_value=(set(),{'reason':'physical_no_change','fallback_reason':'no_change_projection_budget','backfill':[],'evicted':[],'actions':[]})):
   s=self.scheduler();s.add_seq_group(self.group(1,0));s.add_seq_group(self.group(2))
   for _ in range(20):self.fixture.step(s)
   self.assertTrue(s.selected);self.assertTrue(any(g.first_seq.get_output_len()>0 for g in s.running));s.abort_seq_group(['1','2'])
 def test_pure_DR_path_matches_original_v47(self):
  def trace(cls):
   s=self.scheduler(cls)
   for rid in [1,2,3]:s.add_seq_group(self.group(rid))
   rows=[]
   for _ in range(100):
    out=self.fixture.step(s);rows.append((tuple(x.seq_group.request_id for x in out.scheduled_seq_groups),tuple(out.blocks_to_swap_out),tuple(out.blocks_to_swap_in)))
   s.abort_seq_group(['1','2','3']);return rows
  with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):self.assertEqual(trace(MixedScheduler),trace(native_chain.NativeChainScheduler))
 def test_event_coalescing_still_admits_new_work(self):
  with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
   s=self.scheduler();s.add_seq_group(self.group(1,0))
   for _ in range(4):self.fixture.step(s)
   s.add_seq_group(self.group(2));self.fixture.step(s);self.assertGreater(s.mixed_counts['coalesced_plan_events'],0)
   with patch('time.perf_counter',return_value=1000.3):
    for _ in range(20):self.fixture.step(s)
   self.assertIn('2',set(g.request_id for g in s.running));s.abort_seq_group(['1','2'])
if __name__=='__main__':unittest.main(verbosity=2)
