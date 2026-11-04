from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import json,unittest
from forecast_v46 import StageForecaster
from admission_v46 import choose
from observation_v46 import ExecutionCadence
from baseline_budget import cumulative_ratio

def donor(depth=1,width=1,out=100,identifier=1):return {'id':identifier,'stages':[{'inputs':[256]*width,'outputs':[out]*width}for _ in range(depth)]}
def model(records):return StageForecaster({'records':records,'neighbors':32,'prefill_tokens_per_decode_unit':256,'interstage_gap_decode_units':0,'forecast_guard_decode_units':0,'service_safety_factor':1})
def meta(depth=1,width=1):return {'stages':depth,'widths':[width]*depth,'observed':{},'neighbor_cache':{},'future_max_tokens':1024}
def view(member=0,generated=0,cap=1024):return {'request':str(member),'member':member,'declared_input':256,'prompt':256,'generated':generated,'max_tokens':cap,'prefill_tokens':256,'arrival':100.}

class ForecastContract(unittest.TestCase):
    def test_missing_current_members_are_charged_but_do_not_become_real_requests(self):
        f=model([donor(width=3)]);x=f.estimate(meta(width=3),0,[view()],1000,.01)
        self.assertEqual(x['estimated_unsubmitted_members'],2);self.assertEqual(x['remaining_work'],303)
    def test_completed_members_not_charged_again(self):
        m=meta(width=3);m['observed']={0:{1:{'declared_input':256,'output':100}}}
        x=model([donor(width=3)]).estimate(m,0,[view()],1000,.01)
        self.assertEqual(x['estimated_unsubmitted_members'],1);self.assertEqual(x['remaining_work'],202)
    def test_survival_backoff_respects_public_remaining_cap(self):
        v=view(generated=1020,cap=1024);v['prefill_tokens']=0
        x=model([donor(out=100)]).estimate(meta(),0,[v],1.,.1)
        self.assertTrue(x['survival_backoff']);self.assertEqual(x['remaining_work'],4)
        self.assertEqual(x['probability'],1.)
    def test_unsupported_tail_not_omitted_or_raised(self):
        x=model([donor(4),donor(6)]).estimate(meta(8),0,[view()],1000,.01)
        self.assertFalse(x['supported']);self.assertIsNone(x['remaining_work']);self.assertEqual(x['probability'],0)
    def test_cross_depth_donor_must_cover_every_declared_stage(self):
        x=model([donor(6)]).estimate(meta(5),0,[view()],1000,.01)
        self.assertTrue(x['supported']);self.assertEqual(x['remaining_work'],505)
    def test_member_caps_not_replaced_by_other_members_minimum(self):
        vs=[view(0,cap=10),view(1,cap=1000)];x=model([donor(width=2,out=200)]).estimate(meta(width=2),0,vs,1000,.01)
        self.assertEqual(x['remaining_work'],212)
    def test_capacity_waves_increase_duration_without_changing_work(self):
        f=model([donor(width=4)]);vs=[view(i)for i in range(4)]
        a=f.estimate(meta(width=4),0,vs,1000,.01,capacity=1000,max_seqs=32)
        b=f.estimate(meta(width=4),0,vs,1000,.01,capacity=20,max_seqs=1)
        self.assertEqual(a['remaining_work'],b['remaining_work']);self.assertGreater(b['critical_mean_seconds'],a['critical_mean_seconds'])
    def test_current_completed_output_refreshes_donor_retrieval(self):
        m=meta(width=2);f=model([donor(width=2,out=100),donor(width=2,out=1000,identifier=2)])
        f.neighbors(m,0,[view()]);old=deepcopy(m['neighbor_cache'][0][0]);m['observed']={0:{1:{'declared_input':256,'output':1000}}}
        f.neighbors(m,0,[view()]);self.assertNotEqual(old,m['neighbor_cache'][0][0])

class AdmissionContract(unittest.TestCase):
    def goal(self,rid,score=1):return {'chain':int(rid),'probability':1.,'score':score,'deadline':1000.,'arrival':100.,'remaining_work':100.,'members':[view(int(rid))]}
    def test_oversized_first_member_not_unconditionally_accepted(self):
        selected,used,_=choose([self.goal('0')],2,32);self.assertEqual(selected,set());self.assertEqual(used,0)
    def test_protected_calls_charged_and_trimmed_when_capacity_is_insufficient(self):
        a=self.goal('0',2);b=self.goal('1',1);selected,used,_=choose([a,b],20,32,protected={'0','1'})
        self.assertLessEqual(used,20);self.assertEqual(len(selected),1)
    def test_sequence_limit_zero_admits_nothing(self):self.assertEqual(choose([self.goal('0')],100,0)[0],set())
    def test_transition_only_execution_does_not_distort_decode_cadence(self):
        c=ExecutionCadence();c.observe(.05,[S(is_prompt=False)],0)
        for _ in range(100):c.observe(.001,[],10)
        self.assertEqual(c.token_seconds,.05);self.assertEqual(c.decode_batches,1);self.assertEqual(c.transition_batches,100)

class DecisionAuditContract(unittest.TestCase):
    def test_borrowed_foreground_slot_is_not_mistaken_for_a_chain(self):
        from validate_v46 import joint_selected
        kinds={'f1':0,'f2':1,'chain':2}
        self.assertFalse(joint_selected(['f1','f2'],kinds))
        self.assertTrue(joint_selected(['f1','chain'],kinds))
        self.assertFalse(joint_selected(['chain'],kinds))

class BudgetContract(unittest.TestCase):
    def ctx(self,times=None):return S(collection_deepresearch_unfinished_dict={1:S(convert_to_unfinished_graph=lambda *a:None)},_encode_prompt_len=len,dynamic_clustering=S(find_best_match=lambda _:S(times=times)if times else None))
    def test_first_stage_has_positive_budget_last_stage_gets_full_budget(self):
        self.assertEqual(cumulative_ratio(self.ctx(),4,[1]*4,'p',1,0),.25)
        self.assertEqual(cumulative_ratio(self.ctx(),4,[1]*4,'p',1,3),1.)
    def test_histories_only_with_equal_depth_and_nonnegative_times(self):
        self.assertEqual(cumulative_ratio(self.ctx([100,1,1,1]),4,[1]*4,'p',1,0),100/103)
        self.assertEqual(cumulative_ratio(self.ctx([100,1,1,1,1,1]),4,[1]*4,'p',1,0),.25)

class RunnerPathsContract(unittest.TestCase):
    def test_matrix_and_pilot_resolve_same_study_root(self):
        from case_runner import study_root
        root=Path('/study/v46')
        self.assertEqual(study_root(root/'cases/pure_b1'),root)
        self.assertEqual(study_root(root/'pilot_gpu2'),root)

class NativeIntegration(unittest.TestCase):
    def scheduler(self):
        from vllm.config import CacheConfig,SchedulerConfig
        import scheduler_v46
        cfg=SchedulerConfig('generate',64,32,512,enable_chunked_prefill=True,policy='jitserve',penalty_factor=100,top_k_selection=3,max_swaps_per_iter=3)
        cache=CacheConfig(16,.9,1,'auto');cache.num_gpu_blocks=20;cache.num_cpu_blocks=32
        return scheduler_v46.WorkflowScheduler(cfg,cache,None)
    def group(self,rid,kind=2,prompt=64,deadline=2000):
        from jitserve.request_info import RequestInfo,RequestType
        from vllm.sequence import Sequence,SequenceGroup
        from vllm import SamplingParams
        info=RequestInfo(RequestType(kind),(2.,.1,20.),0,int(rid),120,prompt,64)
        if kind==2:info.chain_deadline=deadline;info.chain_stage_count=1;info.chain_stage_widths=[1]
        seq=Sequence(int(rid),inputs={'prompt_token_ids':list(range(prompt)),'prompt':'CPU'},block_size=16)
        return SequenceGroup(str(rid),[seq],100.,info,SamplingParams(temperature=0,max_tokens=256))
    def complete_outputs(self,scheduler,out):
        from vllm.sequence import SequenceStatus
        for item in out.scheduled_seq_groups:
            if item.seq_group.first_seq.get_output_len()>=64:item.seq_group.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED
        scheduler.free_finished_seq_groups()
    def test_native_joint_execution_serves_both_and_fully_releases_kv(self):
        from controlled_native import NativeRuntime
        fixture=NativeRuntime();seen=set()
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.),patch('scheduler_v46.MODEL',model([donor()])):
            scheduler=self.scheduler()
            for i,t in enumerate([0,1,2,2],1):scheduler.add_seq_group(self.group(i,t))
            for tick in range(1000):
                out=fixture.step(scheduler)
                self.complete_outputs(scheduler,out)
                seen.update(x.seq_group.request_type.value for x in out.scheduled_seq_groups)
                if not(scheduler.running or scheduler.waiting or scheduler.swapped):break
            self.assertEqual(seen,{0,1,2});self.assertLess(tick,999)
            self.assertEqual(scheduler.block_manager.get_num_free_gpu_blocks(),20);self.assertEqual(scheduler.block_manager.get_num_free_cpu_blocks(),32)
    def test_prefill_protection_expires_even_without_output_progress(self):
        from vllm.sequence import SequenceStatus
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();g=self.group(1);s.add_seq_group(g);s.block_manager.allocate(g);s.waiting.remove(g);s.running.append(g);g.first_seq.status=SequenceStatus.RUNNING
            self.assertIn(g.request_id,s.protected([g],1000.));self.assertNotIn(g.request_id,s.protected([g],1011.))
    def test_unsupported_chain_does_not_stop_other_live_work(self):
        from controlled_native import NativeRuntime
        fixture=NativeRuntime()
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            s=self.scheduler();s.add_seq_group(self.group(1));s.add_seq_group(self.group(2,0))
            for tick in range(500):
                out=fixture.step(s);self.complete_outputs(s,out)
                if not(s.running or s.waiting or s.swapped):break
            self.assertLess(tick,499)
    def test_corrected_stage_budget_adapter_does_not_change_whole_workflow_deadline(self):
        import scheduler_v46
        server=S(_handle_collective_request=lambda *a:None);scheduler_v46.install(server)
        info=S(stage_id=0,request_id=0,collection_id=1,slo_constraint=(2.,.1,40.))
        with patch('time.perf_counter',return_value=1000.):server._handle_collective_request(info,{'num_stages':4,'requests_per_stage':[1]*4,'served_time':60},'p')
        self.assertEqual(info.chain_deadline,1100.)

if __name__=='__main__':unittest.main(verbosity=2)
