"""v44 policy tests, including real native block-manager swap/recompute paths."""
import ast
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace as S
import unittest

from chain_admission import blocks,select
from chain_forecast import StageForecaster

STAGE=Path(__file__).resolve().parents[1]

class PlanningTests(unittest.TestCase):
    def member(self,name):return {'request':name,'prompt':128,'generated':0,'max_tokens':1024,'arrival':1.}
    def goal(self,name,probability,work,deadline):
        return {'chain':name,'stage':0,'probability':probability,'remaining_work':work,'critical_mean_seconds':10.,
                'score':probability/work,'deadline':deadline,'arrival':1.,'members':[self.member(name)]}
    def test_horizon_replaces_full_length_reservation(self):
        m=self.member('x');m['prediction']=1024
        self.assertEqual(blocks(m),12)
        m['prediction']=20
        self.assertEqual(blocks(m),12)
    def test_completion_density_not_raw_deadline(self):
        chosen,_=select([self.goal('risky',.1,1000,80),self.goal('short',1.,100,240)],12,32)
        self.assertEqual(chosen,{'short'})
    def test_no_predicted_opportunity_returns_native_drain(self):
        selected,rows=select([self.goal('x',0.,100,1)],32,32)
        self.assertIsNone(selected);self.assertEqual(rows[0]['probability'],0.)
    def test_survival_uses_only_observed_generated_prefix(self):
        model={'neighbors':2,'prefill_tokens_per_decode_unit':256,'interstage_gap_decode_units':32,
               'forecast_guard_decode_units':50,'service_safety_factor':1.15,
               'records':[{'id':1,'stages':[{'inputs':[128],'outputs':[20]}]},
                          {'id':2,'stages':[{'inputs':[128],'outputs':[100]}]}]}
        f=StageForecaster(model);meta={'stages':1,'widths':[1],'observed':{},'neighbor_cache':{}}
        v=[{'member':0,'declared_input':128,'generated':50,'prefill_tokens':0,'max_tokens':1024}]
        result=f.estimate(meta,0,v,100.,.03)
        self.assertEqual(result['donors'],1);self.assertEqual(result['current_remaining_tokens'],50)
    def test_fit_only_asset_and_no_runtime_future_target_attributes(self):
        model=json.loads((STAGE/'assets/CHAIN_MODEL.json').read_text())
        self.assertEqual(len(model['records']),295)
        self.assertFalse(set(model['fit_ids'])&set(model['cal_ids']))
        self.assertEqual({x['id'] for x in model['records']},set(model['fit_ids']))
        for name in ['native_chain.py','chain_admission.py','chain_forecast.py']:
            tree=ast.parse((STAGE/'programs'/name).read_text())
            self.assertFalse({n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}&{'target_output_length','real_output_len'})

class DummyModel:
    def __init__(self,drain=False):self.calls=0;self.drain=drain
    def estimate(self,meta,stage,views,slack,seconds):
        self.calls+=1
        return {'probability':0. if self.drain else 1.,'score':10. if slack<200 else 1.,
                'remaining_work':100.,'critical_mean_seconds':1.,'current_remaining_tokens':50.,
                'donors':1,'survival_backoff':False}

class NativeRuntime(unittest.TestCase):
    def make_scheduler(self,cpu=32,drain=False):
        from vllm.config import CacheConfig,SchedulerConfig
        import native_chain
        native_chain.MODEL=DummyModel(drain)
        cfg=SchedulerConfig('generate',64,32,512,enable_chunked_prefill=True,policy='jitserve',top_k_selection=3,max_swaps_per_iter=3)
        cache=CacheConfig(16,.9,1,'auto');cache.num_gpu_blocks=20;cache.num_cpu_blocks=cpu
        scheduler=native_chain.NativeChainScheduler(cfg,cache,None)
        return scheduler,native_chain.MODEL
    def group(self,rid,deadline,prompt=160):
        from jitserve.request_info import RequestInfo,RequestType
        from vllm.sequence import Sequence,SequenceGroup
        from vllm import SamplingParams
        info=RequestInfo(RequestType.COLLECTIVE,(2.,.1,20.),1,int(rid),120,prompt,64)
        info.chain_deadline=deadline;info.chain_stage_count=1;info.chain_stage_widths=[1]
        seq=Sequence(int(rid),inputs={'prompt_token_ids':list(range(prompt)),'prompt':'CPU'},block_size=16)
        return SequenceGroup(rid,[seq],time.time(),info,SamplingParams(temperature=0,max_tokens=256))
    def step(self,scheduler):
        from vllm.sequence import Logprob
        _,out,_=scheduler.schedule()
        for item in out.scheduled_seq_groups:
            item.seq_group.update_num_computed_tokens(item.token_chunk_size)
            if not item.seq_group.is_prefill():item.seq_group.first_seq.append_token_id(1,{1:Logprob(0.)})
        return out
    def exercise_preemption(self,cpu):
        from vllm.sequence import SequenceStatus
        scheduler,_=self.make_scheduler(cpu)
        later=self.group('1',time.perf_counter()+1000);scheduler.add_seq_group(later)
        for _ in range(3):self.step(scheduler)
        urgent=self.group('2',time.perf_counter()+100);scheduler.add_seq_group(urgent)
        for _ in range(5):
            out=self.step(scheduler);self.assertFalse(out.blocks_to_swap_out)
        self.assertEqual(later.first_seq.status,SequenceStatus.RUNNING)
        for _ in range(100):
            out=self.step(scheduler)
            if later.first_seq.status!=SequenceStatus.RUNNING:break
        else:self.fail('residence never releases the lower-value incumbent')
        before=later.first_seq.get_output_len();self.assertGreaterEqual(before,50)
        if cpu==1:
            self.assertEqual(later.first_seq.status,SequenceStatus.WAITING);self.assertFalse(out.blocks_to_swap_out)
        else:
            self.assertEqual(later.first_seq.status,SequenceStatus.SWAPPED);self.assertTrue(out.blocks_to_swap_out)
            self.assertFalse(out.blocks_to_swap_in)
        self.assertFalse(out.scheduled_seq_groups)
        scheduler.abort_seq_group('2')
        for _ in range(8):
            resumed=self.step(scheduler)
            if later.first_seq.get_output_len()>before:break
        self.assertEqual(later.first_seq.get_output_len(),before+1)
        scheduler.abort_seq_group('1')
        self.assertEqual(scheduler.block_manager.get_num_free_gpu_blocks(),20)
        self.assertEqual(scheduler.block_manager.get_num_free_cpu_blocks(),cpu)
    def test_residence_then_native_swap_resume(self):self.exercise_preemption(32)
    def test_cpu_full_native_recompute_preserves_output_prefix(self):self.exercise_preemption(1)
    def test_drain_uses_frame_forecast_and_makes_progress(self):
        scheduler,model=self.make_scheduler(drain=True)
        group=self.group('1',time.perf_counter()+1000,prompt=64);scheduler.add_seq_group(group)
        for _ in range(20):self.step(scheduler)
        self.assertTrue(scheduler.policy.draining)
        self.assertEqual(model.calls,1)
        self.assertGreater(group.first_seq.get_output_len(),10)
        scheduler.abort_seq_group('1')
    def test_observation_accepts_actual_candidate_engine_contract(self):
        import tempfile
        from unittest.mock import patch
        from abc_observation import Observation
        scheduler,_=self.make_scheduler()
        class Executor:
            async def execute_model_async(self,*args,**kwargs):return None
        Executor.__module__='vllm.executor.cpu_fixture'
        llm=S(model_executor=Executor(),scheduler=[scheduler],cache_config=scheduler.cache_config,
              scheduler_config=scheduler.scheduler_config,model_config=S(max_model_len=512,dtype='torch.bfloat16',enforce_eager=True))
        with tempfile.TemporaryDirectory()as directory:
            obs=Observation(Path(directory),'C',Path(directory),{'physical_gpu_index':1})
            try:
                with patch('abc_observation.inspect.getfile',return_value=__file__):obs.engine_ready(S(engine=llm))
                self.assertEqual(obs.resources['mechanism_version'],scheduler.policy.version)
                self.assertTrue(obs.resources['native_executor'])
            finally:obs.finalize()

    def test_public_chain_metadata_is_captured_before_stage_budget(self):
        import native_chain
        server=S(_handle_collective_request=lambda info,data,prompt:(True,info.collection_id))
        native_chain.install(server);info=S(collection_id=123,slo_constraint=(2.,.1,40.))
        now=time.perf_counter();server._handle_collective_request(info,{'num_stages':4,'requests_per_stage':[1,3,1,1],'served_time':60},'p')
        self.assertAlmostEqual(info.chain_deadline-now,100,delta=.1)
        self.assertEqual(info.chain_stage_widths,[1,3,1,1])

if __name__=='__main__':unittest.main(verbosity=2)
