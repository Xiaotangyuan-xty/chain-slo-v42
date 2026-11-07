import asyncio
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from execution_observation import wrap_executor
from step_cost import StepCost


def request(prefill=False,swap=False):
    return NS(virtual_engine=0,seq_group_metadata_list=[NS(is_prompt=prefill,token_chunk_size=512 if prefill else 1,seq_data={1:NS(get_len=lambda:4096)})],blocks_to_swap_in=[(1,2)]if swap else [],blocks_to_swap_out=[],blocks_to_copy=[])


class Execution(unittest.TestCase):
    def test_async_boundary_excludes_scheduler_time_and_keeps_all_flags(self):
        class Executor:
            async def execute_model_async(self,execute_model_req):return ['output']
        e=Executor();s=NS(cost_model=StepCost(),_execution_flags={'planning_seconds':.2,'preempted':2,'ignored':1,'policy_path':'joint'})
        events=[];wrap_executor(e,[s],events.append)
        with patch('execution_observation.time.perf_counter',side_effect=[10.,10.04]):
            result=asyncio.run(e.execute_model_async(request(True,True)))
        self.assertEqual(result,['output']);self.assertAlmostEqual(events[0]['seconds'],.04)
        self.assertEqual(events[0]['context_tokens'],4096);self.assertEqual(events[0]['prefill_tokens'],512)
        self.assertEqual(events[0]['swap_in_blocks'],1);self.assertEqual(events[0]['preempted'],2)
        self.assertEqual(s.cost_model.stats['accepted_samples'],1)
        self.assertEqual(s.cost_model.stats['planned_samples'],1)

    def test_sync_long_call_and_foreground_pure_paths_have_same_coverage(self):
        for path in ['joint','pure','foreground']:
            class Executor:
                def execute_model(self,execute_model_req):return 42
            e=Executor();s=NS(cost_model=StepCost(),_execution_flags={'policy_path':path,'planning_seconds':.1})
            events=[];wrap_executor(e,[s],events.append)
            with patch('execution_observation.time.perf_counter',side_effect=[10.,12.]):self.assertEqual(e.execute_model(execute_model_req=request()),42)
            self.assertEqual(s.cost_model.stats['long_over1s'],1)

    def test_executor_failure_is_preserved_logged_and_not_fit_as_latency(self):
        class Executor:
            def execute_model(self,execute_model_req):raise RuntimeError('fixture')
        e=Executor();s=NS(cost_model=StepCost());events=[];wrap_executor(e,[s],events.append)
        with self.assertRaisesRegex(RuntimeError,'fixture'):e.execute_model(request())
        self.assertEqual(events[0]['error'],'RuntimeError');self.assertFalse(s.cost_model.samples)
        with self.assertRaises(RuntimeError):wrap_executor(e,[s])

    def test_engine_installer_hooks_native_constructor_without_starting_model(self):
        import subprocess,sys
        script=r"""
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile,json,asyncio,atexit
from unittest.mock import patch
import vllm.engine.llm_engine as engine
import vllm.engine.async_llm_engine as async_engine
from execution_observation import install
from step_cost import StepCost
class Executor:
 async def execute_model_async(self,execute_model_req):return 42
 def shutdown(self):pass
class Scheduler:
 def __init__(self):self.cost_model=StepCost();self.last_plan=0
 def schedule(self):
  self.last_plan+=1
  return [],NS(preempted=1,ignored_seq_groups=[]),False
calls=[]
def native_init(self,*args,**kw):calls.append(1);self.model_executor=Executor();self.scheduler=[Scheduler()]
import sys
directory=sys.argv[1]
if True:
 with patch.object(engine.LLMEngine,'__init__',native_init):
  receipt=install(directory);assert not calls
  instance=engine.LLMEngine.__new__(engine.LLMEngine);instance.__init__()
  assert calls==[1]
  instance.scheduler[0].schedule()
  r=NS(virtual_engine=0,seq_group_metadata_list=[NS(is_prompt=False,token_chunk_size=1,seq_data={1:NS(get_len=lambda:128)})],blocks_to_swap_in=[],blocks_to_swap_out=[],blocks_to_copy=[])
  assert asyncio.run(instance.model_executor.execute_model_async(r))==42
  assert receipt['events']==1
  assert instance.scheduler[0].cost_model.stats['preempted_samples']==1
  assert instance.scheduler[0].cost_model.stats['planned_samples']==1
  assert issubclass(async_engine._AsyncLLMEngine,engine.LLMEngine)
  assert receipt['events']==1
import torch
assert not torch.cuda.is_initialized()
"""
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory()as directory:
            result=subprocess.run([sys.executable,'-c',script,directory],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(json.loads((Path(directory)/'EXECUTOR_OBSERVATION.json').read_text())['events'],1)

    def test_native_SequenceGroupMetadata_and_ExecuteModelRequest_ABI(self):
        from vllm.sequence import SequenceData,SequenceGroupMetadata,ExecuteModelRequest
        from execution_observation import features
        data=SequenceData.from_seqs([1,2,3]);data.append_token_id(4,0.)
        meta=SequenceGroupMetadata(request_id='native',is_prompt=False,seq_data={1:data},sampling_params=None,block_tables={1:[0]})
        r=ExecuteModelRequest(seq_group_metadata_list=[meta],virtual_engine=0,blocks_to_swap_in=[],blocks_to_swap_out=[],blocks_to_copy=[])
        f=features(r,NS());self.assertEqual(f['members'],1);self.assertEqual(f['context_tokens'],4)


if __name__=='__main__':unittest.main(verbosity=2)
