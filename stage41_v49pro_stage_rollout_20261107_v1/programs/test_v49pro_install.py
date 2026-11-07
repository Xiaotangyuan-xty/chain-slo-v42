from pathlib import Path
from types import SimpleNamespace as NS
import json,tempfile,unittest
import subprocess,sys
from unittest.mock import patch


class Installer(unittest.TestCase):
    def test_all_plan_events_survive_buffer_flush_at_exit(self):
        script='''from pathlib import Path
from types import SimpleNamespace as NS
import sys,json
from install_v49pro import install
from vllm.config import CacheConfig,SchedulerConfig
server=NS(_handle_collective_request=lambda *a:None)
install(server,sys.argv[1],model_key='llama8')
from mixed_joint import MixedScheduler
cfg=SchedulerConfig('generate',64,32,512,enable_chunked_prefill=True,policy='jitserve',top_k_selection=3,max_swaps_per_iter=3)
cache=CacheConfig(16,.9,1,'auto');cache.num_gpu_blocks=128;cache.num_cpu_blocks=32
s=MixedScheduler(cfg,cache,None)
for i in range(5000):s.decision_sink({'tick':i,'reason':'CPU_logging_fixture'})
import torch
assert not torch.cuda.is_initialized()
'''
        with tempfile.TemporaryDirectory()as directory:
            r=subprocess.run([sys.executable,'-c',script,directory],capture_output=True,text=True,timeout=90)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            rows=[json.loads(line)for line in(Path(directory)/'V49PRO_DECISIONS.jsonl').read_text().splitlines()]
            self.assertEqual([r['tick']for r in rows],list(range(5000)))
            receipt=json.loads((Path(directory)/'V49PRO_RUNTIME.json').read_text())
            self.assertEqual(receipt['logged_plan_events'],5000)
            self.assertEqual(receipt['schedulers'][0]['unstreamed_plan_events_lost'],0)

    def test_actual_installer_metadata_native_ToT_and_domain_registration(self):
        from install_v49pro import install
        server=NS(_handle_collective_request=lambda info,data,prompt:('requests_per_stage'in data,info.collection_id),
                  graph_ctx=NS(generation_tokenizer=NS(encode=lambda text,add_special_tokens:text.split())))
        with tempfile.TemporaryDirectory()as directory:
            with self.assertRaises(ValueError):
                install(server,directory,model_key='llama8',shared_prediction_config={'port':12345,'predictor_protocol':'legacy'})
            self.assertFalse(getattr(server,'_v49pro_candidate_installed',False))
            receipt=install(server,directory,model_key='llama8')
            self.assertFalse(receipt['installer_runs_model']);self.assertFalse(receipt['prediction_comparison_ready'])
            with self.assertRaises(RuntimeError):install(server,directory,model_key='llama8')
            self.assertEqual(json.loads((Path(directory)/'V49PRO_RUNTIME.json').read_text())['version'],receipt['version'])
            from mixed_joint import MixedScheduler,MixedPolicy
            import vllm.engine.llm_engine as engine
            self.assertIs(engine.SLOScheduler,MixedScheduler)
            from jitserve.request_info import RequestInfo,RequestType
            info=RequestInfo(RequestType.COLLECTIVE,(2.,.1,20.),0,123,80,128,32)
            with patch('time.perf_counter',return_value=1000.):
                result=server._handle_collective_request(info,{'num_stages':4,'served_time':5.},'one two')
            self.assertEqual(result,(False,123));self.assertEqual(info.chain_deadline,1075.)
            self.assertEqual(info.chain_stage_widths,[3,1,3,1]);self.assertEqual(info.chain_future_max_tokens,256)
            import controlled_native as reference
            group=reference.NativeRuntime().group('1',1075.)
            group.request_info=info;group.collection_id=123
            policy=MixedPolicy();policy.joint_mode=True;policy.add_seq_group(group)
            self.assertEqual(policy.chains[123]['future_max_tokens'],256)
            goals=policy.goals([group],1000.)
            self.assertEqual(goals[0]['forecast_domain'],'reserved_ToT_calibration')
            self.assertTrue(goals[0]['supported']);self.assertEqual(goals[0]['donors'],8);self.assertTrue(goals[0]['stage_aware_service'])
            self.assertNotIn('baseline_budget',__import__('sys').modules)


if __name__=='__main__':unittest.main(verbosity=2)
