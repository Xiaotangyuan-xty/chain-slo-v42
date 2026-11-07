from pathlib import Path
from types import SimpleNamespace as NS
import json,tempfile,unittest
from unittest.mock import patch


class Installer(unittest.TestCase):
    def test_actual_installer_metadata_native_ToT_and_domain_registration(self):
        from install_v48 import install
        server=NS(_handle_collective_request=lambda info,data,prompt:('requests_per_stage'in data,info.collection_id),
                  graph_ctx=NS(generation_tokenizer=NS(encode=lambda text,add_special_tokens:text.split())))
        with tempfile.TemporaryDirectory()as directory:
            receipt=install(server,directory,model_key='llama8')
            self.assertFalse(receipt['installer_runs_model']);self.assertFalse(receipt['prediction_comparison_ready'])
            with self.assertRaises(RuntimeError):install(server,directory,model_key='llama8')
            self.assertEqual(json.loads((Path(directory)/'V48_RUNTIME.json').read_text())['version'],receipt['version'])
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
            self.assertTrue(goals[0]['supported']);self.assertEqual(goals[0]['donors'],14)
            self.assertNotIn('baseline_budget',__import__('sys').modules)


if __name__=='__main__':unittest.main(verbosity=2)
