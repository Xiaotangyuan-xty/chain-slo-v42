import json
from pathlib import Path
from types import SimpleNamespace as S
import tempfile
import unittest
from prediction_transport import PROTOCOL


class InstallTests(unittest.TestCase):
    def test_actual_native_install_annotations_observer_and_fresh_guard(self):
        from install_v47pro import install
        import vllm.engine.llm_engine as engine
        from mixed_joint import MixedScheduler
        with tempfile.TemporaryDirectory() as directory:
            server=S(_handle_collective_request=lambda info,data,prompt:(False,info.collection_id))
            receipt=install(server,Path(directory),shared_prediction_config={'port':65040,'predictor_protocol':PROTOCOL})
            self.assertIs(engine.SLOScheduler,MixedScheduler)
            self.assertFalse(receipt['installer_runs_model']);self.assertFalse(receipt['installer_allocates_GPU'])
            self.assertTrue(receipt['prediction_comparison_ready']);self.assertFalse(receipt['stage_budget_adapter'])
            info=S(collection_id=1,slo_constraint=(2.,.1,20.))
            server._handle_collective_request(info,{'num_stages':4},'x')
            self.assertEqual(info.chain_stage_widths,[3,1,3,1]);self.assertEqual(info.chain_future_max_tokens,256)
            self.assertTrue((Path(directory)/'V47PRO_RUNTIME.json').is_file())
            with self.assertRaises((RuntimeError,ValueError)):install(server,Path(directory))


if __name__=='__main__':unittest.main(verbosity=2)
