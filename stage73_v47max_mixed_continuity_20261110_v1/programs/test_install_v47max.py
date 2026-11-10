"""Fresh-process installation actually selects max while retaining pure dispatch."""
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile,unittest


class MaxInstallTests(unittest.TestCase):
    def test_actual_installer_and_native_pure_dispatch(self):
        from install_v47max import install
        from mixed_max import MixedScheduler,VERSION
        from mixed_joint import MixedScheduler as Original
        from test_pure_core_restore import PureCoreTraceTests
        from vllm.config import CacheConfig
        from prediction_transport import PROTOCOL
        import vllm.engine.llm_engine as engine
        with tempfile.TemporaryDirectory() as directory:
            server=NS(_handle_collective_request=lambda info,data,prompt:(False,info.collection_id))
            result=install(server,Path(directory),shared_prediction_config={'port':23543,'predictor_protocol':PROTOCOL})
            self.assertIs(engine.SLOScheduler,MixedScheduler)
            self.assertEqual(result['version'],VERSION)
            self.assertFalse(result['installer_runs_model'])
            self.assertFalse(result['installer_allocates_GPU'])
            self.assertIs(MixedScheduler._preempt,Original._preempt)
            f=PureCoreTraceTests();f.setUp()
            try:
                template=f.scheduler()
                s=MixedScheduler(f.shared,template.cache_config,None)
                groups={};f.add_stage(s,groups,1,[1],0)
                f.step(s)
                self.assertTrue(s._pure_core_active)
                self.assertEqual(s.max_counts['mixed_plans'],0)
                f.cleanup(s)
            finally:f.doCleanups()
        import torch
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main(verbosity=2)
