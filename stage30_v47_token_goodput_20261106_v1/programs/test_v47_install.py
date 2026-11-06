"""Fresh CPU interpreter install contract; no engine/model allocation."""
import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import unittest


class Installation(unittest.TestCase):
    def test_public_installer_and_deadline_capture_before_native_budget(self):
        from install_v47 import install
        server=S(_handle_collective_request=lambda info,data,prompt:(True,info.collection_id),
                 graph_ctx=S(generation_tokenizer=S(encode=lambda text,add_special_tokens:text.split())))
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        r=install(server,Path(self.directory.name))
        runtime=json.loads((Path(self.directory.name)/'V47_RUNTIME.json').read_text())
        mixed=json.loads((Path(self.directory.name)/'MIXED_RUNTIME.json').read_text())
        self.assertEqual(runtime['version'],mixed['version'])
        self.assertFalse(runtime['installer_runs_model'])
        self.assertNotIn('model_execution',runtime)
        from mixed_joint import MixedScheduler
        import vllm.engine.llm_engine as engine
        self.assertIs(engine.SLOScheduler,MixedScheduler)
        self.assertFalse(r['stage_budget_adapter']);self.assertNotIn('baseline_budget',sys.modules)
        info=S(collection_id=123,slo_constraint=(2.,.1,40.),input_len=900)
        with patch('time.perf_counter',return_value=1000.):
            server._handle_collective_request(info,{'num_stages':4,'requests_per_stage':[1,3,1,1],'served_time':60},'one two')
        self.assertEqual(info.chain_deadline,1100.)
        self.assertEqual(info.chain_reward_input_tokens,2)
        self.assertEqual(info.chain_stage_widths,[1,3,1,1])


if __name__=='__main__':unittest.main(verbosity=2)
