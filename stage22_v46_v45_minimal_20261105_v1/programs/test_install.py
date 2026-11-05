"""Exercise the public installation path in a fresh CPU-only interpreter."""
import ast
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class Installation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from install_v46 import install
        cls.server=SimpleNamespace(_handle_collective_request=lambda *args:None)
        cls.receipt=install(cls.server)

    def test_default_entry_uses_existing_minimal_observation_and_new_adapter(self):
        import native_chain
        import vllm.engine.llm_engine as engine
        from mixed_joint import MixedScheduler
        self.assertIs(engine.SLOScheduler,MixedScheduler)
        self.assertEqual(self.receipt['minimal_observation']['removed_statements'],
                         ['reservation_events.append(event_snapshot)'])
        self.assertNotIn('baseline_budget',sys.modules)
        self.assertFalse(self.receipt['wall_clock_lease'])
        self.assertFalse(self.receipt['event_driven_replanning'])

    def test_whole_chain_deadline_wrapper_retains_v45_budget_semantics(self):
        info=SimpleNamespace(collection_id=123,slo_constraint=(2.,.1,40.))
        with patch('time.perf_counter',return_value=1000.):
            self.server._handle_collective_request(info,{'num_stages':4,
                'requests_per_stage':[1,3,1,1],'served_time':60},'p')
        self.assertEqual(info.chain_deadline,1100.)
        self.assertEqual(info.chain_stage_widths,[1,3,1,1])


if __name__=='__main__':
    unittest.main()
