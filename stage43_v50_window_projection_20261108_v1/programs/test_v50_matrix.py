from pathlib import Path
import ast,copy,inspect,sys,unittest
H=Path(__file__).resolve().parents[1]/'harness';sys.path.insert(0,str(H))
from study_contract import case_specs,validate_matrix,METHODS,VERSION
from planner_config import CONFIG
from window_control import plan


class Matrix(unittest.TestCase):
    def test_six_cases_unique_ports_identities_and_exact_order(self):
        cases=[]
        for gpu in [0,1]:
            cs=case_specs('/experiment/study',gpu)
            self.assertEqual([c['version']for c in cs],['v50','A','Astar'])
            self.assertEqual([c['arm']for c in cs],['C','A','A'])
            self.assertTrue(validate_matrix({'GPU':gpu,'cases':cs},gpu));cases+=cs
        self.assertEqual(len({c['stage']for c in cases}),6)
        self.assertEqual(len({c['attempt']for c in cases}),6)
        self.assertEqual(len({c['http_port']for c in cases}|{c['prediction_port']for c in cases}),12)

    def test_invalid_device_resource_order_and_duplicates_fail_before_launch(self):
        with self.assertRaises(ValueError):case_specs('/study',2)
        cs=case_specs('/study',0)
        for mutate in ['GPU','order','duplicate','KV']:
            m={'GPU':0,'cases':copy.deepcopy(cs)}
            if mutate=='GPU':m['cases'][1]['GPU']=1
            elif mutate=='order':m['cases'][1]['version']='Astar'
            elif mutate=='duplicate':m['cases'][1]['attempt']=m['cases'][0]['attempt']
            else:m['cases'][0]['gpu_blocks']=9999
            with self.assertRaises(ValueError):validate_matrix(m,0)

    def test_default_planner_contract_has_one_immutable_source(self):
        signature=inspect.signature(plan)
        self.assertEqual(signature.parameters['max_seconds'].default,CONFIG.max_seconds)
        self.assertEqual(signature.parameters['max_evicted_members'].default,CONFIG.max_evicted_members)
        self.assertEqual(signature.parameters['scenario_limit'].default,CONFIG.scenario_limit)
        with self.assertRaises(Exception):CONFIG.max_seconds=.2

    def test_CPU_preview_cannot_be_launched_by_any_entry(self):
        for name in ['runner.py','start.py']:self.assertIn('CPU_PREVIEW_ONLY.json',(H/name).read_text())
        self.assertIn("protocol.get('case_kind') != 'formal'",(H/'controller.py').read_text())

    def test_baselines_keep_native_scheduler_and_final_gate_explicit_root(self):
        import install_comparison_runtime
        src=inspect.getsource(install_comparison_runtime.install)
        self.assertIn("method=='v50'",src);self.assertIn("method not in ['A','Astar']",src)
        gate=(H/'offline_compare.py').read_text()
        self.assertIn('aligned_analyze(root,stage,attempt_name,work,calls,raw_duration)',gate)
        self.assertNotIn('parents[2]',gate)
        self.assertIn("if receipt['arm']=='C':",gate)


if __name__=='__main__':unittest.main(verbosity=2)
