import ast
import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import unittest

from gpu1_platform import adapt_tree
from abc_observation import install_observation
from offline_compare import endpoint_metrics


HERE = Path(__file__).resolve().parent


class QualificationTests(unittest.TestCase):
    def test_device_adapter_changes_only_identity_functions(self):
        source = """import os
def device_id_to_physical_device_id(device_id):
    return int(os.environ['CUDA_VISIBLE_DEVICES'])
def warn_if_different_devices():
    raise Exception('global NVML enumeration')
class Scheduler:
    def choose(self, jobs):
        return min(jobs)
"""
        tree, changed = adapt_tree(source, 1, '1')
        self.assertEqual(set(changed), {'device_id_to_physical_device_id', 'warn_if_different_devices'})
        old = ast.parse(source)
        self.assertEqual(ast.dump(old.body[-1]), ast.dump(tree.body[-1]))

    def test_all_launches_are_gpu1_and_explicit_same_resources(self):
        tree = ast.parse((HERE / 'controller.py').read_text())
        text = (HERE / 'controller.py').read_text()
        self.assertIn('"nvidia-smi", "-i", "1"', text)
        self.assertNotIn('"nvidia-smi", "-i", "0"', text)
        self.assertIn('env["STUDY_GPU"] = "1"', text)
        self.assertIn('str(root), str(stage), str(attempt)', text)
        for flag in ['--dtype', '--max-model-len', '--max-num-seqs', '--max-num-batched-tokens',
                     '--num-gpu-blocks-override', '--swap-space', '--enforce-eager']:
            self.assertIn(flag, text)

    def test_native_launcher_does_not_install_chain_policy(self):
        tree = ast.parse((HERE / 'a_server.py').read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.append(node.module or '')
        self.assertNotIn('v2_slo_scheduler', imported)
        self.assertNotIn('harness_executor', imported)
        self.assertIn('jitserve.scheduler.policy', imported)
        self.assertIn('shared_backend=False', (HERE / 'a_server.py').read_text())

    def test_observation_preserves_generated_output_objects(self):
        class Engine:
            async def generate(self, prompt, request_info, sampling_params, request_id, *args, **kwargs):
                for output in outputs:
                    yield output
        tokens = [1, 2, 3]
        outputs = [S(finished=False), S(finished=True, prompt_token_ids=[4, 5],
                   outputs=[S(token_ids=tokens, finish_reason='stop', stop_reason=0)])]
        info = S(collection_id=0, stage_id=0, request_id=0, slo_constraint=(2., .1, 20.))
        params = S(seed=42, temperature=0., target_output_length=3)
        with tempfile.TemporaryDirectory() as d:
            obs = install_observation(S(), Engine, Path(d), 'A', Path(d), {'physical_gpu_index': 1})
            async def collect():
                return [x async for x in Engine().generate('prompt', info, params, 'uuid')]
            actual = asyncio.run(collect())
            self.assertIs(actual[0], outputs[0])
            self.assertIs(actual[1], outputs[1])
            self.assertEqual(tokens, [1, 2, 3])
            obs.finalize()
            row = json.loads((Path(d) / 'common_audit_summary.json').read_text())['requests'][0]
            self.assertEqual(row['output_tokens'], 3)
            self.assertEqual(row['finish_reasons'], ['stop'])

    def test_endpoint_metrics_use_actual_stage_count_and_scaled_deadline(self):
        calls = [dict(phase='main', collection_id=2, client_chain_start_monotonic=100.,
                      client_chain_end_monotonic=339., arrival_offset_seconds=3.,
                      stages=[{'requests': [1]}] * 4)]
        result = endpoint_metrics(calls, {'2': 4})
        self.assertEqual(result['official_scaled_attained'], 1)
        self.assertEqual(result['strict_attained'], 0)
        calls[0]['client_chain_end_monotonic'] = 341.
        self.assertEqual(endpoint_metrics(calls, {'2': 4})['official_scaled_attained'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
