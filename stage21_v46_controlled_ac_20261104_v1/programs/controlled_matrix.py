"""Behavioral qualification of the four cells and low-observation adapter; CPU only."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

STAGE = Path(__file__).resolve().parents[1]


def child(mode):
    import native_chain as native
    from chain_forecast import StageForecaster
    from ablation import apply, VERSIONS, current_stage_estimate
    model = {'neighbors': 2, 'prefill_tokens_per_decode_unit': 256,
             'interstage_gap_decode_units': 32, 'forecast_guard_decode_units': 50,
             'service_safety_factor': 1.15,
             'records': [{'id': 1, 'stages': [{'inputs': [128], 'outputs': [100]},
                                             {'inputs': [1024], 'outputs': [900]}]}]}
    forecast = StageForecaster(model)
    meta = {'stages': 2, 'widths': [1, 1], 'observed': {}, 'neighbor_cache': {}}
    views = [{'member': 0, 'declared_input': 128, 'generated': 20,
              'prefill_tokens': 128, 'max_tokens': 1024}]
    long = forecast.estimate(copy.deepcopy(meta), 0, views, 10, .03)
    short = current_stage_estimate()(forecast, copy.deepcopy(meta), 0, views, 10, .03)
    assert long['probability'] == 0 and short['probability'] == 1
    assert abs(short['remaining_work'] - 80.5) < 1e-10
    assert abs(long['remaining_work'] - (80.5 + 900 + 4)) < 1e-10
    assert short['donors'] == long['donors'] and short['current_remaining_tokens'] == long['current_remaining_tokens']
    native.MODEL = forecast
    original_goals = native.ChainCompletionPolicy.goals
    original_estimate = native.MODEL.estimate.__func__
    applied = apply(mode)
    result = native.MODEL.estimate(copy.deepcopy(meta), 0, views, 10, .03)
    assert result == (short if mode.startswith('stage_') else long)
    assert native.ChainCompletionPolicy.goals is original_goals
    if mode in {'full', 'chain_max'}:
        assert native.MODEL.estimate.__func__ is original_estimate
    member = {'request': 'r', 'prompt': 128, 'generated': 20, 'max_tokens': 1024, 'arrival': 1.}
    assert native.blocks(member) == (72 if mode.endswith('max') else 13)
    assert applied['mechanism_version'] == VERSIONS[mode]
    # At the last stage, local and chain forecasts are exactly the same.
    assert forecast.estimate(copy.deepcopy(meta), 1, views, 100, .03) == current_stage_estimate()(forecast, copy.deepcopy(meta), 1, views, 100, .03)
    print(json.dumps({'mode': mode, 'gate': 'PASS'}))


class MatrixChecks(unittest.TestCase):
    def test_conditions_and_four_behavioral_cells(self):
        import numpy as np
        p = json.loads((STAGE/'PROTOCOL.json').read_text())
        parent = json.loads((STAGE/'PARENT_V45.json').read_text())
        expected = np.cumsum(np.random.RandomState(42).poisson(1000, 200)/1000).tolist()
        for x, y in zip(expected, p['target_arrivals']): self.assertAlmostEqual(x, y, places=9)
        for k in ['generation', 'stage_counts', 'requests_per_stage', 'trace_sha256']:
            self.assertEqual(p[k], parent[k])
        self.assertEqual({d['physical_gpu_index'] for d in p['arm_devices'].values()}, {2})
        self.assertEqual(p['forbidden_gpu_indices'], [0, 1])
        self.assertEqual(p['cpu_affinity']['A'], p['cpu_affinity']['C'])
        self.assertFalse(p['parallel'])
        for mode in ['full', 'stage_short', 'chain_max', 'stage_max']:
            r = subprocess.run([sys.executable, __file__, '--child', mode], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout+r.stderr)

    def test_frozen_core_and_minimal_common_launchers(self):
        core = json.loads((STAGE/'FROZEN_V45_CORE.json').read_text())
        for n, h in core['files'].items(): self.assertEqual(hashlib.sha256((STAGE/n).read_bytes()).hexdigest(), h)
        a = (STAGE/'programs/a_server.py').read_text()
        c = (STAGE/'programs/c_server.py').read_text()
        for text in [a, c]:
            self.assertNotIn('wrap(SLOPolicy, "allocation_control", allocation)', text)
            self.assertNotIn('queues_before =', text)
            self.assertNotIn('"scheduler_step",', text)
        normalized = c.replace('    from low_observation import install as install_low_observation\n    low_receipt = install_low_observation(stage)\n    dump(output_dir / "LOW_OBSERVATION_RUNTIME.json", low_receipt)\n    from native_chain import install as install_chain\n    from ablation import configure\n    configure(stage,output_dir)\n    install_chain(server)\n\n', '')
        normalized = normalized.replace('output_dir, "C", root', 'output_dir, "A", root')
        self.assertEqual(normalized, a)

    def test_low_observation_preserves_swap_recompute_and_planning_state(self):
        import native_chain as native
        from low_observation import stripped_tree
        from controlled_native import NativeRuntime
        source = (STAGE/'programs/native_chain.py').read_text()
        tree = stripped_tree(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NativeChainScheduler')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_schedule')
        namespace = dict(native.__dict__)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), native.__file__, 'exec'), namespace)
        def simulate(cpu, minimal):
            fixture = NativeRuntime()
            scheduler, model = fixture.make_scheduler(cpu)
            if minimal: scheduler._schedule = types.MethodType(namespace['_schedule'], scheduler)
            scheduler.add_seq_group(fixture.group('1', 2000))
            rows = []
            for tick in range(120):
                if tick == 3: scheduler.add_seq_group(fixture.group('2', 1100))
                if tick == 100: scheduler.abort_seq_group('2')
                out = fixture.step(scheduler)
                rows.append(([x.seq_group.request_id for x in out.scheduled_seq_groups],
                             list(out.blocks_to_swap_in), list(out.blocks_to_swap_out),
                             sorted(scheduler.selected or []), sorted(scheduler.last_members or []),
                             sorted(scheduler.last_recorded_selected or []),
                             scheduler.block_manager.get_num_free_gpu_blocks(),
                             scheduler.block_manager.get_num_free_cpu_blocks(),
                             [(g.request_id, g.first_seq.status.name, g.first_seq.get_output_len())
                              for g in list(scheduler.running)+list(scheduler.waiting)+list(scheduler.swapped)]))
            return rows, len(scheduler.reservation_events)
        with patch('time.perf_counter', return_value=1000.), patch('time.time', return_value=100.):
            for cpu in [1, 32]:
                full, logs = simulate(cpu, False)
                minimal, empty = simulate(cpu, True)
                self.assertEqual(full, minimal)
                self.assertGreater(logs, 0)
                self.assertEqual(empty, 0)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child': child(sys.argv[2])
    else: unittest.main(verbosity=2)
