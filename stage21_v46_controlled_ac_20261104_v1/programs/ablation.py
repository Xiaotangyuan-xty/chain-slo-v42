"""Factorial interventions: value horizon x current-stage KV envelope."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import types

VERSIONS = {'full': 'v45_native_chain_stable_admission',
            'stage_short': 'v45_factorial_stage_short',
            'chain_max': 'v45_factorial_chain_max',
            'stage_max': 'v45_factorial_stage_max'}


def current_stage_estimate():
    """Use identical donors/survival/calibration; omit only future-stage work/time."""
    import chain_forecast as forecast
    source = Path(forecast.__file__)
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StageForecaster')
    function = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'estimate'))
    removed = 0
    for part in ast.walk(function):
        if not isinstance(part, ast.For):
            continue
        keep = []
        for statement in part.body:
            if isinstance(statement, ast.For) and isinstance(statement.target, ast.Name) and statement.target.id == 'sid':
                removed += 1
            else:
                keep.append(statement)
        part.body = keep
    if removed != 1:
        raise RuntimeError('Unexpected forecast future-stage loop')
    namespace = dict(forecast.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(source), 'exec'), namespace)
    return namespace['estimate']


def apply(mode):
    if mode not in VERSIONS:
        raise ValueError('Unknown factorial cell')
    import native_chain as native
    import chain_admission as admission
    if native.ChainCompletionPolicy.version != VERSIONS['full']:
        raise RuntimeError('Intervention already applied')
    changes = []
    if mode in {'stage_short', 'stage_max'}:
        native.MODEL.estimate = types.MethodType(current_stage_estimate(), native.MODEL)
        changes.append('q_and_mean_work_use_only_current_ready_stage; same_chain_slack_and_donor_selection')
    if mode in {'chain_max', 'stage_max'}:
        def maximum_span(member, block_size=16, horizon=50):
            return math.ceil((member['prompt'] + member['max_tokens']) / block_size)
        admission.blocks = maximum_span
        native.blocks = maximum_span
        changes.append('current_ready_requests_reserve_published_max_tokens; no_future_stage_reservation')
    native.ChainCompletionPolicy.version = VERSIONS[mode]
    native.ChainCompletionPolicy.ablation_mode = mode
    return {'mode': mode, 'mechanism_version': VERSIONS[mode], 'interventions': changes,
            'future_evaluation_lengths_accessed': False,
            'value_horizon': 'current_stage' if mode.startswith('stage_') else 'remaining_chain',
            'kv_horizon': 'published_max_tokens' if mode.endswith('max') else 'generated_prefix_plus_50'}


def configure(stage, output):
    stage, output = Path(stage), Path(output)
    p = json.loads((stage / 'PROTOCOL.json').read_text())
    core = json.loads((stage / 'FROZEN_V45_CORE.json').read_text())
    bad = [n for n, h in core['files'].items() if hashlib.sha256((stage/n).read_bytes()).hexdigest() != h]
    if bad:
        raise RuntimeError(f'Frozen core changed: {bad}')
    runtime = apply(p['ablation'])
    if runtime['mechanism_version'] != p['C_version']:
        raise RuntimeError('Protocol/version mismatch')
    runtime.update(core_sha256=core['files'], core_bytes_unchanged=True)
    (output/'ABLATION_RUNTIME.json').write_text(json.dumps(runtime, indent=2)+'\n')
    return runtime
