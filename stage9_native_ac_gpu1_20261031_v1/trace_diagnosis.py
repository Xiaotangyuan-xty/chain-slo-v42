"""Reduce server-resident stage8/9 traces; never transfer prompts or large traces."""
import hashlib
import json
from pathlib import Path
import statistics
import sys


def analyze(path):
    calls = json.loads((path / 'client_calls.json').read_text())
    audit = json.loads((path / 'common_audit_summary.json').read_text())
    requests = {r['request_uuid']: r for r in audit['requests']}
    timing = {k: {'execute': 0., 'steps': 0, 'first': None,
                  'batch_total': 0, 'gaps': [], 'last': None} for k in requests}
    spans = {phase: {'execute_await_wall': 0., 'calls': 0, 'groups': 0,
                     'swap_in_blocks': 0, 'swap_out_blocks': 0} for phase in ['warmup', 'main']}
    current_phase = 'warmup'
    for line in (path / 'common_events.jsonl').open():
        event = json.loads(line)
        if event['kind'] == 'request_start':
            current_phase = event['phase']
        if event['kind'] != 'model_execute':
            continue
        ids = event['request_ids']
        # C has real swap-only executor calls with no sequence metadata.
        phase = requests[ids[0]]['phase'] if ids else current_phase
        if phase:
            total = spans[phase]
            total['execute_await_wall'] += event['elapsed_seconds']
            total['calls'] += 1
            total['groups'] += len(ids)
            total['swap_in_blocks'] += event['swap_in_blocks']
            total['swap_out_blocks'] += event['swap_out_blocks']
        for rid in ids:
            t = timing[rid]
            begin = event['monotonic'] - event['elapsed_seconds']
            if t['first'] is None:
                t['first'] = begin
            if t['last'] is not None:
                t['gaps'].append(max(0., begin - t['last']))
            t['last'] = event['monotonic']
            t['steps'] += 1
            t['batch_total'] += len(ids)
            t['execute'] += event['elapsed_seconds']
    for total in spans.values():
        total['mean_batch_per_execute'] = total['groups'] / max(1, total['calls'])
    lookup = {(r['phase'], r['collection_id'], r['stage_id'], r['request_id']):
              (r, timing[rid]) for rid, r in requests.items()}
    chains = []
    for c in sorted(calls, key=lambda x: (x['phase'] != 'warmup', x['collection_id'])):
        row = {'phase': c['phase'], 'chain': c['collection_id'],
               'jct': c['client_chain_end_monotonic'] - c['client_chain_start_monotonic'],
               'stages': []}
        for stage in c['stages']:
            records = []
            for item in stage['requests']:
                r, t = lookup[(c['phase'], c['collection_id'], stage['stage_id'], item['request_id'])]
                latency = r['finish_monotonic'] - r['start_monotonic']
                records.append({'member': item['request_id'], 'input_tokens': r['prompt_tokens'],
                                'output_tokens': r['output_tokens'], 'engine_latency': latency,
                                'first_forward_wait': t['first'] - r['start_monotonic'],
                                'execute_await_wall': t['execute'],
                                'outside_execute_await': latency - t['execute'],
                                'largest_forward_gap': max(t['gaps'], default=0.),
                                'mean_batch_when_executed': t['batch_total'] / max(1, t['steps'])})
            row['stages'].append({'stage': stage['stage_id'], 'latency': stage['stage_latency'],
                                  'members': records})
        chains.append(row)
    return {'path': str(path), 'spans': spans, 'chains': chains,
            'event_sha256': hashlib.sha256((path / 'common_events.jsonl').read_bytes()).hexdigest(),
            'client_sha256': hashlib.sha256((path / 'client_calls.json').read_bytes()).hexdigest(),
            'caveat': 'execute_await_wall includes native executor/Python wait; it is not a CUDA timing. '
                      'Per-member execute wall overlaps inside a batch; do not sum it as GPU time.'}


if __name__ == '__main__':
    root = Path('/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1')
    stage = sys.argv[1] if len(sys.argv) > 1 else 'stage8_native_abc_gpu1_20261031_v1'
    arms = sys.argv[2:] or ['FULL_A_GPU1_01', 'FULL_C_GPU1_01']
    print(json.dumps({a: analyze(root / stage / 'attempts' / a) for a in arms}, indent=2))
