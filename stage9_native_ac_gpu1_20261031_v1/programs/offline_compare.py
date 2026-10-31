"""One offline client-endpoint metric definition for native A and shared B/C."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def endpoint_metrics(calls, stage_counts):
    jct, attained, strict, arrivals = {}, {}, {}, {}
    main = [c for c in calls if c['phase'] == 'main']
    for c in main:
        cid = c['collection_id']
        jct[cid] = c['client_chain_end_monotonic'] - c['client_chain_start_monotonic']
        attained[cid] = jct[cid] <= 20 * (cid % 4 + 1) * stage_counts[str(cid)]
        strict[cid] = jct[cid] <= 20 * stage_counts[str(cid)]
        arrivals[cid] = c['arrival_offset_seconds']
    origin = statistics.median(c['client_chain_start_monotonic'] - c['arrival_offset_seconds'] for c in main)
    duration = max(c['client_chain_end_monotonic'] for c in main) - origin
    good_requests = sum(len(s['requests']) for c in main if attained[c['collection_id']] for s in c['stages'])
    return {'jct': jct, 'official_scaled_attained': sum(attained.values()),
            'official_scaled_per_chain': attained, 'strict_attained': sum(strict.values()),
            'strict_per_chain': strict, 'arrivals': arrivals, 'episode_duration': duration,
            'request_goodput': good_requests / duration,
            'mean_jct': statistics.fmean(jct.values()),
            'jct_p90': percentile(list(jct.values()), .9),
            'jct_p95': percentile(list(jct.values()), .95)}


def analyze_arm(root, stage, name, mode):
    root, stage = Path(root), Path(stage)
    protocol = json.loads((stage / 'PROTOCOL.json').read_text())
    workload = protocol['workloads'][mode]
    p = stage / 'attempts' / name
    calls = json.loads((p / 'client_calls.json').read_text())
    common = json.loads((p / 'common_audit_summary.json').read_text())
    server = json.loads((p / 'server_audit_summary.json').read_text())
    receipt = json.loads((p / 'controller_receipt.json').read_text())
    contract = json.loads((p / 'CLIENT_CONTRACT.json').read_text())
    arm = common['arm']
    checks = {}
    def check(key, okay, actual=None):
        checks[key] = {'pass': bool(okay), 'actual': actual}
    n = workload['main_chain_count']
    main = [c for c in calls if c.get('phase') == 'main']
    warm = [c for c in calls if c.get('phase') == 'warmup']
    counts = protocol['stage_counts']
    widths = protocol['requests_per_stage']
    def complete(c):
        cid = str(c['collection_id'])
        return c.get('success') and len(c['stages']) == counts[cid] and [
            len(s['requests']) for s in c['stages']] == widths[cid] and all(
                s['success'] and all(r['success'] for r in s['requests']) for s in c['stages'])
    check('complete_selected_chains', len(warm) == 1 and warm[0]['collection_id'] == 0
          and complete(warm[0]) and len(main) == n
          and {c['collection_id'] for c in main} == set(range(n)) and all(map(complete, main)))
    flat = {(c['phase'], c['collection_id'], s['stage_id'], r['request_id']): r
            for c in calls for s in c['stages'] for r in s['requests']}
    observed = {(r['phase'], r['collection_id'], r['stage_id'], r['request_id']): r
                for r in common['requests']}
    expected = workload['main_call_count'] + 7
    check('all_request_identities_preserved', len(flat) == len(observed) == expected
          and set(flat) == set(observed), {'client': len(flat), 'engine': len(observed), 'expected': expected})
    bad_outputs, bad_times, bad_input = [], [], []
    signatures = {}
    for key, r in flat.items():
        o = observed.get(key)
        if not o or not o['finished']:
            bad_outputs.append(key)
            continue
        target = min(r['target_output_tokens'], protocol['generation']['max_output_len'])
        if o['output_tokens'] != target or not o['finish_reasons'] or not all(o['finish_reasons']):
            bad_outputs.append((key, o.get('output_tokens'), target, o.get('finish_reasons')))
        if r['input_prompt_sha256'] != o['prompt_sha256'] or r['received_slo'] != o['received_slo']:
            bad_input.append(key)
        if not (r['client_request_start_monotonic'] <= o['start_monotonic'] + .05
                and o['start_monotonic'] <= o['finish_monotonic']
                and o['finish_monotonic'] <= r['client_request_end_monotonic'] + .05
                and abs(r['reported_request_finish_monotonic'] - r['client_request_end_monotonic']) < .05):
            bad_times.append(key)
        signatures[str(key)] = {'input_sha256': o['prompt_sha256'], 'generation': o['generation'],
                                'slo': o['received_slo'], 'output_tokens': o['output_tokens'],
                                'finish_reasons': o['finish_reasons'], 'stop_reasons': o['stop_reasons'],
                                'token_ids_sha256': o['token_ids_sha256']}
    check('actual_engine_outputs_complete', not bad_outputs, bad_outputs)
    check('client_server_input_contract', not bad_input, bad_input)
    check('client_finish_time_recording', not bad_times, bad_times)
    chain_time_errors = [abs(c['total_latency'] - (
        c['client_chain_end_monotonic'] - c['client_chain_start_monotonic'])) for c in calls]
    check('client_chain_jct_recording', max(chain_time_errors) < .05, max(chain_time_errors))
    barrier_errors = []
    for c in calls:
        previous_end = c['client_chain_start_monotonic']
        for s in c['stages']:
            begin = min(r['client_request_start_monotonic'] for r in s['requests'])
            end = max(r['client_request_end_monotonic'] for r in s['requests'])
            if begin < previous_end - .01:
                barrier_errors.append((c['phase'], c['collection_id'], s['stage_id']))
            previous_end = end
    check('multi_stage_barriers_preserved', not barrier_errors, barrier_errors)
    arrivals = {c['collection_id']: c['arrival_offset_seconds'] for c in main}
    errors = [abs(arrivals[k] - protocol['target_arrivals'][k]) for k in range(n)]
    check('frozen_arrival_plan', len(errors) == n and max(errors) <= .5,
          {'actual': arrivals, 'max_error': max(errors)})
    check('public_trace_sha', contract['trace_sha256'] == protocol['trace_sha256'], contract['trace_sha256'])
    events = [json.loads(line) for line in (p / 'common_events.jsonl').read_text().splitlines()]
    idmap = {r['request_uuid']: (r['phase'], r['collection_id'], r['stage_id']) for r in common['requests']}
    coexecution = 0
    for e in events:
        if e['kind'] != 'model_execute':
            continue
        groups = Counter(idmap[rid] for rid in e['request_ids'] if rid in idmap)
        coexecution = max(coexecution, max(groups.values(), default=0))
    check('compound_width_three_executed', coexecution >= 3, coexecution)
    check('dynamic_batch_above_four', common['stats'].get('max_executed_batch', 0) > 4, common['stats'])
    ec = server['event_counts']
    lifecycle = {k: ec.get(k, 0) for k in ['prediction_submit', 'prediction_worker_begin',
                 'prediction_ready', 'sequence_group_assign', 'graph_stage_share', 'graph_completion']}
    check('prediction_graph_lifecycle', all(v == expected for v in lifecycle.values()), lifecycle)
    raw_events = [json.loads(line) for line in (p / 'server_events.jsonl').read_text().splitlines() if line.strip()]
    queues = defaultdict(list)
    completed_predictions = 0
    prediction_errors = []
    for e in raw_events:
        if e['kind'] not in {'prediction_submit', 'prediction_worker_begin', 'prediction_ready'}:
            continue
        key = (e.get('info_object'), e.get('collection_id'), e.get('stage_id'), e.get('request_id'))
        if e['kind'] == 'prediction_submit':
            queues[key].append({'submit': e['n']})
        elif e['kind'] == 'prediction_worker_begin':
            pending = next((q for q in queues[key] if 'begin' not in q), None)
            if pending is None:
                prediction_errors.append(e['n'])
            else:
                pending['begin'] = e['n']
        else:
            pending = next((q for q in queues[key] if 'begin' in q and 'ready' not in q), None)
            if pending is None:
                prediction_errors.append(e['n'])
            else:
                pending['ready'] = e['n']
                completed_predictions += int(pending['submit'] < pending['begin'] < pending['ready'])
    check('ordered_prediction_lifecycles', not prediction_errors and completed_predictions == expected,
          {'complete': completed_predictions, 'errors': prediction_errors})
    check('official_prediction_graph_enabled', server.get('use_prediction') and server.get('use_graph_matching')
          and not server.get('use_default_length') and server.get('base_graphs') == 394)
    rs = common['resources'] or {}
    want = {'gpu_blocks': 512, 'cpu_blocks': 1170, 'block_size': 16, 'max_num_seqs': 32,
            'max_num_batched_tokens': 2048, 'max_model_len': 8192, 'dtype': 'torch.bfloat16',
            'enforce_eager': True, 'single_engine': True, 'schedule_interval': 50,
            'search_strategy': 'sliding_window', 'max_swaps_per_iter': 3,
            'enable_chunked_prefill': True, 'enable_prefix_caching': False}
    check('aligned_resource_caps', all(rs.get(k) == v for k, v in want.items()),
          {'actual': rs, 'expected': want})
    check('final_kv_release', common.get('final_kv_free') == {'gpu_blocks': 512, 'cpu_blocks': 1170},
          common.get('final_kv_free'))
    device = common['device']
    check('only_physical_gpu1', device['physical_gpu_index'] == 1 and device['actual_uuid'] == protocol['gpu_uuid']
          and receipt['final_gpu1']['gpu'].startswith('1, ' + protocol['gpu_uuid'] + ','), device)
    check('controller_and_cleanup', receipt['failure'] is None and receipt['client_returncode'] == 0
          and receipt['gpu_released'] and not receipt['own_gpu_processes_remaining']
          and not receipt['final_gpu1']['processes'], receipt['failure'])
    if arm == 'A':
        check('native_A_unmodified_scheduler', rs.get('native_executor')
              and rs.get('scheduler_type') == 'jitserve.scheduler.slo_scheduler.SLOScheduler'
              and rs.get('policy_type') == 'jitserve.scheduler.policy.SLOPolicy'
              and str(root / 'repo') in rs.get('executor_source', '')
              and common.get('scheduling_changes') is False, rs)
    else:
        check('native_C_backend', rs.get('native_executor')
              and rs.get('executor_type') == 'vllm.executor.gpu_executor.GPUExecutorAsync'
              and rs.get('scheduler_type') == 'native_chain.NativeChainScheduler'
              and rs.get('mechanism_version') == 'v43_native_chain_deadline_reservation', rs)
        check('chain_reservations_executed', common.get('reservation_events', 0) > 0,
              common.get('reservation_events'))
    metrics = endpoint_metrics(calls, counts)
    return {'schema': 'stage8-unified-arm-gate/v1', 'arm': arm, 'attempt': name, 'mode': mode,
            'gate': 'PASS' if all(c['pass'] for c in checks.values()) else 'FAIL',
            'checks': checks, 'metrics': metrics, 'request_signatures': signatures,
            'resources': rs, 'scheduler_stats': server['scheduler_stats'],
            'scope': 'Small frozen public trace; native A unchanged; no superiority/statistical claim'}


def main():
    root, stage = Path(sys.argv[1]), Path(sys.argv[2])
    command = sys.argv[3]
    if command == 'arm':
        name, mode = sys.argv[4:6]
        result = analyze_arm(root, stage, name, mode)
        filename = 'C_COMPATIBILITY_GATE.json' if mode == 'compat' else f'RUN_GATE_{name}.json'
    else:
        names = sys.argv[4:6]
        arms = {arm: analyze_arm(root, stage, name, 'full') for arm, name in zip('AC', names)}
        checks = {f'arm_{arm}': {'pass': x['gate'] == 'PASS'} for arm, x in arms.items()}
        baseline, other = arms['A']['request_signatures'], arms['C']['request_signatures']
        bad = [k for k, v in baseline.items() if k not in other or any(
            v[f] != other[k][f] for f in ['input_sha256', 'generation', 'slo', 'output_tokens',
                                         'finish_reasons', 'stop_reasons'])]
        checks['same_ID_contract_A_C'] = {'pass': not bad and len(other) == len(baseline), 'bad': bad}
        metrics = {arm: x['metrics'] for arm, x in arms.items()}
        result = {'schema': 'stage9-native-AC-comparison/v1',
                  'comparison_valid': all(x['pass'] for x in checks.values()),
                  'checks': checks, 'attempts': dict(zip('AC', names)), 'metrics': metrics,
                  'arms': arms, 'interpretation': 'Same native executor and frozen DEV workload. '
                  'Single exploratory comparison; not a generalization claim.'}
        result['gate'] = 'PASS' if result['comparison_valid'] else 'FAIL'
        result['C_strictly_more_scaled_chains'] = metrics['C']['official_scaled_attained'] > metrics['A']['official_scaled_attained']
        filename = 'AC_COMPARISON.json'
    path = stage / 'receipts' / filename
    with path.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'gate': result['gate'], 'receipt': str(path),
                      'failed_checks': [k for k, v in result['checks'].items() if not v['pass']],
                      'metrics': result['metrics']}, indent=2))
    return 0 if result['gate'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
