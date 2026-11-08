"""Actual-text native numerator and fixed-hour settlement ledger for Stage35."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics


def analyze(root, stage, attempt_name, work, calls, raw_duration, *, emit_outputs=True):
    from transformers import AutoTokenizer
    from metric_events import retokenized_counts, credited_ledger, window_statistics, native_oracle_per_root
    from jitserve_mixed_goodput import native_function
    from native_graph_observation import route_checks
    stage = Path(stage)
    from protocol_paths import experiment_root
    root = experiment_root(stage,root)
    protocol = json.loads((stage / 'PROTOCOL.json').read_text())
    attempt = stage / 'attempts' / attempt_name
    horizon=protocol['observation_horizon_seconds']
    origin = json.loads((attempt / 'MAIN_START.json').read_text())['origin_monotonic']
    tokenizer = AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True, trust_remote_code=False)
    counts = retokenized_counts(work, calls, tokenizer)
    lifetimes = {}
    for line in (attempt / 'LIFETIME_TIMINGS.jsonl').open():
        r = json.loads(line)
        if r['phase'] == 'main':
            k = (r['collection_id'], r['stage_id'], r['request_id'])
            assert k not in lifetimes
            lifetimes[k] = r['token_append_monotonic']
    ledger, details, attained = credited_ledger(work, calls, lifetimes, counts, origin)
    summary = window_statistics(ledger, raw_duration, horizon=horizon, bin_seconds=60.)
    seconds = window_statistics(ledger, raw_duration, horizon=horizon, bin_seconds=1.)
    oracle = native_oracle_per_root(native_function(root), work, calls, tokenizer, max(horizon, raw_duration))
    full = summary['complete_replay']
    checks = {'native_actual_text_numerator_equal': oracle['credited_weighted_tokens'] == sum(full['credit_by_class']) and oracle['credit_by_class'] == full['credit_by_class'],
              'fixed_hour_numerator_conservation': summary['conservation_verified'],
              'warmup_end_to_end_gate_pass': json.loads((attempt / 'WARMUP_NATIVE_TOT_GATE.json').read_text())['gate'] == 'PASS'}
    common = json.loads((attempt / 'common_audit_summary.json').read_text())
    expected_generation = {}
    for phase in ['warmup', 'main']:
        for job in work[phase]:
            for sid, stage_requests in enumerate(job['stages']):
                for r in stage_requests:
                    expected_generation[phase, job['collection_id'], sid, r['request_id']] = {
                        'temperature': 0., 'top_p': 1., 'top_k': -1, 'seed': 42,
                        'ignore_eos': True, 'max_tokens': 256 if job['application'] == 'tot_style' else 1024,
                        'n': 1, 'target_output_length': r['target_output_tokens']}
    checks['actual_engine_sampling_matches_frozen_contract'] = all(
        all(row['generation'].get(k) == v for k, v in expected_generation[row['phase'], row['collection_id'], row['stage_id'], row['request_id']].items())
        for row in common['requests'])
    route_gate, bad = route_checks(attempt, work, protocol['candidate_version'], 'main')
    checks.update(route_gate)
    jobs = {j['collection_id']: j for j in work['main']}
    per_app = {app: {'roots': 0, 'attained_roots': 0, 'full_credit': 0, 'hour_credit': 0, 'post_window_credit': 0, 'JCT': []} for app in ['latency', 'deadline', 'dr', 'tot_style']}
    for c in calls:
        if c['phase'] != 'main':
            continue
        j = jobs[c['collection_id']]
        a = per_app[j['application']]
        a['roots'] += 1
        jct = c['client_chain_end_monotonic'] - c['client_chain_start_monotonic']
        a['JCT'].append(jct)
        if j['request_type'] == 2:
            a['attained_roots'] += jct <= j['received_slo'][2] * len(j['stages'])
    for event in ledger:
        a = per_app[jobs[event['collection_id']]['application']]
        a['full_credit'] += event['weighted_tokens']
        a['hour_credit' if event['time_seconds'] < 3600 else 'post_window_credit'] += event['weighted_tokens']
    from offline_compare import percentile
    for app, a in per_app.items():
        if app in ['latency', 'deadline']:
            a['attained_roots'] = attained.get(0 if app == 'latency' else 1, 0)
        vals = a.pop('JCT')
        a.update(mean_JCT=statistics.fmean(vals), p95_JCT=percentile(vals, .95), hour_goodput=a['hour_credit'] / 3600., full_goodput=a['full_credit'] / full['duration_seconds'])
    if emit_outputs:
        with (attempt / 'CREDIT_LEDGER.jsonl').open('x') as stream:
            for e in ledger:
                stream.write(json.dumps(e) + '\n')
        for name, obj in [('FIXED_HOUR_AND_FULL_REPLAY.json', summary), ('FIXED_HOUR_1S.json', seconds),
                          ('NATIVE_METRIC_ORACLE.json', oracle), ('APPLICATION_CONTRIBUTIONS.json', per_app),
                          ('ACTUAL_TEXT_TOKEN_CREDITS.json', details)]:
            (attempt / name).write_text(json.dumps(obj, indent=2) + '\n')
        with (attempt / 'GOODPUT_1S.csv').open('w') as stream:
            writer = csv.writer(stream); writer.writerow(['start_seconds', 'end_seconds', 'goodput_total', 'latency', 'deadline', 'compound'])
            for b in seconds['bins']:
                writer.writerow([b['start_seconds'], b['end_seconds'], b['goodput_total'], *b['goodput_by_class']])
    jcts = [c['client_chain_end_monotonic'] - c['client_chain_start_monotonic'] for c in calls if c['phase'] == 'main']
    metrics = {'jitserve_token_goodput': summary['fixed_hour']['goodput_total'],
               'fixed_hour_token_goodput': summary['fixed_hour']['goodput_total'],
               'fixed_hour_credited_weighted_tokens': sum(summary['fixed_hour']['credit_by_class']),
               'complete_replay_token_goodput': full['goodput_total'],
               'jitserve_credited_weighted_tokens': sum(full['credit_by_class']),
               'main_duration': full['duration_seconds'], 'raw_last_completion_seconds': raw_duration,
               'drain_seconds': full['drain_seconds'], 'compound_attained': attained.get(2, 0),
               'compound_total': sum(j['request_type']==2 for j in work['main']), 'DR_attained': per_app['dr']['attained_roots'],
               'ToT_attained': per_app['tot_style']['attained_roots'],
               'mean_JCT': statistics.fmean(jcts), 'p95_JCT': percentile(jcts, .95), 'request_failures': 0}
    return {'checks': checks, 'metrics': metrics, 'summary': summary, 'applications': per_app, 'route_errors': bad}
