"""Actual-text native credit with separate fixed-600s and full-drain accounting."""
from pathlib import Path
import csv
import json
import statistics


def analyze(stage, attempt_name, work, calls, raw_duration):
    from transformers import AutoTokenizer
    from metric_events import retokenized_counts, credited_ledger, window_statistics, native_oracle_per_root
    from jitserve_mixed_goodput import native_function
    stage = Path(stage)
    from protocol_paths import experiment_root
    protocol = json.loads((stage / 'PROTOCOL.json').read_text())
    root=experiment_root(stage,protocol['experiment_root'])
    horizon = protocol['observation_horizon_seconds']
    assert horizon == 600.
    attempt = stage / 'attempts' / attempt_name
    origin = json.loads((attempt / 'MAIN_START.json').read_text())['origin_monotonic']
    tokenizer = AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True,
                                               trust_remote_code=False)
    counts = retokenized_counts(work, calls, tokenizer)
    lifetimes = {}
    for line in (attempt / 'LIFETIME_TIMINGS.jsonl').open():
        row = json.loads(line)
        if row['phase'] == 'main':
            key = (row['collection_id'], row['stage_id'], row['request_id'])
            assert key not in lifetimes
            lifetimes[key] = row['token_append_monotonic']
    ledger, details, attained = credited_ledger(work, calls, lifetimes, counts, origin)
    summary = window_statistics(ledger, raw_duration, horizon=horizon, bin_seconds=60.)
    seconds = window_statistics(ledger, raw_duration, horizon=horizon, bin_seconds=1.)
    summary['fixed_window']=summary.pop('fixed_hour');seconds['fixed_window']=seconds.pop('fixed_hour')
    oracle = native_oracle_per_root(native_function(root), work, calls, tokenizer,
                                    max(horizon, raw_duration))
    full = summary['complete_replay']
    checks = {'native_actual_text_numerator_equal':
                  oracle['credited_weighted_tokens'] == sum(full['credit_by_class']) and
                  oracle['credit_by_class'] == full['credit_by_class'],
              'fixed_window_numerator_conservation': summary['conservation_verified'],
              'warmup_end_to_end_gate_pass':
                  json.loads((attempt / 'WARMUP_NATIVE_TOT_GATE.json').read_text())['gate'] == 'PASS'}
    common = json.loads((attempt / 'common_audit_summary.json').read_text())
    expected_generation = {}
    for phase in ['warmup', 'main']:
        for job in work[phase]:
            for sid, stage_requests in enumerate(job['stages']):
                for request in stage_requests:
                    expected_generation[phase, job['collection_id'], sid, request['request_id']] = {
                        'temperature': 0., 'top_p': 1., 'top_k': -1, 'seed': 42,
                        'ignore_eos': True,
                        'max_tokens': 256 if job['application'] == 'tot_style' else 1024,
                        'n': 1, 'target_output_length': request['target_output_tokens']}
    checks['actual_engine_sampling_matches_frozen_contract'] = all(
        all(row['generation'].get(key) == value for key, value in
            expected_generation[row['phase'], row['collection_id'], row['stage_id'],
                                row['request_id']].items())
        for row in common['requests'])
    jobs = {job['collection_id']: job for job in work['main']}
    per_app = {app: {'roots': 0, 'attained_roots': 0, 'full_credit': 0,
                     'window_credit': 0, 'post_window_credit': 0, 'JCT': []}
               for app in ['latency', 'deadline', 'dr', 'tot_style']}
    for call in calls:
        if call['phase'] != 'main':
            continue
        job = jobs[call['collection_id']]
        app = per_app[job['application']]
        app['roots'] += 1
        jct = call['client_chain_end_monotonic'] - call['client_chain_start_monotonic']
        app['JCT'].append(jct)
        if job['request_type'] == 2:
            app['attained_roots'] += jct <= job['received_slo'][2] * len(job['stages'])
    for event in ledger:
        app = per_app[jobs[event['collection_id']]['application']]
        app['full_credit'] += event['weighted_tokens']
        app['window_credit' if event['time_seconds'] < horizon else 'post_window_credit'] += event['weighted_tokens']
    from offline_compare import percentile
    for app_name, app in per_app.items():
        if app_name in ['latency', 'deadline']:
            app['attained_roots'] = attained.get(0 if app_name == 'latency' else 1, 0)
        values = app.pop('JCT')
        app.update(mean_JCT=statistics.fmean(values) if values else None, p95_JCT=percentile(values, .95),
                   window_goodput=app['window_credit'] / horizon,
                   full_goodput=app['full_credit'] / full['duration_seconds'])
    with (attempt / 'CREDIT_LEDGER.jsonl').open('x') as stream:
        for event in ledger:
            stream.write(json.dumps(event) + '\n')
    for name, obj in [('FIXED_600S_AND_FULL_REPLAY.json', summary),
                      ('FIXED_600S_1S.json', seconds), ('NATIVE_METRIC_ORACLE.json', oracle),
                      ('APPLICATION_CONTRIBUTIONS.json', per_app),
                      ('ACTUAL_TEXT_TOKEN_CREDITS.json', details)]:
        (attempt / name).write_text(json.dumps(obj, indent=2) + '\n')
    with (attempt / 'GOODPUT_1S.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['start_seconds', 'end_seconds', 'goodput_total',
                         'latency', 'deadline', 'compound'])
        for bucket in seconds['bins']:
            writer.writerow([bucket['start_seconds'], bucket['end_seconds'],
                             bucket['goodput_total'], *bucket['goodput_by_class']])
    jcts = [call['client_chain_end_monotonic'] - call['client_chain_start_monotonic']
            for call in calls if call['phase'] == 'main']
    compound_total = sum(job['request_type'] == 2 for job in work['main'])
    assert compound_total == 200
    metrics = {'jitserve_token_goodput': summary['fixed_window']['goodput_total'],
               'fixed_600s_token_goodput': summary['fixed_window']['goodput_total'],
               'fixed_600s_credited_weighted_tokens': sum(summary['fixed_window']['credit_by_class']),
               'complete_replay_token_goodput': full['goodput_total'],
               'jitserve_credited_weighted_tokens': sum(full['credit_by_class']),
               'main_duration': full['duration_seconds'],
               'raw_last_completion_seconds': raw_duration,
               'drain_seconds': full['drain_seconds'],
               'compound_attained': attained.get(2, 0), 'compound_total': compound_total,
               'DR_attained': per_app['dr']['attained_roots'],
               'ToT_attained': per_app['tot_style']['attained_roots'],
               'mean_JCT': statistics.fmean(jcts), 'p95_JCT': percentile(jcts, .95),
               'request_failures': 0}
    return {'checks': checks, 'metrics': metrics, 'summary': summary,
            'applications': per_app, 'details': details}
