"""Finite pure200 native credit plus an explicitly separate 500s settlement."""
import json
from pathlib import Path
import statistics


def expected_engine_sampling(generation, target_output_tokens):
    return {
        'temperature': generation['temperature'], 'top_p': generation['top_p'],
        'top_k': -1 if generation['temperature'] == 0. else generation['top_k'],
        'seed': generation['seed'], 'ignore_eos': True, 'max_tokens': 1024, 'n': 1,
        'target_output_length': target_output_tokens}


def analyze(root, stage, attempt_name, work, calls, raw_duration):
    from transformers import AutoTokenizer
    from metric_events import retokenized_counts, credited_ledger, window_statistics
    from jitserve_mixed_goodput import compute
    from offline_compare import percentile
    stage = Path(stage)
    attempt = stage / 'attempts' / attempt_name
    p = json.loads((stage / 'PROTOCOL.json').read_text())
    assert all(j['request_type'] == 2 and j.get('application', 'dr') == 'dr' for ph in ['warmup', 'main'] for j in work[ph])
    assert len(work['main']) == 200 and raw_duration >= 500. - 1e-8
    origin = json.loads((attempt / 'MAIN_START.json').read_text())['origin_monotonic']
    tokenizer = AutoTokenizer.from_pretrained(p['model_path'], local_files_only=True, trust_remote_code=False)
    counts = retokenized_counts(work, calls, tokenizer)
    lifetimes = {}
    for line in (attempt / 'LIFETIME_TIMINGS.jsonl').open():
        r = json.loads(line)
        if r['phase'] == 'main':
            k = (r['collection_id'], r['stage_id'], r['request_id'])
            assert k not in lifetimes
            lifetimes[k] = r['token_append_monotonic']
    ledger, details, attained = credited_ledger(work, calls, lifetimes, counts, origin)
    summary = window_statistics(ledger, raw_duration, horizon=500., bin_seconds=10.)
    summary['fixed_500s'] = summary.pop('fixed_hour')
    oracle = compute(root, work, calls, raw_duration, tokenizer)
    full = summary['complete_replay']
    checks = {'finite_pure200_actual_text_native_oracle_equal': oracle['credited_weighted_tokens'] == sum(full['credit_by_class'])
              and oracle['credited_weighted_tokens_by_class'] == full['credit_by_class'],
              'finite_pure200_credit_conserved': summary['conservation_verified'],
              'all_pure_main_raw_timing_requests': len(lifetimes) == len(counts) == 1197,
              'finite_replay_denominator_exact': full['duration_seconds'] == raw_duration}
    common = json.loads((attempt / 'common_audit_summary.json').read_text())
    expected = {(ph, j['collection_id'], sid, r['request_id']): r for ph in ['warmup', 'main'] for j in work[ph] for sid, rs in enumerate(j['stages']) for r in rs}
    gen = p['generation']
    checks['all_actual_sampling_fields_preserved'] = all(all(row['generation'].get(k) == v for k, v in expected_engine_sampling(
        gen, expected[row['phase'], row['collection_id'], row['stage_id'], row['request_id']]['target_output_tokens']).items()) for row in common['requests'])
    with (attempt / 'CREDIT_LEDGER.jsonl').open('x') as f:
        for e in ledger:
            f.write(json.dumps(e) + '\n')
    for name, obj in [('FINITE_REPLAY_AND_500S.json', summary), ('NATIVE_METRIC_ORACLE.json', oracle), ('ACTUAL_TEXT_TOKEN_CREDITS.json', details)]:
        (attempt / name).write_text(json.dumps(obj, indent=2) + '\n')
    jcts = [c['client_chain_end_monotonic'] - c['client_chain_start_monotonic'] for c in calls if c['phase'] == 'main']
    metrics = {'jitserve_token_goodput': full['goodput_total'], 'complete_replay_token_goodput': full['goodput_total'],
               'jitserve_credited_weighted_tokens': sum(full['credit_by_class']), 'main_duration': raw_duration,
               'fixed_500s_settled_credit': sum(summary['fixed_500s']['credit_by_class']),
               'fixed_500s_settled_goodput': summary['fixed_500s']['goodput_total'],
               'post_500s_credit': sum(summary['post_window_credit_by_class']), 'drain_seconds': full['drain_seconds'],
               'compound_attained': attained.get(2, 0), 'compound_total': 200,
               'mean_JCT': statistics.fmean(jcts), 'p95_JCT': percentile(jcts, .95), 'max_JCT': max(jcts), 'request_failures': 0}
    return {'checks': checks, 'metrics': metrics, 'summary': summary, 'details': details}
