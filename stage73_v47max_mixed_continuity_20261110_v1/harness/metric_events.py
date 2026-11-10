"""Native weighted-token credit, attributed to user-selected settlement time.

SLO comes from each actual request, never collection_id %4. An unchanged
public native-function oracle can validate each root with adjusted base SLO.
No GPU or serving process is started by this module.
"""
from collections import Counter
import hashlib
import math


def request_key(r):
    return r['collection_id'], r['stage_id'], r['request_id']


def retokenized_counts(work, calls, tokenizer):
    """Actual submitted/returned text, add_special_tokens=False; never target counts."""
    jobs = {j['collection_id']: j for j in work['main']}
    counts = {}
    for root in calls:
        if root['phase'] != 'main':
            continue
        job = jobs[root['collection_id']]
        for stage in root['stages']:
            expected = {r['request_id']: r for r in job['stages'][stage['stage_id']]}
            for request in stage['requests']:
                k = request_key(request)
                if k in counts:
                    raise ValueError('Duplicate request while retokenizing')
                prompt = expected[request['request_id']]['prompt']
                if hashlib.sha256(prompt.encode()).hexdigest() != request['input_prompt_sha256']:
                    raise ValueError('Actual submitted prompt differs')
                counts[k] = (len(tokenizer(prompt, add_special_tokens=False).input_ids),
                             len(tokenizer(request['generated_text'], add_special_tokens=False).input_ids))
    return counts


def credited_ledger(work, calls, lifetime, token_counts, origin):
    jobs = {j['collection_id']: j for j in work['main']}
    main = [c for c in calls if c['phase'] == 'main']
    if len(main) != len(jobs) or {c['collection_id'] for c in main} != set(jobs):
        raise ValueError('Incomplete/duplicate main cohort; no partial result promoted as valid')
    ledger, details, attained = [], [], Counter()
    failed = 0
    def add(timestamp, amount, kind, cid, sid=None, rid=None):
        if amount:
            t = timestamp - origin
            if not math.isfinite(t) or t < 0:
                raise ValueError('Credit before main origin or invalid timestamp')
            ledger.append({'time_seconds': t, 'weighted_tokens': int(amount), 'type': kind,
                           'collection_id': cid, 'stage_id': sid, 'request_id': rid})
    for root in main:
        cid, kind = root['collection_id'], root['request_type']
        job = jobs[cid]
        if root['request_type'] != job['request_type'] or root['received_slo'] != job['received_slo'] or len(root['stages']) != len(job['stages']):
            raise ValueError('SLO or stage identity differs from frozen workload')
        if not root['success']:
            raise ValueError('Failed root: preserve technical failure, do not publish valid goodput')
        slo = root['received_slo']
        chain_met = root['client_chain_end_monotonic'] - root['client_chain_start_monotonic'] <= slo[2] * len(job['stages'])
        chain_credit = 0
        previous_end = None
        for expected_sid, stage in enumerate(root['stages']):
            if stage['stage_id'] != expected_sid:
                raise ValueError('Missing/reordered stage identity')
            expected_requests = {r['request_id']: r for r in job['stages'][expected_sid]}
            if {r['request_id'] for r in stage['requests']} != set(expected_requests):
                raise ValueError('Request member identity mismatch')
            if previous_end is not None and min(r['client_request_start_monotonic'] for r in stage['requests']) < previous_end:
                raise ValueError('Stage barrier violated')
            previous_end = max(r['client_request_end_monotonic'] for r in stage['requests'])
            if len(stage['requests']) != len(job['stages'][stage['stage_id']]):
                raise ValueError('Stage width mismatch')
            seen = set()
            for r in stage['requests']:
                k = request_key(r)
                expected_request = expected_requests[r['request_id']]
                expected_hash = hashlib.sha256(expected_request['prompt'].encode()).hexdigest()
                if r['collection_id'] != cid or r['stage_id'] != expected_sid or r['input_prompt_sha256'] != expected_hash:
                    raise ValueError('Request identity or actual input prompt mismatch')
                if k in seen or not r['success']:
                    raise ValueError('Duplicate/failed request')
                seen.add(k)
                if r['received_slo'] != slo:
                    raise ValueError('Request SLO differs from root SLO')
                times = lifetime[k]
                if len(times) != r['target_output_tokens'] or len(r['server_tbt']) != len(times) - 1:
                    raise ValueError('Incomplete raw token/TBT timing coverage')
                if any(b < a for a, b in zip(times, times[1:])):
                    raise ValueError('Nonmonotone token append times')
                pin, pout = token_counts[k]
                credit = 0
                if kind == 0:
                    # Preserve the public native formula's first-output-token exclusion.
                    pc = pin if r['server_ttft'] <= slo[0] else 0
                    add(times[0], pc, kind, cid, r['stage_id'], r['request_id'])
                    credit += pc
                    elapsed = r['server_ttft']
                    for i, dt in enumerate(r['server_tbt']):
                        elapsed += dt
                        if elapsed <= slo[0] + slo[1] * (i + 1):
                            add(times[i + 1], 8, kind, cid, r['stage_id'], r['request_id'])
                            credit += 8
                    first_text = r.get('client_first_text_monotonic')
                    attained[0] += bool(first_text is not None and first_text - r['client_request_start_monotonic'] <= slo[0]
                                        and all(tbt <= slo[1] for tbt in r['server_tbt']))
                elif kind == 1:
                    met = r['client_request_end_monotonic'] - r['client_request_start_monotonic'] <= slo[2]
                    credit = pin + 8 * pout if met else 0
                    add(r['client_request_end_monotonic'], credit, kind, cid, r['stage_id'], r['request_id'])
                    attained[1] += met
                elif kind == 2:
                    credit = pin + 8 * pout if chain_met else 0
                    chain_credit += credit
                else:
                    raise ValueError('Unknown request type')
                details.append({'collection_id': cid, 'stage_id': r['stage_id'], 'request_id': r['request_id'],
                                'type': kind, 'credited_weighted_tokens': credit,
                                'retokenized_input_tokens': pin, 'retokenized_output_tokens': pout})
        if kind == 2:
            attained[2] += chain_met
            add(root['client_chain_end_monotonic'], chain_credit, 2, cid)
    ledger.sort(key=lambda e: (e['time_seconds'], e['type'], e['collection_id']))
    expected = sum(r['credited_weighted_tokens'] for r in details)
    assert sum(e['weighted_tokens'] for e in ledger) == expected
    return ledger, details, dict(attained)


def window_statistics(ledger, last_completion_seconds, horizon=3600., bin_seconds=60.):
    if horizon <= 0 or bin_seconds <= 0:
        raise ValueError('Positive window/bin width required')
    bins = [{'start_seconds': i * bin_seconds, 'end_seconds': min(horizon, (i + 1) * bin_seconds),
             'credit_by_class': [0, 0, 0]} for i in range(math.ceil(horizon / bin_seconds))]
    inside, after, full = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    for event in ledger:
        t, weight, kind = event['time_seconds'], event['weighted_tokens'], event['type']
        if t < 0 or not math.isfinite(t) or weight < 0:
            raise ValueError('Invalid event')
        full[kind] += weight
        if t < horizon:  # [0,H), including exact left edges, excluding exact H
            inside[kind] += weight
            bins[int(t // bin_seconds)]['credit_by_class'][kind] += weight
        else:
            after[kind] += weight
    for row in bins:
        width = row['end_seconds'] - row['start_seconds']
        row['goodput_by_class'] = [v / width for v in row['credit_by_class']]
        row['goodput_total'] = sum(row['credit_by_class']) / width
    assert all(full[i] == inside[i] + after[i] == sum(b['credit_by_class'][i] for b in bins) + after[i] for i in range(3))
    duration = max(horizon, last_completion_seconds)
    return {'schema': 'native-credit-settlement-window/v1',
            'window': '[main_origin,main_origin+H), fixed common origin; warmup excluded',
            'attribution': 'latency prompt/eligible decode at token append; deadline at request completion; compound at whole-root completion',
            'weighted_token_metric': 'native prompt+8*decode rules; latency native first-decode exclusion retained',
            'fixed_hour': {'seconds': horizon, 'credit_by_class': inside,
                           'goodput_by_class': [v / horizon for v in inside], 'goodput_total': sum(inside) / horizon},
            'post_window_credit_by_class': after,
            'complete_replay': {'duration_seconds': duration, 'raw_last_completion_seconds': last_completion_seconds,
                                'drain_seconds': max(0., last_completion_seconds - horizon),
                                'credit_by_class': full, 'goodput_total': sum(full) / duration},
            'conservation_verified': True, 'bin_seconds': bin_seconds, 'bins': bins,
            'interpretation': 'Completion-gated compound credit has lumpy recognition. This is credit flow, not instantaneous GPU output throughput.'}


def native_oracle_per_root(native_function, work, calls, tokenizer, duration):
    """Unchanged native function, base SLO adjusted per root to cancel its id rule.

    IDs on the wire and in evidence are unchanged. Only the oracle's base-SLO
    argument is supplied per root; actual received SLO remains authoritative.
    """
    from types import SimpleNamespace as NS
    jobs = {j['collection_id']: j for j in work['main']}
    total, per_class = 0, [0, 0, 0]
    for c in calls:
        if c['phase'] != 'main':
            continue
        job = jobs[c['collection_id']]
        met = c['success'] and c['client_chain_end_monotonic'] - c['client_chain_start_monotonic'] <= c['received_slo'][2] * len(job['stages'])
        outputs = []
        for s in c['stages']:
            expected = {r['request_id']: r for r in job['stages'][s['stage_id']]}
            for r in s['requests']:
                outputs.append(NS(request_type=c['request_type'], collection_id=c['collection_id'], success=r['success'],
                                  request_input=expected[r['request_id']]['prompt'], request_output=r['generated_text'],
                                  request_ttft=r['server_ttft'], request_tbt=r['server_tbt'],
                                  request_latency=r['client_request_end_monotonic'] - r['client_request_start_monotonic'],
                                  request_finish_time=r['client_request_end_monotonic'],
                                  request_service_gain=r.get('logged_service_gain', 0), finish_before_ddl=met))
        base = tuple(v / (c['collection_id'] % 4 + 1) for v in c['received_slo'])
        oracle = native_function([outputs], duration, tokenizer, [], slo_constraint=base)
        credit = sum(d['token_goodput'] for d in oracle.request_details)
        total += credit
        per_class[c['request_type']] += credit
    return {'credited_weighted_tokens': int(total), 'credit_by_class': list(map(int, per_class)),
            'oracle': 'unchanged published calculate_metrics, per-root base-SLO argument; no online ID/SLO mutation'}
