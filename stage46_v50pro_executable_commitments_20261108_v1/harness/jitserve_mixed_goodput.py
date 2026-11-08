"""Native JITServe mixed token credit, with independent numerator verification."""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

SOURCE = 'repo/benchmark/benchmark_scheduler.py'
SOURCE_SHA = '208a817a166709ce69ea4eb9d85ba53f590d0b4494bc45fdd43b8d9d91928227'


def native_function(root):
    import numpy as np
    import warnings
    path = Path(root) / SOURCE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'calculate_metrics')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    namespace = {'np': np, 'warnings': warnings, 'BenchmarkMetrics': NS}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[fn.name]


def compute(root, work, calls, duration, tokenizer):
    main = [c for c in calls if c['phase'] == 'main']
    jobs = {j['collection_id']: j for j in work['main']}
    assert duration > 0 and len(main) == len(jobs)
    first = next(iter(jobs.values()))
    base = tuple(v / (first['collection_id'] % 4 + 1) for v in first['received_slo'])
    native, rows, chains = [], [], []
    credits = [0, 0, 0]
    total_input = total_output = attained = 0
    for c in main:
        cid, kind = c['collection_id'], c['request_type']
        job = jobs[cid]
        assert kind in [0, 1, 2]
        assert all(math.isclose(a, b * (cid % 4 + 1), abs_tol=1e-10) for a, b in zip(c['received_slo'], base))
        jct = c['client_chain_end_monotonic'] - c['client_chain_start_monotonic']
        met = bool(c['success'] and jct <= c['received_slo'][2] * len(job['stages']))
        chain_credit = 0
        outputs = []
        for stage in c['stages']:
            expected = {r['request_id']: r for r in job['stages'][stage['stage_id']]}
            for r in stage['requests']:
                prompt = expected[r['request_id']]['prompt']
                assert hashlib.sha256(prompt.encode()).hexdigest() == r['input_prompt_sha256']
                inp = len(tokenizer(prompt, add_special_tokens=False).input_ids) if r['success'] else 0
                out = len(tokenizer(r['generated_text'], add_special_tokens=False).input_ids) if r['success'] else 0
                total_input += inp
                total_output += out
                latency = r['client_request_end_monotonic'] - r['client_request_start_monotonic']
                credit = 0
                if r['success']:
                    if kind == 0:
                        elapsed = r['server_ttft']
                        good = 0
                        for index, interval in enumerate(r['server_tbt']):
                            elapsed += interval
                            good += elapsed <= c['received_slo'][0] + c['received_slo'][1] * (index + 1)
                        credit = (inp if r['server_ttft'] <= c['received_slo'][0] else 0) + 8 * good
                    elif (kind == 1 and latency <= c['received_slo'][2]) or (kind == 2 and met):
                        credit = inp + 8 * out
                credits[kind] += credit
                chain_credit += credit
                outputs.append(NS(request_type=kind, collection_id=cid, success=r['success'],
                                  request_input=prompt, request_output=r['generated_text'],
                                  request_ttft=r['server_ttft'], request_tbt=r['server_tbt'],
                                  request_latency=latency, request_finish_time=r['client_request_end_monotonic'],
                                  request_service_gain=r.get('logged_service_gain', 0), finish_before_ddl=met))
                rows.append({'collection_id': cid, 'stage_id': stage['stage_id'], 'request_id': r['request_id'],
                             'request_type': kind, 'retokenized_input_tokens': inp, 'retokenized_output_tokens': out,
                             'credited_weighted_tokens': int(credit)})
        native.append(outputs)
        if kind == 2:
            attained += met
            chains.append({'collection_id': cid, 'attained': met, 'jct_seconds': jct,
                           'credited_weighted_tokens': chain_credit})
    oracle = native_function(root)(native, duration, tokenizer, [50, 95], slo_constraint=base)
    numerator = sum(credits)
    native_numerator = sum(r['token_goodput'] for r in oracle.request_details)
    assert native_numerator == numerator
    assert oracle.request_total_input == total_input and oracle.request_total_output == total_output
    for kind in [0, 1, 2]:
        assert sum(r['token_goodput'] for r in oracle.request_details if r['request_type'] == kind) == credits[kind]
    return {'gate': 'PASS', 'primary_metric': 'jitserve_token_goodput',
            'jitserve_token_goodput': numerator / duration, 'credited_weighted_tokens': numerator,
            'credited_weighted_tokens_by_class': credits, 'retokenized_input_tokens': total_input,
            'retokenized_output_tokens': total_output, 'finite_replay_seconds': duration,
            'compound_attained': attained, 'compound_total': len(chains),
            'native_function': SOURCE + ':calculate_metrics', 'native_source_sha256': SOURCE_SHA,
            'native_request_counts': oracle.request_num,
            'native_request_slo_meet_counts': oracle.request_slo_meet,
            'native_request_count_caveat': 'Preserved native counter is separate from strict client-TTFT/all-TBT class0 attainment in RUN_GATE classes.',
            'tokenization': 'Actual prompt/returned text; add_special_tokens=False',
            'numerator': 'Native mixed semantics: latency partial token credit; deadline full credit if timely; chain full credit iff whole chain timely.',
            'denominator': 'Main origin to last root completion, including one-hour arrival window and full drain; warmup/loading excluded.',
            'per_chain': chains, 'per_request': rows}


def analyze(root, stage, attempt_name, duration):
    from transformers import AutoTokenizer
    stage = Path(stage)
    protocol = json.loads((stage / 'PROTOCOL.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True, trust_remote_code=False)
    work = json.loads((stage / 'WORKLOAD.json').read_text())
    calls = json.loads((stage / 'attempts' / attempt_name / 'client_calls.json').read_text())
    result = compute(root, work, calls, duration, tokenizer)
    (stage / 'JITSERVE_GOODPUT.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
