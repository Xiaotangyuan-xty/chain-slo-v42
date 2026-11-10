"""Pure-chain goodput, using the unchanged native DeepResearch metric function."""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

SOURCE = 'repo/benchmark/benchmark_scheduler_deepresearch.py'

def native_function(root):
    import numpy as np
    path = Path(root) / SOURCE
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'calculate_deepresearch_metrics')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    ns = {'np': np, 'DeepResearchMetrics': NS}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
    return ns[fn.name]

def compute(root, work, calls, duration, tokenizer):
    main = [c for c in calls if c['phase'] == 'main']
    jobs = {j['collection_id']: j for j in work['main']}
    assert duration > 0 and len(main) == len(jobs)
    assert all(j['request_type'] == 2 for j in jobs.values())
    first = next(iter(jobs.values()))
    base = tuple(v / (first['collection_id'] % 4 + 1) for v in first['received_slo'])
    assert all(all(math.isclose(a, b*(j['collection_id'] % 4+1), rel_tol=1e-12, abs_tol=1e-12)
                       for a,b in zip(j['received_slo'],base)) for j in jobs.values())
    native = []
    chains = []
    credited = total_input = total_output = attained_count = 0
    for c in main:
        job = jobs[c['collection_id']]
        jct = c['client_chain_end_monotonic'] - c['client_chain_start_monotonic']
        attained = bool(c['success'] and jct <= c['received_slo'][2] * len(job['stages']))
        stages = []
        inputs = outputs = 0
        assert len(c['stages']) == len(job['stages'])
        for s in c['stages']:
            expected = {r['request_id']: r for r in job['stages'][s['stage_id']]}
            requests = []
            for r in s['requests']:
                prompt = expected[r['request_id']]['prompt']
                text = r['generated_text']
                assert hashlib.sha256(prompt.encode()).hexdigest() == r['input_prompt_sha256']
                inp = len(tokenizer(prompt, add_special_tokens=False).input_ids)
                out = len(tokenizer(text, add_special_tokens=False).input_ids)
                inputs += inp; outputs += out
                requests.append(NS(success=r['success'], request_input=prompt, request_output=text,
                    request_latency=r['client_request_end_monotonic']-r['client_request_start_monotonic'],
                    request_finish_time=r['client_request_end_monotonic'], request_ttft=r['server_ttft'],
                    request_service_gain=r.get('logged_service_gain', 0)))
            times = s['requests']
            stages.append(NS(success=s['success'], stage_outputs=requests,
                stage_latency=max(r['client_request_end_monotonic'] for r in times)-min(r['client_request_start_monotonic'] for r in times)))
        native.append(NS(collection_id=c['collection_id'], success=c['success'], total_latency=jct, stage_outputs=stages))
        credit = inputs + 8 * outputs if attained else 0
        credited += credit; total_input += inputs; total_output += outputs; attained_count += attained
        chains.append({'collection_id':c['collection_id'], 'attained':attained, 'jct_seconds':jct,
                       'retokenized_input_tokens':inputs, 'retokenized_output_tokens':outputs, 'credited_weighted_tokens':credit})
    oracle = native_function(root)(native, duration, tokenizer, [50,95], slo_constraint=base)
    assert oracle.slo_met_collections == attained_count
    assert oracle.total_input_tokens == total_input and oracle.total_output_tokens == total_output
    assert abs(oracle.slo_token_goodput - credited/duration) < 1e-9
    return {'gate':'PASS', 'primary_metric':'jitserve_token_goodput',
            'jitserve_token_goodput':float(oracle.slo_token_goodput), 'credited_weighted_tokens':credited,
            'retokenized_input_tokens':total_input, 'retokenized_output_tokens':total_output,
            'finite_replay_seconds':duration, 'compound_attained':attained_count, 'compound_total':len(main),
            'native_function':SOURCE+':calculate_deepresearch_metrics',
            'native_source_sha256':hashlib.sha256((Path(root)/SOURCE).read_bytes()).hexdigest(),
            'tokenization':'Actual submitted prompt and returned generated text, add_special_tokens=False',
            'numerator':'Only successful whole-chain deadline attainment credits prompt tokens + 8*decode tokens',
            'denominator':'Main replay origin through last root completion, including arrival window and drain, excluding warmup/model load',
            'per_chain':chains}

def analyze(root, stage, attempt_name, duration):
    from transformers import AutoTokenizer
    root,stage=Path(root),Path(stage)
    read=lambda p:json.loads(p.read_text())
    p=read(stage/'PROTOCOL.json')
    if p['workload_kind']=='mixed':
        from jitserve_mixed_goodput import analyze as mixed_analyze
        return mixed_analyze(root,stage,attempt_name,duration)
    work=read(stage/'WORKLOAD.json')
    tokenizer=AutoTokenizer.from_pretrained(p['model_path'], local_files_only=True, trust_remote_code=False)
    result=compute(root,work,read(stage/'attempts'/attempt_name/'client_calls.json'),duration,tokenizer)
    (stage/'JITSERVE_GOODPUT.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
