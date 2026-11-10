"""Shared route/progress observation; no scheduling or budget changes."""
import atexit
from collections import Counter
import functools
import json
from pathlib import Path
import time


def install(server, output, protocol):
    output = Path(output)
    stream = (output / 'NATIVE_GRAPH_EVENTS.jsonl').open('x')
    phases = {}
    def emit(event, **fields):
        stream.write(json.dumps({'event': event, 'monotonic': time.perf_counter(), **fields}) + '\n')
        stream.flush()
    atexit.register(stream.close)
    previous = server._handle_collective_request
    @functools.wraps(previous)
    def route(info, request, prompt):
        wire_widths = request.get('requests_per_stage')
        declared_stages = request.get('num_stages')
        result = previous(info, request, prompt)
        cid = info.collection_id
        phase = 'warmup' if info.client_id == 900000 else 'main'
        phases[cid] = phase
        implicit = wire_widths is None
        structure = server.graph_ctx.collection_graph_unfinished_dict.get(cid) if implicit else None
        emit('route', collection_id=cid, stage_id=info.stage_id, request_id=info.request_id,
             phase=phase, wire_widths=wire_widths, declared_stages=declared_stages,
             native_is_deepresearch=result[0], application=getattr(info, 'application', None),
             chain_widths=getattr(info, 'chain_stage_widths', None),
             future_cap=getattr(info, 'chain_future_max_tokens', None),
             native_stage_finished=getattr(structure, 'stage_finished', None))
        return result
    server._handle_collective_request = route
    prior_complete = server._track_collection_completion
    @functools.wraps(prior_complete)
    def complete(cid, is_dr, input_length, output_length):
        result = prior_complete(cid, is_dr, input_length, output_length)
        if hasattr(server,'_comparison_observation_snapshot'):
            (output/'LIVE_EXECUTOR_OBSERVATION.json').write_text(json.dumps(server._comparison_observation_snapshot(),indent=2)+'\n')
        structure = server.graph_ctx.collection_graph_unfinished_dict.get(cid) if not is_dr else None
        emit('completion', collection_id=cid, phase=phases[cid], native_is_deepresearch=is_dr,
             native_stage_finished=getattr(structure, 'stage_finished', None),
             native_finished=getattr(structure, 'is_finished', None))
        return result
    server._track_collection_completion = complete
    if protocol['candidate_version'] == 'v47pro':
        from mixed_joint import MixedScheduler
        prior_add = MixedScheduler.add_seq_group
        @functools.wraps(prior_add)
        def add(self, seq_group):
            group = seq_group
            result = prior_add(self, seq_group=seq_group)
            info = group.request_info
            if getattr(info, 'application', None) == 'tot_native':
                meta = self.policy.chains[group.collection_id]
                emit('policy_registered', collection_id=info.collection_id, stage_id=info.stage_id,
                     request_id=info.request_id, phase=phases[info.collection_id],
                     chain_widths=list(meta['widths']),
                     future_cap=getattr(info, 'chain_future_max_tokens', None),
                     metadata_future_cap=meta.get('future_max_tokens'),
                     policy_type=type(self.policy).__module__+'.'+type(self.policy).__name__,
                     pure_core_active=getattr(self, '_pure_core_active', False))
            return result
        MixedScheduler.add_seq_group = add


def route_checks(output, work, method, phase):
    rows = [json.loads(line) for line in (Path(output) / 'NATIVE_GRAPH_EVENTS.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['phase'] == phase]
    jobs = [j for j in work[phase] if j['request_type'] == 2]
    checks = {'native_ToT_presence_matches_frozen_' + phase: any(j.get('application','dr') == 'tot_style' for j in jobs) == any(j.get('application','dr') == 'tot_style' for j in work[phase])}
    bad = []
    for job in jobs:
        cid = job['collection_id']
        events = [r for r in rows if r['collection_id'] == cid]
        routes = [r for r in events if r['event'] == 'route']
        completions = [r for r in events if r['event'] == 'completion']
        expected = {(sid, r['request_id']) for sid, stage in enumerate(job['stages']) for r in stage}
        if len(routes) != len(expected) or {(r['stage_id'], r['request_id']) for r in routes} != expected or len(completions) != len(expected):
            bad.append([cid, 'route/completion/member coverage'])
            continue
        is_tot = job.get('application','dr') == 'tot_style'
        if any(r['native_is_deepresearch'] == is_tot for r in routes + completions):
            bad.append([cid, 'wrong native route'])
        if is_tot:
            if any(r['wire_widths'] is not None or r['declared_stages'] != 4 or r['application'] != 'tot_native'
                   or r['chain_widths'] != [3, 1, 3, 1] or r['future_cap'] != 256
                   or r['native_stage_finished'] != r['stage_id'] for r in routes):
                bad.append([cid, 'implicit ToT structure/annotation mismatch'])
            if [r['native_stage_finished'] for r in completions] != [0, 0, 1, 2, 2, 2, 3, 4] or [r['native_finished'] for r in completions] != [False] * 7 + [True]:
                bad.append([cid, 'native eight-call state-machine progress'])
            if method == 'v47pro':
                registered = [r for r in events if r['event'] == 'policy_registered']
                if len(registered) != 8 or {(r['stage_id'], r['request_id']) for r in registered} != expected or any(r['chain_widths'] != [3, 1, 3, 1] or r['future_cap'] != 256 or r['metadata_future_cap']!=256 for r in registered):
                    bad.append([cid, 'v47pro scheduler metadata coverage'])
        elif any(r['wire_widths'] != [len(s) for s in job['stages']] for r in routes):
            bad.append([cid, 'DR explicit route changed'])
    checks['native_graph_route_and_progress_' + phase] = not bad
    return checks, bad


def shared_checks(output, protocol, expected_count):
    output=Path(output)
    receipt=json.loads((output/'SHARED_PREDICTION_RUNTIME.json').read_text())
    rows=[json.loads(line) for line in (output/'SHARED_PREDICTION_EVENTS.jsonl').read_text().splitlines()]
    return {'shared_prediction_protocol':receipt['version']=='shared_framed_dynamic_prediction_v1' and receipt['protocol']==protocol['predictor_protocol']=='length_framed_pickle_v1' and receipt['method']==protocol['candidate_version'] and receipt['port']==protocol['prediction_ports'][protocol['runtime_arm']],
            'no_prediction_fallback':not any(r['kind']=='prediction_fallback' for r in rows),
            'prediction_success_covers_requests':sum(r['kind']=='prediction_success' for r in rows)==expected_count}

def qualify_warmup(output, work, protocol, calls):
    output=Path(output)
    checks,bad=route_checks(output,work,protocol['candidate_version'],'warmup')
    warm=[c for c in calls if c['phase']=='warmup']
    checks['warmup_complete']=len(warm)==len(work['warmup']) and all(c['success']for c in warm)
    checks['warmup_expected_applications']={j.get('application','dr')for j in work['warmup']}==({'latency','deadline','dr','tot_style'}if protocol['workload_kind']=='mixed'else {'dr'})
    checks['warmup_TBT_coverage']=all(r['success']and len(r['server_tbt'])==r['target_output_tokens']-1 for c in warm for s in c['stages']for r in s['requests'])
    events=[json.loads(line)for line in(output/'common_events.jsonl').read_text().splitlines()]
    ready=[r for r in events if r['kind']=='engine_ready'];devices=[r for r in events if r['kind']=='device_identity']
    checks['model_and_resource_caps_before_main']=len(ready)==1 and all(ready[0].get(k)==v for k,v in protocol['expected_resource_caps'].items()) and ready[0]['model_path']==protocol['model_path']
    checks['assigned_GPU_identity_before_main']=len(devices)==1 and devices[0]['physical_gpu_index']==protocol['resources']['physical_gpu_index']and devices[0]['actual_uuid']==protocol['resources']['gpu_uuid']
    method=protocol['candidate_version']
    checks['Astar_budget_adapter_presence']=(output/'BASELINE_BUDGET_ADAPTER.json').exists()==(method=='Astar')
    receipt=json.loads((output/'COMPARISON_SUITE_RUNTIME.json').read_text())
    checks['correct_declared_runtime']=receipt['method']==method and receipt['frozen_candidate_modules_exact'] and receipt['candidate_scheduler_installed']==(method not in ['A','Astar'])
    expected_warmup_calls=protocol['workloads']['full']['warmup_call_count']
    checks['warmup_exact_declared_cohort']=len(warm)==len(work['warmup']) and sum(len(s['requests']) for c in warm for s in c['stages'])==expected_warmup_calls
    checks['v50_budget_adapter_presence']=(output/'V50_BUDGET_ADAPTER.json').exists()==(method=='v50')
    execution=json.loads((output/'LIVE_EXECUTOR_OBSERVATION.json').read_text())
    checks['real_executor_observer_installed']=execution['executor_hooked']and execution['execution_observer']['version']=='executor_host_boundary_v1'
    checks['real_executor_observations_before_main']=execution['execution_observer']['events']>0 and execution['execution_observer']['failed_calls']==0 and (method not in ['v47pro','v49pro','v50'] or execution['candidate_cost_models_present']and execution['cost_accepted_samples']==execution['execution_observer']['events'])
    checks['candidate_retains_v47_budget']=not(output/'V47PRO_BUDGET_ADAPTER.json').exists()
    from fairness_audit import checks as fairness_checks
    checks.update(fairness_checks(output,work,method,'warmup'))
    count=sum(len(s)for j in work['warmup']for s in j['stages'])
    until=time.monotonic()+10.
    while True:
        shared=shared_checks(output,protocol,count)
        if shared['prediction_success_covers_requests']or not shared['no_prediction_fallback']or time.monotonic()>=until:break
        time.sleep(.05)
    checks.update(shared)
    result={'gate':'PASS'if all(checks.values())else 'FAIL','checks':checks,'bad':bad,'warmup_excluded_from_main_metric':True,'method':method,'main_GPU_replay_may_proceed':all(checks.values())}
    if not all(checks.values()):
        (output/'WARMUP_NATIVE_TOT_GATE.json').write_text(json.dumps(result,indent=2)+'\n')
        raise RuntimeError('End-to-end warmup qualification failed: '+repr(checks))
    return result
