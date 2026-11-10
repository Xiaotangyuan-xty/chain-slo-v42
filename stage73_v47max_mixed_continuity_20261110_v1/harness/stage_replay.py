"""Injected async transport validates barriers without starting an engine."""
import asyncio
import time


def payload_for(root, stage_id, request, phase='main'):
    kind = root['request_type']
    app = root.get('application', 'dr' if kind == 2 else 'foreground')
    implicit_tot = app == 'tot_style'
    if implicit_tot and [len(stage) for stage in root['stages']] != [3, 1, 3, 1]:
        raise ValueError('Native ToT implicit counter requires the qualified3,1,3,1 blueprint')
    maximum = 256 if implicit_tot else 1024
    target = request['target_output_tokens']
    if target > maximum:
        raise ValueError('Frozen output target exceeds declared sampling cap')
    sl = root['received_slo']
    info = {'collection_id': root['collection_id'], 'stage_id': stage_id,
            'request_id': request['request_id'], 'request_type': kind,
            'state': request.get('state', ''), 'prompt': request['prompt'],
            'input_len': request['declared_input_tokens'], 'output_len': target,
            # Preserve the public benchmark auxiliary budget transparently.
            # This is a shared trace-derived hint, not an additional metric/JCT gate.
            'deadline': sl[0] + sl[1] * target if kind == 0 else sl[2]}
    data = {'request_info': info, 'slo_constraint': sl,
            'sampling_params': {'temperature': 0., 'top_p': 1., 'top_k': -1, 'seed': 42,
                                'ignore_eos': True, 'max_tokens': maximum, 'n': 1},
            'target_output_length': target, 'client_id': 900000 if phase == 'warmup' else 0,
            'stream': True}
    if kind == 2:
        data['num_stages'] = len(root['stages'])
        if not implicit_tot:
            data['requests_per_stage'] = [len(s) for s in root['stages']]
    # Reference responses, later prompts, and frozen branch-selection labels are not sent.
    return data


async def replay_root(root, send, clock=time.perf_counter, phase='main'):
    start = clock()
    events, outputs = [], []
    for sid, requests in enumerate(root['stages']):
        async def call(r):
            p = payload_for(root, sid, r, phase=phase)
            if root['request_type'] == 2:
                p['served_time'] = clock() - start
            events.append({'stage': sid, 'request_id': r['request_id'], 'event': 'start', 'time': clock()})
            result = await send(p)
            events.append({'stage': sid, 'request_id': r['request_id'], 'event': 'finish', 'time': clock()})
            if not result.get('success', False):
                raise RuntimeError('Request failed; preserve case and do not retry')
            return result
        # Full stage barrier, including slow siblings, before any next-stage submission.
        outputs.append(await asyncio.gather(*(call(r) for r in requests)))
    return {'outputs': outputs, 'events': events, 'complete': True}
