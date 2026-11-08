"""v47 empirical root reward with lightweight current-batch repricing."""
from collections import Counter, OrderedDict
import json
import math
from pathlib import Path
import statistics
import time


class StageForecaster:
    def __init__(self, model, future_cache_capacity=8192):
        self.model = model
        self.records = model['records']
        self.k = model['neighbors']
        self.prefill_unit = model['prefill_tokens_per_decode_unit']
        self.gap_units = model['interstage_gap_decode_units']
        self.guard_units = model['forecast_guard_decode_units']
        self.factor = model['service_safety_factor']
        self.features = {r['id']: [(math.log1p(sum(z['inputs'])), math.log1p(max(z['outputs'])),
                                    len(z['outputs'])) for z in r['stages']] for r in self.records}
        self.future_cache = OrderedDict()
        self.future_cache_capacity = future_cache_capacity
        self.stats = Counter()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def neighbors(self, meta, stage, views):
        observed, widths = meta['observed'], meta['widths']
        current = {i: z['declared_input'] for i, z in observed.get(stage, {}).items()}
        current.update({v['member']: v['declared_input'] for v in views})
        signature = (tuple(widths), meta['stages'], tuple(sorted(current.items())),
                     tuple((sid, tuple((i, z['declared_input'], z['output'])
                           for i, z in sorted(members.items())))
                           for sid, members in sorted(observed.items()) if sid < stage))
        cached = meta['neighbor_cache'].get(stage)
        if cached is not None and cached[0] == signature:
            self.stats['neighbor_cache_hits'] += 1
            return cached[1]
        self.stats['neighbor_cache_misses'] += 1
        current_log = math.log1p(sum(current.values()) * widths[stage] / max(1, len(current)))
        pool = [r for r in self.records if len(r['stages']) == meta['stages']]
        if not pool:
            pool = [r for r in self.records if len(r['stages']) >= meta['stages']]
        if not pool:
            return []

        def distance(row):
            fs = self.features[row['id']]
            score = abs(fs[stage][0] - current_log) + .35 * sum(abs(a - b[2]) for a, b in zip(widths, fs))
            for sid, members in observed.items():
                if sid < stage and sid < len(fs):
                    score += .5 * abs(math.log1p(sum(x['declared_input'] for x in members.values())) - fs[sid][0])
                    score += .5 * abs(math.log1p(max(x['output'] for x in members.values())) - fs[sid][1])
            return score, row['id']

        result = sorted(pool, key=distance)[:self.k]
        meta['neighbor_cache'][stage] = (signature, result)
        return result

    @staticmethod
    def member_target(row, stage, member, width, cap):
        outputs = row['stages'][stage]['outputs']
        index = min(len(outputs) - 1, member * len(outputs) // max(1, width))
        return min(cap, outputs[index])

    def future_components(self, row, stage, widths, cap):
        key = (row['id'], stage, tuple(widths), cap)
        if key in self.future_cache:
            self.stats['future_cache_hits'] += 1
            self.future_cache.move_to_end(key)
            return self.future_cache[key]
        self.stats['future_cache_misses'] += 1
        work = critical = reward = 0.
        for sid in range(stage + 1, len(widths)):
            z = row['stages'][sid]
            outputs = [min(cap, x) for x in z['outputs']]
            ratio = widths[sid] / max(1, len(outputs))
            work += (sum(outputs) + sum(z['inputs']) / self.prefill_unit) * ratio
            critical += max(outputs) + max(z['inputs']) / self.prefill_unit + self.gap_units
            reward += (sum(z['inputs']) + 8 * sum(outputs)) * ratio
        result = (work, critical, reward)
        self.future_cache[key] = result
        if len(self.future_cache) > self.future_cache_capacity:
            self.future_cache.popitem(last=False)
        return result

    @staticmethod
    def zero(reason, missing=0, supported=False):
        return {'supported': supported, 'probability': 0., 'remaining_work': None,
                'critical_mean_seconds': None, 'current_remaining_tokens': None,
                'score': 0., 'expected_credited_tokens': 0., 'mean_chain_tokens': None,
                'donors': 0, 'survival_backoff': False,
                'estimated_unsubmitted_members': missing, 'reason': reason}

    def estimate(self, meta, stage, views, slack, token_seconds=None, *, service=None,
                 capacity=1000000, slots=32, block_size=16, token_budget=2048):
        from service_model import ServiceModel
        start = time.perf_counter()
        self.stats['estimate_calls'] += 1
        if service is None:
            service = ServiceModel()
            if token_seconds is not None:
                service.observe_round(token_seconds)
        try:
            if slack <= 0:
                return self.zero('whole_chain_deadline_expired', supported=True)
            width = meta['widths'][stage]
            done = meta['observed'].get(stage, {})
            live = {v['member'] for v in views}
            missing = [i for i in range(width) if i not in live and i not in done]
            donors = self.neighbors(meta, stage, views)
            if not donors:
                return self.zero('no_full_depth_training_support', len(missing))
            cap = meta.get('future_max_tokens', 1024)
            survivors = [r for r in donors if all(
                self.member_target(r, stage, v['member'], width, v['max_tokens']) > v['generated']
                for v in views)]
            selected = survivors or donors
            prefix = sum(z.get('reward_input', z['declared_input']) + 8 * z.get('reward_output', z['output'])
                         for members in meta['observed'].values() for z in members.values())
            scenarios = []
            for row in selected:
                current = []
                remaining = []
                prefill = []
                reward = prefix
                for v in views:
                    n = max(0., self.member_target(row, stage, v['member'], width, v['max_tokens']) - v['generated'])
                    if not survivors:
                        n = min(max(0, v['max_tokens'] - v['generated']), max(50., n))
                    current.append({**v, 'remaining': n})
                    remaining.append(n)
                    prefill.append(v['prefill_tokens'])
                    reward += v.get('reward_input', v['prompt']) + 8 * (v['generated'] + n)
                for member in missing:
                    z = row['stages'][stage]
                    index = min(len(z['outputs']) - 1, member * len(z['outputs']) // max(1, width))
                    n = min(cap, z['outputs'][index])
                    inp = z['inputs'][min(index, len(z['inputs']) - 1)]
                    current.append({'request': None, 'prompt': inp, 'generated': 0,
                                    'prefill_tokens': inp, 'remaining': n, 'max_tokens': cap})
                    remaining.append(n)
                    prefill.append(inp)
                    reward += inp + 8 * n
                fw, fc, fr = self.future_components(row, stage, meta['widths'], cap)
                work = sum(remaining) + sum(prefill) / self.prefill_unit + fw
                critical = max(remaining, default=0.) + max(prefill, default=0.) / self.prefill_unit
                scenarios.append({'current': current, 'current_critical_units': critical,
                                  'future_critical_units': fc, 'remaining_work_units': work,
                                  'terminal_credit': reward + fr,
                                  'current_remaining': max(remaining, default=0.)})
            state = {'scenarios': scenarios, 'physical': {v['request']: v for v in views},
                     'slack': slack, 'service': service, 'capacity': capacity, 'slots': slots,
                     'block_size': block_size, 'token_budget': token_budget,
                     'supported': True, 'survival_backoff': not bool(survivors),
                     'estimated_unsubmitted_members': len(missing)}
            value = self._value(state)
            state['initial_score'] = value['score']
            return {**value, '_forecast': state}
        finally:
            self.stats['estimate_seconds'] += time.perf_counter() - start

    @staticmethod
    def _context(task):
        remaining = task.get('remaining', max(0, task.get('max_tokens', 0) - task.get('generated', 0)))
        return task['prompt'] + task.get('generated', 0) + remaining

    def _current_seconds(self, state, scenario, first, background):
        service = state['service']
        selected = [t for t in scenario['current'] if t.get('request') in first]
        if not selected:
            return math.inf
        batch = selected + background
        members = len(batch)
        context = sum(self._context(t) for t in batch)
        seconds = max(t['remaining'] for t in selected) * service.unit(members, context)
        prefill = sum(t.get('prefill_tokens', 0) for t in selected)
        seconds += math.ceil(prefill / state['token_budget']) * service.unit(
            members, context, min(prefill, state['token_budget']))
        if any(t.get('swapped') for t in selected):
            seconds += service.unit(members, context, 0, True)
        background_context = sum(self._context(t) for t in background)
        for task in scenario['current']:
            if task.get('request') in first:
                continue
            context = self._context(task) + background_context
            members = 1 + len(background)
            seconds += task['remaining'] * service.unit(members, context)
            prefill = task.get('prefill_tokens', 0)
            seconds += math.ceil(prefill / state['token_budget']) * service.unit(
                members, context, min(prefill, state['token_budget']))
            if task.get('swapped'):
                seconds += service.unit(members, context, 0, True)
        return seconds

    def _value(self, state, first=None, background=()):
        service = state['service']
        unit = service.unit(1)
        feasible = True
        if first is not None:
            from service_model import member_blocks
            first = set(first)
            selected = [state['physical'][r] for r in first if r in state['physical']]
            unique = {}
            for task in background:
                rid = task.get('request')
                if rid not in first:
                    unique[rid] = task
            background = list(unique.values())
            physical = selected + background
            feasible = bool(first) and len(selected) == len(first)
            feasible = feasible and len(physical) <= state['slots']
            feasible = feasible and sum(member_blocks(t, state['block_size']) for t in physical) <= state['capacity']
        critical = []
        works = []
        rewards = []
        current_remaining = []
        for scenario in state['scenarios']:
            if first is None:
                current = scenario['current_critical_units'] * unit
            elif feasible:
                current = self._current_seconds(state, scenario, first, background)
            else:
                current = math.inf
            future = scenario['future_critical_units'] * unit
            critical.append(self.factor * (current + future) + self.guard_units * unit)
            works.append(scenario['remaining_work_units'] * unit)
            rewards.append(scenario['terminal_credit'])
            current_remaining.append(scenario['current_remaining'])
        timely = [c <= state['slack'] for c in critical]
        expected = statistics.fmean(r if ok else 0. for r, ok in zip(rewards, timely))
        work = max(.001, statistics.fmean(works))
        finite = all(math.isfinite(c) for c in critical)
        return {'supported': True, 'probability': sum(timely) / len(timely),
                'remaining_work': work, 'critical_mean_seconds': statistics.fmean(critical) if finite else None,
                'current_remaining_tokens': statistics.fmean(current_remaining),
                'score': expected / work, 'expected_credited_tokens': expected,
                'mean_chain_tokens': statistics.fmean(rewards), 'donors': len(critical),
                'survival_backoff': state['survival_backoff'],
                'estimated_unsubmitted_members': state['estimated_unsubmitted_members'],
                'stage_waves_mean': None,
                'wave_model': 'current selected batch, serial residual, isolated future units'}

    def revalue(self, goal, first, background=()):
        start = time.perf_counter()
        try:
            if '_forecast' not in goal:
                return goal
            self.stats['selected_wave_revalues'] += 1
            state = goal['_forecast']
            value = self._value(state, first, background)
            if 'score_factor' not in state:
                initial = state['initial_score']
                state['score_factor'] = goal.get('score', initial) / initial if initial > 0 else 1.
            value['score'] *= state['score_factor']
            return {**goal, **value}
        finally:
            self.stats['revalue_seconds'] += time.perf_counter() - start
