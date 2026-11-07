"""v46 work boundaries with conditional full-chain token reward and static caches."""
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

    def estimate(self, meta, stage, views, slack, token_seconds):
        start = time.perf_counter()
        self.stats['estimate_calls'] += 1
        try:
            return self._estimate(meta, stage, views, slack, token_seconds)
        finally:
            self.stats['estimate_seconds'] += time.perf_counter() - start

    def _estimate(self, meta, stage, views, slack, token_seconds):
        if slack <= 0:
            self.stats['expired_short_circuits'] += 1
            return self.zero('whole_chain_deadline_expired', supported=True)
        width = meta['widths'][stage]
        live = {v['member'] for v in views}
        done = meta['observed'].get(stage, {})
        missing = [i for i in range(width) if i not in live and i not in done]
        donors = self.neighbors(meta, stage, views)
        if not donors:
            return self.zero('no_full_depth_training_support', len(missing))
        cap = meta.get('future_max_tokens', 1024)
        survivors = [r for r in donors if all(self.member_target(r, stage, v['member'], width,
                     v['max_tokens']) > v['generated'] for v in views)]
        selected = survivors or donors
        prefix_reward = sum(z.get('reward_input', z['declared_input']) + 8 * z.get('reward_output', z['output'])
                            for members in meta['observed'].values() for z in members.values())
        work, critical, current, rewards = [], [], [], []
        for row in selected:
            remaining = [max(0., self.member_target(row, stage, v['member'], width, v['max_tokens']) - v['generated']) for v in views]
            if not survivors:
                remaining = [min(max(0, v['max_tokens'] - v['generated']), max(50., n)) for v, n in zip(views, remaining)]
            prefill = [v['prefill_tokens'] for v in views]
            reward = prefix_reward + sum(v.get('reward_input', v.get('prompt', v['declared_input']))
                                          + 8 * (v['generated'] + n) for v, n in zip(views, remaining))
            for member in missing:
                z = row['stages'][stage]
                index = min(len(z['outputs']) - 1, member * len(z['outputs']) // max(1, width))
                n = min(cap, z['outputs'][index])
                inp = z['inputs'][min(index, len(z['inputs']) - 1)]
                remaining.append(n)
                prefill.append(inp)
                reward += inp + 8 * n
            w = sum(remaining) + sum(prefill) / self.prefill_unit
            c = max(remaining, default=0.) + max(prefill, default=0.) / self.prefill_unit
            current.append(max(remaining, default=0.))
            fw, fc, fr = self.future_components(row, stage, meta['widths'], cap)
            work.append(w + fw)
            critical.append((self.factor * (c + fc) + self.guard_units) * token_seconds)
            rewards.append(reward + fr)
        timely = [t <= slack for t in critical]
        probability = sum(timely) / len(timely)
        expected_credit = statistics.fmean(r if met else 0. for r, met in zip(rewards, timely))
        mean_work = max(1., statistics.fmean(work))
        return {'supported': True, 'probability': probability, 'remaining_work': mean_work,
                'critical_mean_seconds': statistics.fmean(critical),
                'current_remaining_tokens': statistics.fmean(current),
                'score': expected_credit / mean_work, 'expected_credited_tokens': expected_credit,
                'mean_chain_tokens': statistics.fmean(rewards), 'donors': len(selected),
                'survival_backoff': not bool(survivors), 'estimated_unsubmitted_members': len(missing)}
