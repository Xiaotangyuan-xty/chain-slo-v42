"""Donor scenarios retain stage width, prefill and context; no evaluation truth.

A stage is a conservative barrier envelope: members stay at declared width until
its longest member finishes. The rollout explicitly scales service when only a
wave fits. This is an approximation, not a GPU latency or calibrated probability
oracle. Future profiles come only from the frozen historical donor bank.
"""
from collections import OrderedDict
from dataclasses import dataclass
import statistics
import math
import time


@dataclass(frozen=True)
class Stage:
    width: int
    decode: float
    prefill: float
    context: float
    gap_units: float = 0.


@dataclass(frozen=True)
class Scenario:
    stages: tuple
    credit: float
    donor: object = None


class Scenarios:
    def __init__(self, rows, slack, *, supported=True, reason='supported'):
        self.rows = tuple(rows)
        self.slack = slack
        self.supported, self.reason = supported, reason

    def isolated(self, costs, current_members=None, planning_seconds=0., prefill_limit=2048):
        times = []
        for row in self.rows:
            elapsed = planning_seconds
            for sid, stage in enumerate(row.stages):
                n = current_members if sid == 0 and current_members else stage.width
                step = costs.seconds(n, stage.context * n / max(1, stage.width), 0)
                pref = costs.seconds(n, stage.context * n / max(1, stage.width),
                                     min(prefill_limit, stage.prefill)) if stage.prefill else 0.
                elapsed += stage.decode * step + (math.ceil(stage.prefill / prefill_limit) * pref if stage.prefill else 0.) + stage.gap_units * costs.seconds(stage.width, stage.context, 0)
            times.append(elapsed)
        timely = [t <= self.slack for t in times]
        return {'probability': statistics.fmean(timely) if timely else 0.,
                'expected_credit': statistics.fmean(r.credit if ok else 0. for r, ok in zip(self.rows, timely)) if timely else 0.,
                'scenario_completion_seconds': times}


class ScenarioCache:
    """Bounded immutable future-stage templates; current observations never cached away."""
    def __init__(self, capacity=2048):
        self.rows = OrderedDict()
        self.capacity = capacity
        self.hits = self.misses = 0

    def future(self, model, row, stage, widths, cap):
        key = (model, row['id'], stage, tuple(widths), cap)
        if key in self.rows:
            self.hits += 1; self.rows.move_to_end(key)
            return self.rows[key]
        self.misses += 1; result = []; credit = 0.
        for sid in range(stage + 1, len(widths)):
            z = row['stages'][sid]; width = widths[sid]
            outputs = [model.member_target(row, sid, i, width, cap) for i in range(width)]
            inputs = [z['inputs'][min(len(z['inputs'])-1, i*len(z['inputs'])//width)] for i in range(width)]
            result.append(Stage(width, model.factor * max(outputs, default=0.),
                                model.factor * sum(inputs), sum(inputs) + sum(outputs)/2,
                                model.gap_units))
            credit += sum(inputs) + 8 * sum(outputs)
        value = (tuple(result), credit); self.rows[key] = value
        if len(self.rows) > self.capacity:self.rows.popitem(last=False)
        return value


CACHE = ScenarioCache()


def forecast_scenarios(model, meta, stage, views, slack, max_scenarios=8):
    if slack <= 0:return Scenarios([], slack, reason='root_deadline_expired')
    started = time.perf_counter(); model.stats['v49pro_scenario_calls'] += 1
    try:
        width = meta['widths'][stage]; live = {v['member'] for v in views}
        done = meta['observed'].get(stage, {})
        missing = [i for i in range(width) if i not in live and i not in done]
        donors = model.neighbors(meta, stage, views)
        if not donors:return Scenarios([], slack, supported=False, reason='no_training_support')
        survivors = [r for r in donors if all(model.member_target(r, stage, v['member'], width, v['max_tokens']) > v['generated'] for v in views)]
        selected = survivors or donors
        # Fixed, evenly spaced support indices; no runtime outcome based selection.
        if len(selected) > max_scenarios:
            selected = [selected[i * len(selected) // max_scenarios] for i in range(max_scenarios)]
        cap = meta.get('future_max_tokens', 1024)
        prefix = sum(z.get('reward_input', z['declared_input']) + 8*z.get('reward_output', z['output']) for ms in meta['observed'].values() for z in ms.values())
        rows = []
        for row in selected:
            remaining = [max(0., model.member_target(row, stage, v['member'], width, v['max_tokens'])-v['generated']) for v in views]
            if not survivors:remaining = [min(max(0, v['max_tokens']-v['generated']), max(50., n)) for v,n in zip(views,remaining)]
            prefills = [v['prefill_tokens'] for v in views]
            contexts = [v['prompt']+v['generated']+n/2 for v,n in zip(views,remaining)]
            credit = prefix + sum(v['reward_input']+8*(v['generated']+n) for v,n in zip(views,remaining))
            for member in missing:
                z = row['stages'][stage]; i = min(len(z['inputs'])-1, member*len(z['inputs'])//width)
                n = model.member_target(row,stage,member,width,cap); inp=z['inputs'][i]
                remaining.append(n);prefills.append(inp);contexts.append(inp+n/2);credit+=inp+8*n
            first=Stage(max(1,len(remaining)),model.factor*max(remaining,default=0.),model.factor*sum(prefills),sum(contexts))
            future, reward = CACHE.future(model,row,stage,meta['widths'],cap)
            stages=[first,*future]
            # Safety guard is an explicit final-stage service envelope.
            last=stages[-1];stages[-1]=Stage(last.width,last.decode+model.guard_units,last.prefill,last.context,last.gap_units)
            rows.append(Scenario(tuple(stages),credit+reward,row['id']))
        return Scenarios(rows,slack,reason='survival_backoff'if not survivors else 'supported')
    finally:model.stats['v49pro_scenario_seconds']+=time.perf_counter()-started
