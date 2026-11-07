"""Terminal token-credit scenarios from the unchanged historical donor bank.

No test workload, future arrival, benchmark horizon or true remaining output is
available here. Pure DR continues to use the original StageForecaster unchanged.
"""
import statistics
import time


class CreditCurve:
    def __init__(self, rows, slack, *, supported=True, reason='supported'):
        self.rows = tuple(rows)  # (critical decode units, total work, full credit)
        self.slack = slack
        self.supported, self.reason = supported, reason
        self.work = max(1., statistics.fmean(x[1] for x in rows)) if rows else 1.

    def credit(self, seconds, pause=0.):
        if not self.rows or self.slack <= pause:
            return 0.
        return statistics.fmean(credit if critical * seconds <= self.slack - pause else 0.
                                for critical, _, credit in self.rows)

    def view(self, seconds):
        credit = self.credit(seconds)
        return {'expected_credit': credit, 'work': self.work,
                'probability': (sum(c * seconds <= self.slack for c, _, _ in self.rows)
                                / len(self.rows)) if self.rows and self.slack > 0 else 0.,
                'critical_seconds': statistics.fmean(c for c, _, _ in self.rows) * seconds if self.rows else 0.,
                'supported': self.supported, 'reason': self.reason}


def forecast_curve(model, meta, stage, views, slack):
    if slack <= 0:
        return CreditCurve([], slack, reason='root_deadline_expired')
    started = time.perf_counter()
    model.stats['v49_curve_calls'] += 1
    try:
        width = meta['widths'][stage]
        live = {v['member'] for v in views}
        done = meta['observed'].get(stage, {})
        missing = [i for i in range(width) if i not in live and i not in done]
        donors = model.neighbors(meta, stage, views)
        if not donors:
            return CreditCurve([], slack, supported=False, reason='no_training_support')
        cap = meta.get('future_max_tokens', 1024)
        survivors = [row for row in donors if all(
            model.member_target(row, stage, v['member'], width, v['max_tokens']) > v['generated']
            for v in views)]
        prefix = sum(z.get('reward_input', z['declared_input']) + 8 * z.get('reward_output', z['output'])
                     for members in meta['observed'].values() for z in members.values())
        rows = []
        for row in survivors or donors:
            remaining = [max(0., model.member_target(row, stage, v['member'], width,
                                                    v['max_tokens']) - v['generated']) for v in views]
            if not survivors:
                remaining = [min(max(0, v['max_tokens'] - v['generated']), max(50., n))
                             for v, n in zip(views, remaining)]
            prefill = [v['prefill_tokens'] for v in views]
            credit = prefix + sum(v['reward_input'] + 8 * (v['generated'] + n)
                                  for v, n in zip(views, remaining))
            for member in missing:
                z = row['stages'][stage]
                index = min(len(z['outputs']) - 1, member * len(z['outputs']) // max(1, width))
                n = min(cap, z['outputs'][index]); inp = z['inputs'][min(index, len(z['inputs']) - 1)]
                remaining.append(n); prefill.append(inp); credit += inp + 8 * n
            work = sum(remaining) + sum(prefill) / model.prefill_unit
            critical = max(remaining, default=0.) + max(prefill, default=0.) / model.prefill_unit
            future_work, future_critical, future_credit = model.future_components(row, stage, meta['widths'], cap)
            rows.append((model.factor * (critical + future_critical) + model.guard_units,
                         work + future_work, credit + future_credit))
        return CreditCurve(rows, slack)
    finally:
        model.stats['v49_curve_seconds'] += time.perf_counter() - started
