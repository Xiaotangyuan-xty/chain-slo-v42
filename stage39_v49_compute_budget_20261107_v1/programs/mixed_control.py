"""Mixed terminal-credit rate search with an independent compute admission cost.

Physical KV/slot fit remains necessary; it is never proof that another active
decode member has no cost. Historical forecasts are evaluated over a frozen
online cost snapshot once per candidate batch width, not re-fit from test data.
"""
import math
from chain_admission import blocks
from step_cost import StepCost


def eligible_decode_count(generated, remaining, elapsed, prefill_seconds, unit, ttft, tbt):
    """Native cumulative deadlines, including possible recovery after late TTFT."""
    lo, hi = max(1, 2 - generated), remaining
    if hi < lo:
        return 0
    intercept = elapsed + prefill_seconds - ttft - (generated - 1) * tbt
    slope = unit - tbt
    if abs(slope) < 1e-12:
        return hi - lo + 1 if intercept <= 1e-12 else 0
    boundary = -intercept / slope
    if slope > 0:
        hi = min(hi, math.floor(boundary + 1e-10))
    else:
        lo = max(lo, math.ceil(boundary - 1e-10))
    return max(0, hi - lo + 1)


def foreground_value(kind, prompt, generated, prediction, cap, prefill, elapsed,
                     slo, unit, prefill_unit=256., penalty=100):
    prediction = min(cap, max(1, int(prediction)))
    remaining = max(0, prediction - generated)
    if not remaining and generated < cap:
        remaining = min(cap - generated, 50)  # Existing v47 survival backoff.
    prefill_work = prefill / prefill_unit
    work = max(1., remaining + prefill_work)
    first = max(1., prefill_work) * unit if prefill else unit
    if kind == 0:
        pc = prompt if generated == 0 and elapsed + first <= slo[0] else 0
        n = eligible_decode_count(generated, remaining, elapsed, first - unit,
                                  unit, slo[0], slo[1])
        reward = pc + 8 * n  # Already settled prefix credit is never re-earned.
        probability = float(reward > 0)
        slack = slo[0] + slo[1] * max(0, generated + remaining - 1) - elapsed
    else:
        finish = elapsed + work * unit
        probability = min(1., max(0., slo[2] / max(1e-9, finish))) ** penalty
        if elapsed >= slo[2]:
            probability = 0.
        reward = (prompt + 8 * (generated + remaining)) * probability
        slack = slo[2] - elapsed
    return {'expected_credit': reward, 'density': reward / work,
            'work': work, 'stage_work': work, 'probability': probability,
            'critical_seconds': work * unit, 'slack': slack, 'prefix_at_risk': 0.}


def _view(action, unit):
    if 'curve' in action:
        return action['curve'].view(unit)
    context = action['foreground']
    return foreground_value(*context, unit, penalty=action.get('penalty', 100))


def plan(actions, members, running, capacity, slots, unit, block_size=16,
         *, costs=None, previous=(), lease_remaining=None, horizon=50,
         candidate_multiplier=3, prefill_unit=256.):
    by_id = {m['request']: m for m in members}
    if len(by_id) != len(members) or slots < 1 or capacity < 1:
        raise ValueError('Invalid physical candidate/budget')
    owner = {}
    for action in actions:
        if not action['members']:
            raise ValueError('Empty action')
        for rid in action['members']:
            if rid in owner or rid not in by_id:
                raise ValueError('Invalid/duplicate action membership')
            owner[rid] = action['id']
    if set(owner) != set(by_id):
        raise ValueError('Every live physical member needs an owner')
    action_map = {a['id']: a for a in actions}
    if len(action_map) != len(actions):
        raise ValueError('Duplicate action identity')
    charges = {rid: blocks(m, block_size, horizon) for rid, m in by_id.items()}
    costs = costs or StepCost().snapshot(unit, max(1, len(running)))
    lease_remaining = lease_remaining or {}
    maximum = min(slots, len(members))
    units = {n: costs.seconds(n) for n in range(1, maximum + 1)}
    values, rates = {}, {}
    for aid, action in action_map.items():
        values[aid] = {n: _view(action, seconds) for n, seconds in units.items()}
        width = len(action['members'])
        rates[aid] = {n: v['expected_credit'] / max(1., v['work']) * width / units[n]
                      for n, v in values[aid].items()}
    def physical(ids):
        return {rid for aid in ids for rid in action_map[aid]['members']}
    def fits(ids):
        rs = physical(ids)
        return len(rs) <= slots and sum(charges[r] for r in rs) <= capacity
    def score(ids):
        count = len(physical(ids))
        return sum(rates[aid][count] for aid in ids) if count else 0.
    def positive(aid):
        width = len(action_map[aid]['members'])
        return max((v for n, v in rates[aid].items() if n >= width), default=0.)
    # Bound CPU work using the native top-k pool convention. Required work is
    # still registered outside the pool and is eligible for native drain.
    ranked = sorted(action_map, key=lambda aid: (-positive(aid), action_map[aid]['arrival'], aid))
    pool = [aid for aid in ranked[:max(1, slots * candidate_multiplier)] if positive(aid) > 0]
    chosen, decisions = set(), []
    while True:
        before = score(chosen); best = None; best_score = before
        for aid in pool:
            if aid in chosen or not fits(chosen | {aid}):
                continue
            value = score(chosen | {aid})
            if value > best_score + 1e-10 * max(1., abs(best_score)):
                best, best_score = aid, value
        if best is None:
            break
        chosen.add(best)
        decisions.append({'reason': 'positive_net_compute_credit', 'action': best,
                          'rate_before': before, 'rate_after': best_score})
    if not chosen:
        return None, {'reason': 'native_required_work_fallback', 'decisions': [],
                      'selected_charge': 0, 'backfill': [], 'evicted': [],
                      'cost_model': costs.describe(), 'action_pool': len(pool),
                      'all_actions': len(actions)}
    # Retention is conditional and bounded, rather than automatic for every
    # resident. Expired/zero-value work has no credit protection or restore rent.
    incumbent = {aid for aid, a in action_map.items()
                 if set(a['members']) <= set(previous) & set(running) and positive(aid) > 0}
    if incumbent and fits(incumbent):
        old_score, new_score = score(incumbent), score(chosen)
        old_n = len(physical(incumbent)); old_unit = units[old_n]
        pause = max(.001, horizon * old_unit)
        penalty = 0.
        for aid in incumbent - chosen:
            a = action_map[aid]; value = values[aid][old_n]
            if value['expected_credit'] <= 0:
                continue
            restore_units = sum((by_id[r]['prompt'] + by_id[r]['generated']) / prefill_unit
                                for r in a['members'])
            penalty += rates[aid][old_n] * min(1., restore_units / horizon)
            if 'curve' in a and any(lease_remaining.get(r, 0) > 0 for r in a['members']):
                lost = max(0., a['curve'].credit(old_unit)
                           - a['curve'].credit(old_unit, pause + restore_units * old_unit))
                penalty += lost / pause
        if new_score <= old_score + penalty + 1e-10:
            decisions.append({'reason': 'bounded_live_continuity', 'proposed_rate': new_score,
                              'incumbent_rate': old_score, 'switch_loss_rate_proxy': penalty})
            chosen = incumbent
    selected = physical(chosen); background = []
    current_score = score(chosen)
    # A learned plateau may allow already-prefilled background decode for free.
    # KV space alone, a cold model or a new background prefill never qualifies.
    for aid in ranked:
        a = action_map[aid]
        if aid in chosen or positive(aid) > 0 or not set(a['members']) <= set(running):
            continue
        if any(by_id[r].get('prefill_tokens', 0) for r in a['members']):
            continue
        trial = chosen | {aid}; after = len(physical(trial))
        if (fits(trial) and costs.evidenced_flat(len(selected), after)
                and score(trial) >= current_score - 1e-10):
            chosen = trial; selected = physical(chosen); background.extend(a['members'])
    actual_n = len(selected)
    details = []
    for aid, a in action_map.items():
        details.append({k: v for k, v in a.items() if k not in ['curve', 'foreground']})
        details[-1].update(values[aid][actual_n], selected=aid in chosen,
                           density=values[aid][actual_n]['expected_credit'] / max(1., values[aid][actual_n]['work']))
    return selected, {'reason': 'terminal_credit_compute_budget', 'decisions': decisions,
                      'objective_rate_proxy': score(chosen), 'selected_step_seconds': units[actual_n],
                      'selected_charge': sum(charges[r] for r in selected),
                      'backfill': background, 'evicted': sorted(set(running) - selected),
                      'actions': details, 'cost_model': costs.describe(),
                      'action_pool': len(pool), 'all_actions': len(actions)}
