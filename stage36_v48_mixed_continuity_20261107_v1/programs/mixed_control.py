"""Mixed-only marginal credit and resident-first, bounded backfill planning.

Work units and restore costs are explicit heuristics, not measured GPU latency.
Only current inputs/prefixes, public caps, predictions and deadlines are used.
"""
import math
from chain_admission import blocks


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


def interruption_loss(action, pause_units, unit, restore_units):
    """Credit opportunity cost; a short stage/critical prefix gets extra retention."""
    density = max(0., action['density'])
    loss = density * (pause_units + restore_units)
    live = action['probability'] > 0 and action['slack'] > 0
    margin = action['slack'] - action['critical_seconds']
    near_finish = action['stage_work'] <= restore_units + 1.
    critical_pause = margin <= (pause_units + restore_units) * unit
    if live and (near_finish or critical_pause):
        loss += action.get('prefix_at_risk', 0.) * action['probability']
    return loss


def plan(actions, members, running, capacity, slots, unit, block_size=16,
         horizon=50, prefill_unit=256.):
    """Keep residents unless real admission pressure justifies a costly swap.

    New positive work fills first, then native-ranked required work fills spare
    capacity. A zero forecast never removes required work from consideration.
    The backend allocator remains authoritative, including physical preemption.
    """
    by_id = {m['request']: m for m in members}
    if len(by_id) != len(members):
        raise ValueError('Duplicate physical request identity')
    charges = {rid: blocks(m, block_size, horizon) for rid, m in by_id.items()}
    owner = {}
    for action in actions:
        for rid in action['members']:
            if rid in owner or rid not in by_id:
                raise ValueError('Invalid action membership')
            owner[rid] = action
    if set(owner) != set(by_id):
        raise ValueError('Every live physical member needs one action')
    selected = set(running) & by_id.keys()
    evicted, decisions = set(), []
    def used(ids):
        return sum(charges[r] for r in ids)
    def fits(ids):
        return len(ids) <= slots and used(ids) <= capacity
    # Future horizon charges may exceed the actual resident allocation. Do not
    # force an oversized protected set through the physical admission budget.
    for rid in sorted(selected, key=lambda r:(owner[r]['density'], -by_id[r]['arrival'], r)):
        if fits(selected):
            break
        selected.remove(rid); evicted.add(rid)
        decisions.append({'reason': 'resident_horizon_pressure', 'victims': [rid]})
    candidates = sorted((a for a in actions if a['expected_credit'] > 0),
                        key=lambda a:(-a['density'], a['arrival'], a['id']))
    for candidate in candidates:
        need = set(candidate['members']) - selected
        if not need:
            continue
        if fits(selected | need):
            selected |= need
            continue
        trial = set(selected)
        victims = []
        pause = min(float(horizon), max(1., candidate['stage_work']))
        gain, loss = candidate['density'] * pause, 0.
        possible = sorted((rid for rid in selected if rid not in candidate['members']),
                          key=lambda rid:(owner[rid]['density'], -by_id[rid]['arrival'], rid))
        seen_actions = set()
        for rid in possible:
            old = owner[rid]
            if old['id'] in seen_actions:
                continue
            seen_actions.add(old['id'])
            remove = trial & set(old['members'])
            restore = sum((by_id[r]['prompt'] + by_id[r]['generated']) / prefill_unit
                          for r in remove)
            loss += interruption_loss(old, pause, unit, restore)
            trial -= remove; victims.extend(sorted(remove))
            if fits(trial | need):
                break
        if fits(trial | need) and gain > loss:
            selected = trial | need; evicted.update(victims)
            decisions.append({'reason': 'positive_net_credit_admission',
                              'action': candidate['id'], 'gain_proxy': gain,
                              'interruption_loss_proxy': loss, 'victims': victims})
    # Spare capacity is shared rather than left behind a probability gate.
    backfill = []
    for member in sorted(members, key=lambda m:(m['native_priority'], m['arrival'], m['request'])):
        rid = member['request']
        if rid in selected or rid in evicted:
            continue
        if fits(selected | {rid}):
            selected.add(rid); backfill.append(rid)
    if not selected and members:
        # An oversized context cannot be excluded forever by the forecast
        # horizon. Let the existing native block manager chunk/admit it safely.
        return None, {'reason':'native_oversized_fallback', 'decisions':decisions,
                      'backfill':[], 'evicted':[], 'selected_charge':0}
    return selected, {'reason':'resident_first_marginal_credit', 'decisions':decisions,
                      'backfill':backfill, 'evicted':sorted(set(running)-selected),
                      'selected_charge':used(selected)}
