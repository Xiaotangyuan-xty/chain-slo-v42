"""Shared admission decisions for the policy and its first-step projection."""


def admission_pauses(risk_work, deadlines, now, ready_keys, *,
                     defer_hopeless=False, defer_safe_later=False,
                     frame_reserve=0.0, recovery_costs=None):
    ready = set(ready_keys)
    attainable = {key for key in ready
                  if risk_work.get(key, float("inf")) <= deadlines[key] - now}
    if not attainable:
        return set()
    paused = ready - attainable if defer_hopeless else set()
    if defer_safe_later:
        earliest = min(deadlines[key] for key in attainable)
        recovery_costs = recovery_costs or {}
        for key in attainable:
            reserve = frame_reserve + recovery_costs.get(key, 0.0)
            if (deadlines[key] > earliest and
                    deadlines[key] - risk_work[key] >= earliest + reserve):
                paused.add(key)
    return paused
