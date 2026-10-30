"""Finite-anchor-lease rollout with atomic, slot-bounded compound batches."""
import math
import time


VERSION = "lease-slots-v1"


def project(jobs, action, now, perf, first_instance, deadline=math.inf, batch_slots=4):
    if not jobs:
        raise ValueError("empty goal denominator")
    if not isinstance(batch_slots, int) or batch_slots <= 0:
        raise ValueError("batch_slots must be a positive integer")
    anchor, horizon = action
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("lease must be positive and finite")
    if first_instance not in (0, 1):
        raise ValueError("unknown first instance")
    if len({job["key"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate job key")

    states = {}
    for job in jobs:
        slots = job.get("slots", 1)
        if not isinstance(slots, int) or not 0 < slots <= batch_slots:
            raise ValueError("compound cannot fit atomic request-slot boundary")
        if job["instance"] not in (0, 1):
            raise ValueError("unknown job instance")
        states[job["key"]] = {
            "job": job,
            "phases": list(job["phases"]),
            "idx": 0,
            "remaining": 0.0,
            "ready_at": 0.0,
            "ready_wait": job.get("ready_wait", 0.0),
        }

    t = 0.0
    busy = 0.0
    idle = 0.0
    compute_done = 0.0
    initial_compute = 0.0
    finished = {}
    timer_intervals = 0
    lease_runs = 0
    max_batch_slots = 0
    max_batch_goals = 0

    def transition(key, at):
        nonlocal timer_intervals
        state = states[key]
        if state["idx"] == len(state["phases"]):
            finished[key] = now + at
            return
        kind, value = state["phases"][state["idx"]]
        state["idx"] += 1
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid phase")
        if kind == "away":
            timer_intervals += int(value > 0)
            state["remaining"] = None
            state["ready_at"] = at + value
        elif kind == "compute":
            state["remaining"] = value
            state["ready_at"] = at
        else:
            raise ValueError("unknown phase kind")

    for key, state in states.items():
        for kind, value in state["phases"]:
            if kind == "compute":
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid compute phase")
                initial_compute += value
        transition(key, 0.0)

    def activate():
        changed = True
        while changed:
            changed = False
            for key, state in states.items():
                if key in finished:
                    continue
                if state["remaining"] is None and state["ready_at"] <= t + 1e-12:
                    transition(key, state["ready_at"])
                    changed = True
                elif state["remaining"] == 0.0 and state["ready_at"] <= t + 1e-12:
                    transition(key, state["ready_at"])
                    changed = True

    def batch(instance, lease_anchor=None):
        ready = [state for key, state in states.items()
                 if key not in finished and state["remaining"] is not None
                 and state["ready_at"] <= t + 1e-12
                 and state["job"]["instance"] == instance]
        if lease_anchor is not None:
            ready.sort(key=lambda state: (
                state["job"].get("level", 0),
                -state["ready_wait"],
                state["job"]["deadline"],
                state["job"]["key"],
            ))
        else:
            ready.sort(key=lambda state: (state["job"]["deadline"], state["job"]["key"]))
        by_key = {state["job"]["key"]: state for state in ready}
        if lease_anchor is not None:
            if lease_anchor not in by_key:
                return []
            ordered = [by_key[lease_anchor]] + [state for state in ready
                                                 if state["job"]["key"] != lease_anchor]
        else:
            ordered = ready
        chosen = []
        used = 0
        for state in ordered:
            slots = state["job"].get("slots", 1)
            if used + slots <= batch_slots:
                chosen.append(state["job"]["key"])
                used += slots
        return chosen

    def run(keys, lease=None):
        nonlocal t, busy, compute_done, lease_runs, max_batch_slots, max_batch_goals
        if not keys:
            raise ValueError("empty batch")
        slots = sum(states[key]["job"].get("slots", 1) for key in keys)
        if slots > batch_slots:
            raise RuntimeError("request-slot boundary exceeded")
        max_batch_slots = max(max_batch_slots, slots)
        max_batch_goals = max(max_batch_goals, len(keys))
        context = math.fsum(states[key]["job"]["context"] for key in keys)
        speed = perf.decode_seconds / perf.iteration(slots, context)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("invalid modeled speed")
        until_transition = min(states[key]["remaining"] / speed for key in keys)
        duration = until_transition if lease is None else min(until_transition, lease)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid batch duration")
        work = duration * speed
        t += duration
        busy += duration
        compute_done += work * len(keys)
        if lease is not None:
            lease_runs += 1
        for key, state in states.items():
            if key in finished or state["remaining"] is None:
                continue
            if state["ready_at"] <= t + 1e-12 and key not in keys:
                state["ready_wait"] += duration
        for key in keys:
            state = states[key]
            state["remaining"] -= work
            state["ready_wait"] = 0.0
            if state["remaining"] < 1e-9:
                state["remaining"] = 0.0
                transition(key, t)
        return duration

    activate()
    if anchor not in states or states[anchor]["job"]["state"] == "AWAY":
        raise ValueError("first action must target READY")
    if states[anchor]["job"]["instance"] != first_instance:
        raise ValueError("nonlocal immediate dispatch")
    if anchor not in batch(first_instance, anchor):
        raise ValueError("anchor is not immediately dispatchable")

    instance = first_instance
    lease_remaining = horizon
    lease_active = True
    events = 0
    while len(finished) < len(jobs):
        events += 1
        if events > 16384 or time.perf_counter() > deadline:
            raise TimeoutError("lease-aware reference budget")
        activate()
        if lease_active:
            anchor_state = states[anchor]
            if (lease_remaining <= 1e-12 or anchor in finished
                    or anchor_state["remaining"] is None
                    or anchor_state["ready_at"] > t + 1e-12):
                lease_active = False

        keys = []
        anchored = False
        if lease_active:
            selected_instance = instance
            selected_anchor = anchor if selected_instance == first_instance else None
            keys = batch(selected_instance, selected_anchor)
            if not keys:
                selected_instance = 1 - selected_instance
                selected_anchor = anchor if selected_instance == first_instance else None
                keys = batch(selected_instance, selected_anchor)
            anchored = anchor in keys
        else:
            selected_instance = instance
            keys = batch(selected_instance)
            if not keys:
                selected_instance = 1 - selected_instance
                keys = batch(selected_instance)

        if keys:
            consumed = run(keys, lease_remaining if anchored else None)
            if anchored:
                lease_remaining = max(0.0, lease_remaining - consumed)
            instance = 1 - selected_instance
            continue

        future = [state["ready_at"] for key, state in states.items()
                  if key not in finished and state["ready_at"] > t + 1e-12]
        if not future:
            raise RuntimeError("incomplete modeled denominator without a future event")
        next_event = min(future)
        idle += next_event - t
        t = next_event

    if len(finished) != len(jobs):
        raise RuntimeError("incomplete modeled denominator")
    if not math.isclose(compute_done, initial_compute, rel_tol=1e-9, abs_tol=1e-7):
        raise RuntimeError("modeled compute conservation")
    if not math.isclose(t, busy + idle, rel_tol=1e-9, abs_tol=1e-7):
        raise RuntimeError("modeled clock conservation")
    good = sum(finished[job["key"]] <= job["deadline"] for job in jobs)
    mean = math.fsum(finished.values()) / len(jobs) - now
    return {
        "good": good,
        "mean": mean,
        "denominator": len(jobs),
        "busy": busy,
        "idle": idle,
        "elapsed": t,
        "finishes": finished,
        "timer_intervals": timer_intervals,
        "lease_runs": lease_runs,
        "max_batch_slots": max_batch_slots,
        "max_batch_goals": max_batch_goals,
        "evaluation_version": VERSION,
    }
