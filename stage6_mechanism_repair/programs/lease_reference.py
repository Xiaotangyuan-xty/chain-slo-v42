import math
import time

from first_step import plan_first_step


VERSION = "execution-aligned-lease-v2-tardiness"


def compare_scores(left, right, tolerance=1e-9):
    if left["good"] != right["good"]:
        return 1 if left["good"] > right["good"] else -1
    for key, direction in (("tardiness", -1), ("mean", -1)):
        if not math.isclose(left[key], right[key], rel_tol=1e-12, abs_tol=tolerance):
            difference = left[key] - right[key]
            return direction if difference > 0 else -direction
    return 0


def first_plan_for_action(jobs, action, now, perf, first_instance,
                          batch_slots, resource):
    anchor_key, horizon = action
    current = [job for job in jobs if job.get("state") != "AWAY"
               and job["instance"] == first_instance]
    if anchor_key not in {job["key"] for job in current}:
        raise ValueError("anchor is not immediately dispatchable")
    decode_seconds = perf.decode_seconds
    default = min(current, key=lambda job: job.get("native_priority") or (
        job.get("level", 0), -job.get("ready_wait", 0.0), job["key"]))
    model_control = (anchor_key != default["key"]
                     or not math.isclose(horizon, 4 * decode_seconds,
                                         rel_tol=1e-12, abs_tol=1e-12))
    quantum = 64 * decode_seconds
    forced = max(current, key=lambda job: (job.get("ready_wait", 0.0), job["key"]))
    forced_key = forced["key"] if forced.get("ready_wait", 0.0) > 8 * quantum else None

    def priority(job):
        key = job["key"]
        rank = 0 if key == forced_key else 1 if key == anchor_key else 2 + job.get("level", 0)
        return rank, -job.get("ready_wait", 0.0), key

    protected = [job for job in current if job.get("running") and job.get("protected")]
    remaining = [job for job in current if job not in protected]
    ordered = protected + sorted(remaining, key=priority)
    units = []
    for job in ordered:
        required = job.get("kv_required")
        if required is None and job.get("resource_reason") is None:
            raise ValueError(f"missing first-step KV requirement for {job['key']}")
        running = [job] if job.get("running") else []
        units.append({
            "ids": [job["key"]],
            "slots": job.get("slots", 1),
            "tokens": job.get("tokens", 0),
            "required_blocks": required,
            "reason": job.get("resource_reason"),
            "running": running,
            "resident_blocks": job.get("resident_blocks", 0),
            "protected": job.get("protected", False),
            "state": job,
        })
    available = resource["free_blocks"] - resource["watermark_blocks"]
    plan = plan_first_step(units, available, batch_slots,
                           resource["max_swaps"], model_control,
                           resource.get("token_budget"))
    copy_seconds = getattr(perf, "copy_seconds_per_token", 0.0)
    evicted_states = [job for unit, running, _ in plan["evictions"] for job in running]
    switch_cost = math.fsum(job.get("gpu_tokens", 0) for job in evicted_states) * copy_seconds
    return {
        "plan": plan,
        "selected": [entry["state"] for entry in plan["accepted"]],
        "used_slots": plan["used_slots"],
        "evicted_states": evicted_states,
        "switch_cost": switch_cost,
        "model_control": model_control,
        "forced_key": forced_key,
    }


def project(jobs, action, now, perf, first_instance, deadline=math.inf,
            batch_slots=4, trace_limit=16, resource=None):
    if not jobs:
        raise ValueError("empty goal denominator")
    if not isinstance(batch_slots, int) or batch_slots <= 0:
        raise ValueError("invalid batch slot limit")
    anchor_key, horizon = action
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("invalid lease")
    if first_instance not in (0, 1):
        raise ValueError("invalid instance")
    if len({job["key"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate goal key")

    states = {}
    for job in jobs:
        slots = job.get("slots", 1)
        if not isinstance(slots, int) or not 0 < slots <= batch_slots:
            raise ValueError("compound cannot fit atomic request-slot boundary")
        states[job["key"]] = {
            "job": job,
            "phases": list(job["phases"]),
            "idx": 0,
            "remaining": 0.0,
            "ready_at": 0.0,
            "ready_wait": job.get("ready_wait", 0.0),
            "virtual": job.get("virtual", 0.0),
            "level": job.get("level", 0),
            "quantum": job.get("quantum", 0.0),
            "context": job["context"],
            "running": job.get("running", False),
        }
    t = 0.0
    busy = 0.0
    idle = 0.0
    compute_done = 0.0
    initial_compute = math.fsum(value for job in jobs for kind, value in job["phases"]
                                if kind == "compute")
    finished = {}
    timer_intervals = 0
    max_batch_slots = 0
    batch_trace = []

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
            state["remaining"] = None
            state["running"] = False
            state["ready_at"] = at + value
            timer_intervals += int(value > 0)
        elif kind == "compute":
            state["remaining"] = value
            state["ready_at"] = at
        else:
            raise ValueError("unknown phase")

    for key in states:
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

    def ready(instance):
        return [state for key, state in states.items()
                if key not in finished and state["remaining"] is not None
                and state["ready_at"] <= t + 1e-12
                and state["job"]["instance"] == instance]

    activate()
    if anchor_key not in states or states[anchor_key]["job"]["state"] == "AWAY":
        raise ValueError("first action must target READY")
    if states[anchor_key]["job"]["instance"] != first_instance:
        raise ValueError("nonlocal immediate dispatch")
    if not any(state["job"]["key"] == anchor_key for state in ready(first_instance)):
        raise ValueError("anchor is not immediately dispatchable")

    decode_seconds = perf.decode_seconds
    initial_default = min(ready(first_instance), key=lambda state: state["job"].get(
        "native_priority") or (state["level"], -state["ready_wait"], state["job"]["key"]))
    initial_model = (anchor_key != initial_default["job"]["key"]
                     or not math.isclose(horizon, 4 * decode_seconds,
                                         rel_tol=1e-12, abs_tol=1e-12))
    anchors = {0: [None, 0.0, False], 1: [None, 0.0, False]}
    anchors[first_instance] = [anchor_key, horizon, initial_model]
    instance = first_instance
    steps = 0
    while len(finished) < len(jobs):
        steps += 1
        if steps > 65536 or time.perf_counter() > deadline:
            raise TimeoutError("execution-aligned reference budget")
        activate()
        current = ready(instance)
        other = ready(1 - instance)
        if not current and other:
            instance = 1 - instance
            current = other
        if not current:
            future = [state["ready_at"] for key, state in states.items()
                      if key not in finished and state["ready_at"] > t + 1e-12]
            if not future:
                raise RuntimeError("incomplete denominator without future event")
            next_event = min(future)
            idle += next_event - t
            t = next_event
            continue

        quantum = 64 * decode_seconds
        floor = min(state["virtual"] for state in current)
        eligible = [state for state in current if state["virtual"] <= floor + 2 * quantum]
        if not eligible:
            raise RuntimeError("empty eligible set")
        forced = max(eligible, key=lambda state: (state["ready_wait"], state["job"]["key"]))
        forced_key = forced["job"]["key"] if forced["ready_wait"] > 8 * quantum else None
        anchor, lease, model_control = anchors[instance]
        eligible_keys = {state["job"]["key"] for state in eligible}
        if anchor not in eligible_keys or lease <= 1e-12:
            default = min(eligible, key=lambda state: state["job"].get(
                "native_priority") or (state["level"], -state["ready_wait"], state["job"]["key"]))
            anchor = default["job"]["key"]
            lease = 4 * decode_seconds
            model_control = False
        model_control = model_control or forced_key is not None
        anchors[instance] = [anchor, lease, model_control]

        def priority(state):
            key = state["job"]["key"]
            rank = 0 if key == forced_key else 1 if key == anchor else 2 + state["level"]
            return rank, -state["ready_wait"], key

        protected = [state for state in current if state["running"] and state["job"].get("protected")]
        remaining = [state for state in current if state not in protected]
        ordered = protected + sorted(remaining, key=priority)
        if not model_control:
            running = sorted((state for state in ordered if state["running"]),
                             key=lambda state: state["job"].get("queue_rank", len(jobs)))
            waiting = [state for state in ordered if not state["running"]]
            ordered = running + waiting
        first_plan = None
        if steps == 1 and resource is not None:
            first_plan = first_plan_for_action(
                [state["job"] for state in current], (anchor_key, horizon), now,
                perf, instance, batch_slots, resource)
            selected = [states[entry["key"]] for entry in first_plan["selected"]]
            used = first_plan["used_slots"]
            for state in first_plan["evicted_states"]:
                states[state["key"]]["running"] = False
        else:
            selected = []
            used = 0
            for state in ordered:
                slots = state["job"].get("slots", 1)
                if used + slots <= batch_slots:
                    selected.append(state)
                    used += slots
        if not selected:
            raise RuntimeError("no atomic batch fits")
        selected_keys = {state["job"]["key"] for state in selected}
        if model_control:
            for state in current:
                if state["job"]["key"] not in selected_keys and not state["job"].get("protected"):
                    state["running"] = False
        for state in selected:
            state["running"] = True

        context = math.fsum(state["context"] for state in selected)
        duration = perf.iteration(used, context)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid iteration cost")
        t += duration
        busy += duration
        max_batch_slots = max(max_batch_slots, used)
        selected_keys = {state["job"]["key"] for state in selected}
        for state in states.values():
            if state["remaining"] is not None and state["ready_at"] <= t + 1e-12:
                if state["job"]["key"] not in selected_keys:
                    state["ready_wait"] += duration
        for state in selected:
            key = state["job"]["key"]
            work = min(decode_seconds, state["remaining"])
            state["remaining"] -= work
            compute_done += work
            state["ready_wait"] = 0.0
            state["virtual"] += state["job"].get("slots", 1) * duration / used
            state["quantum"] += state["job"].get("slots", 1) * decode_seconds
            state["context"] += state["job"].get("slots", 1)
            if state["quantum"] >= quantum * (2 ** state["level"]):
                state["level"] = min(3, state["level"] + 1)
                state["quantum"] = 0.0
            if state["remaining"] < 1e-10:
                state["remaining"] = 0.0
                transition(key, t)
        if anchor in selected_keys:
            anchors[instance][1] = max(0.0, anchors[instance][1] - decode_seconds)
        if len(batch_trace) < trace_limit:
            batch_trace.append({
                "step": steps,
                "instance": instance,
                "keys": [state["job"]["key"] for state in selected],
                "slots": used,
                "duration": duration,
                "anchor": anchor,
                "lease_after": anchors[instance][1],
                "forced": forced_key,
                "switch_cost": first_plan["switch_cost"] if first_plan else 0.0,
                "first_deferred": first_plan["plan"]["deferred"] if first_plan else [],
            })
        for index, state in enumerate(selected):
            state["job"]["queue_rank"] = index
        for state in current:
            if state["running"] and state not in selected:
                state["job"]["queue_rank"] = len(selected) + state["job"].get("queue_rank", 0)
        instance = 1 - instance

    if not math.isclose(compute_done, initial_compute, rel_tol=1e-9, abs_tol=1e-7):
        raise RuntimeError("compute conservation")
    if not math.isclose(t, busy + idle, rel_tol=1e-9, abs_tol=1e-7):
        raise RuntimeError("clock conservation")
    good = sum(finished[job["key"]] <= job["deadline"] for job in jobs)
    tardiness = math.fsum(max(0.0, finished[job["key"]] - job["deadline"])
                          for job in jobs)
    mean = math.fsum(finished.values()) / len(jobs) - now
    return {
        "good": good,
        "tardiness": tardiness,
        "mean": mean,
        "denominator": len(jobs),
        "busy": busy,
        "idle": idle,
        "elapsed": t,
        "finishes": finished,
        "timer_intervals": timer_intervals,
        "max_batch_slots": max_batch_slots,
        "switch_cost": batch_trace[0]["switch_cost"] if batch_trace else 0.0,
        "first_deferred": batch_trace[0]["first_deferred"] if batch_trace else [],
        "batch_trace": batch_trace,
        "final_virtual": {key: state["virtual"] for key, state in states.items()},
        "evaluation_version": VERSION,
    }
