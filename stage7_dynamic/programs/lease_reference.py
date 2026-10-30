from collections.abc import Mapping
import math
import time

from conditional_predictor import safe_prefill
from first_step import plan_first_step


VERSION = "execution-aligned-lease-v3-dynamic-resources"


def resolve_batch_slots(batch_slots, step=1, instance=0):
    value = batch_slots
    if callable(value):
        value = value(step, instance)
    elif isinstance(value, Mapping):
        value = value.get((step, instance), value.get(step, value.get(
            instance, value.get("default"))))
    elif not isinstance(value, int):
        values = tuple(value)
        if not values:
            raise ValueError("empty dynamic batch slot schedule")
        value = values[min(step - 1, len(values) - 1)]
    if not isinstance(value, int) or value <= 0:
        raise ValueError("invalid dynamic batch slot limit")
    return value


def phase_parts(phase):
    if isinstance(phase, Mapping):
        return phase["kind"], float(phase.get("seconds", phase.get("value", 0.0)))
    kind, value = phase
    return kind, float(value)


def _member_progress(value):
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): max(0, int(progress)) for key, progress in value.items()}
    result = {}
    for index, item in enumerate(value):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            key, progress = item
        else:
            key, progress = str(index), item
        result[str(key)] = max(0, int(progress))
    return result


def _default_priority(state, use_native=False):
    job = state["job"]
    native = job.get("native_priority")
    if use_native and native is not None:
        return native
    return state["level"], -state["ready_wait"], job["key"]


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
    forced_key = (forced["key"] if forced.get("custom_forced_enabled", True)
                  and forced.get("ready_wait", 0.0) > 8 * quantum else None)

    def priority(job):
        key = job["key"]
        rank = (0 if key == forced_key else 1 if key == anchor_key else
                2 if job.get("deadline_attainable", True) else 3)
        native = job.get("native_priority")
        order = native if native is not None else (
            job.get("level", 0), -job.get("ready_wait", 0.0), key)
        return rank, order, key

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
            "min_tokens": min(job.get("slots", 1), job.get("tokens", 0)),
            "required_blocks": required,
            "reason": ("chain_deadline_deferred" if job.get("admission_paused")
                       else job.get("resource_reason")),
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
            batch_slots=4, trace_limit=16, resource=None, evaluation_cost=0.0,
            max_steps=65536):
    if not jobs:
        raise ValueError("empty goal denominator")
    if not isinstance(batch_slots, int) and not callable(batch_slots) and not isinstance(
            batch_slots, Mapping):
        batch_slots = tuple(batch_slots)
    resolve_batch_slots(batch_slots)
    anchor_key, horizon = action
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("invalid lease")
    if not math.isfinite(evaluation_cost) or evaluation_cost < 0:
        raise ValueError("invalid synchronous evaluation cost")
    if first_instance not in (0, 1):
        raise ValueError("invalid instance")
    if len({job["key"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate goal key")

    resource = dict(resource) if resource is not None else None
    block_size = resource.get("block_size", 16) if resource is not None else 16
    free_blocks = resource["free_blocks"] if resource is not None else 0
    watermark = resource.get("watermark_blocks", 0) if resource is not None else 0
    host_free_tokens = (resource["host_free_blocks"] * block_size
                        if resource is not None and "host_free_blocks" in resource else math.inf)
    copy_seconds = getattr(perf, "copy_seconds_per_token", 0.0)
    states = {}
    for original in jobs:
        job = dict(original)
        slots = job.get("slots", 1)
        if not isinstance(slots, int) or slots <= 0:
            raise ValueError("invalid compound width")
        states[job["key"]] = {
            "job": job,
            "phases": list(job["phases"]),
            "metadata": list(job.get("phase_metadata", ())),
            "idx": 0,
            "remaining": 0.0,
            "ready_at": 0.0,
            "ready_wait": job.get("ready_wait", 0.0),
            "virtual": job.get("virtual", 0.0),
            "level": job.get("level", 0),
            "quantum": job.get("quantum", 0.0),
            "context": job["context"],
            "running": job.get("running", False),
            "slots": slots,
            "gpu_tokens": job.get("gpu_tokens", 0),
            "host_tokens": job.get("host_tokens", 0),
            "resident_blocks": job.get("resident_blocks", 0),
            "queue_rank": job.get("queue_rank", len(jobs)),
            "member_progress": _member_progress(job.get("member_progress")),
            "decode_progress": job.get("decode_progress", 0),
            "prefill_remaining": 0,
            "prefill_seconds_remaining": 0.0,
            "recompute_required": False,
            "deferred_copy": 0.0,
            "ever_selected": False,
        }

    t = evaluation_cost
    busy = 0.0
    idle = 0.0
    switch_cost = 0.0
    resume_cost = 0.0
    compute_done = 0.0
    finished = {}
    timer_intervals = 0
    max_batch_slots = 0
    batch_trace = []
    kv_trace = []
    deferred_copy_total = 0.0
    recompute_extra = 0.0

    def transition(key, at, initial=False):
        nonlocal timer_intervals, deferred_copy_total
        state = states[key]
        if state["idx"] == len(state["phases"]):
            finished[key] = now + at
            return
        phase_index = state["idx"]
        phase = state["phases"][phase_index]
        kind, value = phase_parts(phase)
        state["idx"] += 1
        metadata = dict(phase) if isinstance(phase, Mapping) else {}
        if phase_index < len(state["metadata"]):
            metadata.update(state["metadata"][phase_index])
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid phase")
        if kind == "away":
            if not initial and (state["resident_blocks"] or state["host_tokens"]):
                release(key)
            state["remaining"] = None
            state["running"] = False
            state["ready_at"] = at + value
            state["prefill_remaining"] = 0
            state["prefill_seconds_remaining"] = 0.0
            timer_intervals += int(value > 0)
        elif kind == "compute":
            if initial and resource is not None:
                state["deferred_copy"] = min(value, state["job"].get("recovery_copy", 0.0))
                value -= state["deferred_copy"]
                deferred_copy_total += state["deferred_copy"]
            state["remaining"] = value
            state["ready_at"] = at
            state["slots"] = max(1, int(round(metadata.get(
                "width", state["job"].get("slots", 1)))))
            prefill_tokens = metadata.get("prefill_tokens", 0)
            if initial and not metadata:
                prefill_tokens = state["job"].get("prefill_tokens", 0)
            state["prefill_remaining"] = max(0, int(round(prefill_tokens)))
            state["prefill_seconds_remaining"] = min(value, max(
                0.0, metadata.get("prefill_seconds", state["job"].get(
                    "prefill_seconds", 0.0) if initial else 0.0)))
            if not initial:
                state["decode_progress"] = 0
                if len(state["member_progress"]) == state["slots"]:
                    state["member_progress"] = {key: 0 for key in state["member_progress"]}
                elif state["member_progress"]:
                    state["member_progress"] = {f"future:{index}": 0
                                                for index in range(state["slots"])}
                if metadata.get("prompt_tokens", 0):
                    state["context"] = int(round(metadata["prompt_tokens"]))
        else:
            raise ValueError("unknown phase")

    for key in states:
        transition(key, 0.0, initial=True)
    initial_compute = math.fsum(
        value for job in jobs for phase in job["phases"]
        for kind, value in [phase_parts(phase)] if kind == "compute") - deferred_copy_total

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

    def ready(owner):
        return [state for key, state in states.items()
                if key not in finished and state["remaining"] is not None
                and state["ready_at"] <= t + 1e-12
                and state["job"]["instance"] == owner]

    def release(key):
        nonlocal free_blocks, host_free_tokens
        state = states[key]
        free_blocks += state["resident_blocks"]
        host_free_tokens += state["host_tokens"]
        state["resident_blocks"] = 0
        state["gpu_tokens"] = 0
        state["host_tokens"] = 0
        state["running"] = False

    activate()
    if anchor_key not in states or states[anchor_key]["job"]["state"] == "AWAY":
        raise ValueError("first action must target READY")
    if states[anchor_key]["job"]["instance"] != first_instance:
        raise ValueError("nonlocal immediate dispatch")
    if not any(state["job"]["key"] == anchor_key for state in ready(first_instance)):
        raise ValueError("anchor is not immediately dispatchable")

    decode_seconds = perf.decode_seconds
    initial_default = min(ready(first_instance), key=lambda state: _default_priority(state, True))
    initial_model = (anchor_key != initial_default["job"]["key"]
                     or not math.isclose(horizon, 4 * decode_seconds,
                                         rel_tol=1e-12, abs_tol=1e-12))
    anchors = {0: [None, 0.0, False], 1: [None, 0.0, False]}
    anchors[first_instance] = [anchor_key, horizon, initial_model]
    instance = first_instance
    steps = 0
    first_deferred = []
    while len(finished) < len(jobs):
        steps += 1
        if steps > max_steps or time.perf_counter() > deadline:
            raise TimeoutError("execution-aligned reference budget")
        activate()
        for key in list(finished):
            if states[key]["resident_blocks"] or states[key]["host_tokens"]:
                release(key)
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
        forced = max(eligible, key=lambda state: (state["ready_wait"], state["job"]["key"]))
        forced_key = (forced["job"]["key"] if forced["job"].get("custom_forced_enabled", True)
                      and forced["ready_wait"] > 8 * quantum else None)
        anchor, lease, model_control = anchors[instance]
        eligible_keys = {state["job"]["key"] for state in eligible}
        if anchor not in eligible_keys or lease <= 1e-12:
            default = min(eligible, key=lambda state: _default_priority(state, steps == 1))
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
        ordered = protected + sorted((state for state in current if state not in protected),
                                     key=priority)
        if not model_control:
            running = sorted((state for state in ordered if state["running"]),
                             key=lambda state: state["queue_rank"])
            ordered = running + [state for state in ordered if not state["running"]]
        slot_limit = resolve_batch_slots(batch_slots, steps, instance)
        token_budget = resource.get("token_budget") if resource is not None else None
        step_switch = 0.0
        step_resume = 0.0
        first_plan = None

        def scheduled_tokens(state):
            return state["prefill_remaining"] or state["slots"]

        def make_unit(state):
            nonlocal recompute_extra
            if state["recompute_required"]:
                state["recompute_required"] = False
                tokens_to_recompute = max(1, int(state["context"]))
                seconds = safe_prefill(perf, tokens_to_recompute)
                state["prefill_remaining"] = tokens_to_recompute
                state["prefill_seconds_remaining"] = seconds
                state["remaining"] += seconds
                recompute_extra += seconds
            first = steps == 1
            tokens = scheduled_tokens(state)
            if first and "tokens" in state["job"] and (
                    state["job"]["tokens"] or not state["prefill_remaining"]):
                tokens = state["job"]["tokens"]
            required = state["job"].get("kv_required") if first else None
            if required is None:
                prefix = max(state["gpu_tokens"], state["host_tokens"])
                total_blocks = math.ceil((prefix + tokens) / block_size)
                required = max(0, total_blocks - state["resident_blocks"])
            return {
                "ids": [state["job"]["key"]], "slots": state["slots"], "tokens": tokens,
                "min_tokens": 1 if state["prefill_remaining"] else state["slots"],
                "required_blocks": required,
                "reason": state["job"].get("resource_reason") if first else None,
                "running": [state["job"]] if state["running"] else [],
                "resident_blocks": state["resident_blocks"],
                "protected": state["job"].get("protected", False), "state": state,
            }

        def swap_state(state, count_free):
            nonlocal free_blocks, host_free_tokens
            gpu_tokens = state["gpu_tokens"]
            old_host = state["host_tokens"]
            moved = max(0, gpu_tokens - old_host)
            if moved > host_free_tokens:
                host_free_tokens += old_host
                state["recompute_required"] = True
                state["host_tokens"] = 0
                state["gpu_tokens"] = 0
                if count_free:
                    free_blocks += state["resident_blocks"]
                state["resident_blocks"] = 0
                state["running"] = False
                return 0.0
            host_free_tokens -= moved
            state["host_tokens"] = old_host + moved
            state["gpu_tokens"] = 0
            if count_free:
                free_blocks += state["resident_blocks"]
            state["resident_blocks"] = 0
            state["running"] = False
            return moved * copy_seconds

        def evict_inactive(current_keys):
            nonlocal step_switch
            candidates = [state for key, state in states.items()
                          if key not in current_keys and key not in finished
                          and state["resident_blocks"] > 0
                          and not state["job"].get("protected", False)]
            if not candidates:
                return False
            state = min(candidates, key=lambda item: item["job"]["key"])
            moved = swap_state(state, True)
            if moved is None:
                return False
            step_switch += moved
            return True

        units = []
        selected = []
        if resource is not None:
            while True:
                units = [make_unit(state) for state in ordered]
                current_resource = dict(resource, free_blocks=free_blocks)
                if steps == 1:
                    first_plan = first_plan_for_action(
                        [state["job"] for state in current], (anchor_key, horizon), now,
                        perf, instance, slot_limit, current_resource)
                    plan = first_plan["plan"]
                else:
                    plan = plan_first_step(units, free_blocks - watermark, slot_limit,
                                           resource.get("max_swaps", 3), model_control,
                                           token_budget)
                if plan["accepted"] or not evict_inactive({state["job"]["key"] for state in current}):
                    break
            for unit, running, _ in plan["evictions"]:
                moved = swap_state(states[unit["ids"][0]], False)
                if moved is None:
                    raise RuntimeError("host capacity prevents planned swap")
                step_switch += moved
            selected = [states[unit["ids"][0]] for unit in plan["accepted"]]
            units = plan["accepted"]
            free_blocks = plan["remaining_available_blocks"] + watermark
            first_deferred = plan["deferred"] if steps == 1 else first_deferred
        else:
            used_slots = 0
            for state in ordered:
                if used_slots + state["slots"] <= slot_limit:
                    selected.append(state)
                    used_slots += state["slots"]
            units = [make_unit(state) for state in selected]
        if not selected:
            raise RuntimeError("no atomic batch fits current slot/token/KV budgets")

        selected_keys = {state["job"]["key"] for state in selected}
        if model_control:
            for state in current:
                if state["job"]["key"] not in selected_keys and not state["job"].get("protected"):
                    moved = swap_state(state, True)
                    if moved is None:
                        raise RuntimeError("host capacity prevents model eviction")
                    step_switch += moved
        prefill_work = {}
        for state, unit in zip(selected, units):
            required = unit["required_blocks"]
            state["resident_blocks"] += required
            if state["host_tokens"] > state["gpu_tokens"]:
                moved = state["host_tokens"] - state["gpu_tokens"]
                state["gpu_tokens"] += moved
                state["host_tokens"] -= moved
                host_free_tokens += moved
                step_resume += moved * copy_seconds
            state["running"] = True
            state["ever_selected"] = True
            chunk = min(unit["tokens"], state["prefill_remaining"])
            if chunk:
                portion = state["prefill_seconds_remaining"] * chunk / state["prefill_remaining"]
                prefill_work[state["job"]["key"]] = (chunk, portion)

        context = math.fsum(state["context"] for state in selected)
        iteration = perf.iteration(sum(state["slots"] for state in selected), context)
        duration = max([iteration] + [portion for _, portion in prefill_work.values()])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid iteration cost")
        started = t
        t += step_switch + step_resume + duration
        switch_cost += step_switch
        resume_cost += step_resume
        busy += duration
        used = sum(state["slots"] for state in selected)
        max_batch_slots = max(max_batch_slots, used)
        span = t - started
        for state in states.values():
            if (state["remaining"] is not None and state["ready_at"] <= started + 1e-12
                    and state["job"]["key"] not in selected_keys):
                state["ready_wait"] += span
        for state in selected:
            key = state["job"]["key"]
            if key in prefill_work:
                chunk, portion = prefill_work[key]
                work = min(portion, state["remaining"])
                state["prefill_remaining"] -= chunk
                state["prefill_seconds_remaining"] -= portion
                state["gpu_tokens"] += chunk
            else:
                work = min(decode_seconds, state["remaining"])
                state["context"] += state["slots"]
                state["gpu_tokens"] += state["slots"]
                state["decode_progress"] += 1
                for member in state["member_progress"]:
                    state["member_progress"][member] += 1
            state["remaining"] -= work
            compute_done += work
            state["ready_wait"] = 0.0
            state["virtual"] += state["slots"] * duration / used
            state["quantum"] += state["slots"] * decode_seconds
            if state["quantum"] >= quantum * (2 ** state["level"]):
                state["level"] = min(3, state["level"] + 1)
                state["quantum"] = 0.0
            if state["remaining"] < 1e-10 and not state["prefill_remaining"]:
                state["remaining"] = 0.0
                transition(key, t)
        if anchor in selected_keys:
            anchors[instance][1] = max(0.0, anchors[instance][1] - decode_seconds)
        trace = {
            "step": steps, "instance": instance,
            "keys": [state["job"]["key"] for state in selected], "slots": used,
            "duration": duration, "anchor": anchor,
            "lease_after": anchors[instance][1], "forced": forced_key,
            "tokens": sum(unit["tokens"] for unit in units),
            "token_budget": token_budget, "slot_limit": slot_limit,
            "switch_cost": step_switch, "resume_cost": step_resume,
            "evaluation_cost": evaluation_cost if steps == 1 else 0.0,
            "free_blocks": free_blocks,
            "gpu_tokens": {state["job"]["key"]: state["gpu_tokens"] for state in selected},
            "host_tokens": {state["job"]["key"]: state["host_tokens"] for state in selected},
            "decode_progress": {state["job"]["key"]: state["decode_progress"] for state in selected},
            "first_deferred": plan["deferred"] if steps == 1 and resource is not None else [],
        }
        if len(batch_trace) < trace_limit:
            batch_trace.append(trace)
        if len(kv_trace) < trace_limit:
            kv_trace.append({"step": steps, "free_blocks": free_blocks,
                             "host_free_tokens": (host_free_tokens if math.isfinite(host_free_tokens)
                                                  else None),
                             "residency": {key: {"gpu": state["gpu_tokens"],
                                                "host": state["host_tokens"],
                                                "blocks": state["resident_blocks"]}
                                           for key, state in states.items()}})
        for index, state in enumerate(selected):
            state["queue_rank"] = index
        for state in current:
            if state["running"] and state not in selected:
                state["queue_rank"] = len(selected) + state["queue_rank"]
        instance = 1 - instance

    for key in list(finished):
        if states[key]["resident_blocks"] or states[key]["host_tokens"]:
            release(key)
    if not math.isclose(compute_done, initial_compute + recompute_extra,
                        rel_tol=1e-9, abs_tol=1e-7):
        raise RuntimeError(
            f"compute conservation: {compute_done} != {initial_compute + recompute_extra}")
    if not math.isclose(t, busy + idle + switch_cost + resume_cost + evaluation_cost,
                        rel_tol=1e-9, abs_tol=1e-7):
        raise RuntimeError("clock conservation")
    good = sum(finished[job["key"]] <= job["deadline"] for job in jobs)
    tardiness = math.fsum(max(0.0, finished[job["key"]] - job["deadline"])
                          for job in jobs)
    mean = math.fsum(finished.values()) / len(jobs) - now
    return {
        "good": good, "tardiness": tardiness, "mean": mean,
        "denominator": len(jobs), "busy": busy, "idle": idle, "elapsed": t,
        "finishes": finished, "timer_intervals": timer_intervals,
        "max_batch_slots": max_batch_slots, "switch_cost": switch_cost,
        "resume_cost": resume_cost, "evaluation_cost": evaluation_cost,
        "recompute_seconds": recompute_extra,
        "costs_in_finishes": True, "first_deferred": first_deferred,
        "batch_trace": batch_trace, "kv_trace": kv_trace,
        "final_virtual": {key: state["virtual"] for key, state in states.items()},
        "final_progress": {key: {"decode": state["decode_progress"],
                                 "members": dict(state["member_progress"])}
                           for key, state in states.items()},
        "final_free_blocks": free_blocks,
        "evaluation_version": VERSION,
    }
