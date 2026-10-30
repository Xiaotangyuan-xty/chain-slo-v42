def plan_first_step(units, available_blocks, batch_slots, max_swaps,
                    policy_preemption, token_budget=None):
    accepted = []
    accepted_by_index = {}
    evictions = []
    evicted = set()
    deferred_by_index = {}
    accounts = {}
    available = int(available_blocks)
    swaps_remaining = max(0, int(max_swaps))
    used_slots = 0
    used_tokens = 0
    used_blocks = 0
    planned_freed = 0

    def admission(unit, slots, tokens, required, slot_used, token_used,
                  block_used, free_blocks):
        if slots > batch_slots:
            return None, "compound_slot_limit"
        if slot_used + slots > batch_slots:
            return None, "capacity"
        planned = tokens
        if token_budget is not None:
            planned = min(tokens, max(0, token_budget - token_used))
        minimum = unit.get("min_tokens", tokens)
        if planned < minimum:
            return None, "token_budget"
        if required is None:
            return None, unit.get("reason") or "resource"
        if required > free_blocks:
            return None, "kv"
        return planned, None

    for index, original in enumerate(units):
        if index in evicted:
            continue
        unit = dict(original)
        ids = unit["ids"]
        slots = int(unit["slots"])
        requested_tokens = int(unit.get("tokens", 0))
        required = unit.get("required_blocks")
        running = list(unit.get("running", []))
        reason = unit.get("reason")
        protected = unit.get("protected", False)
        if reason is None:
            planned_tokens, reason = admission(
                unit, slots, requested_tokens, required, used_slots,
                used_tokens, used_blocks, available)
        else:
            planned_tokens = None

        if reason is None:
            unit.update(requested_tokens=requested_tokens,
                        tokens=planned_tokens)
            accepted.append(unit)
            accepted_by_index[index] = unit
            used_slots += slots
            used_tokens += planned_tokens
            used_blocks += required
            available -= required
            accounts[index] = {
                "ids": ids, "required_blocks": required,
                "requested_tokens": requested_tokens,
                "planned_tokens": planned_tokens,
                "disposition": "accepted", "available_after": available,
            }
            continue

        if running and protected:
            deferred_by_index[index] = (ids, reason, "residency_guard")
            accounts[index] = {
                "ids": ids, "required_blocks": required,
                "requested_tokens": requested_tokens,
                "disposition": "residency_guard", "reason": reason,
                "available_after": available,
            }
            continue

        allowed = (original.get("reason") is None
                   and (policy_preemption
                        or reason not in ("capacity", "no_tokens", "token_budget")))
        if not allowed:
            deferred_by_index[index] = (ids, reason, "waiting_or_swapped")
            accounts[index] = {
                "ids": ids, "required_blocks": required,
                "requested_tokens": requested_tokens,
                "disposition": "waiting_or_swapped", "reason": reason,
                "available_after": available,
            }
            continue

        candidate_victims = []
        trial_available = available
        trial_slots = used_slots
        trial_tokens = used_tokens
        trial_blocks = used_blocks
        trial_swaps = swaps_remaining
        trial_freed = 0
        trial_planned = None
        trial_reason = reason
        for victim_index in range(len(units) - 1, index, -1):
            if victim_index in evicted:
                continue
            victim = units[victim_index]
            victim_running = list(victim.get("running", []))
            if not victim_running or victim.get("protected", False):
                continue
            if len(victim_running) > trial_swaps:
                continue
            candidate_victims.append(victim_index)
            trial_swaps -= len(victim_running)
            resident = int(victim.get("resident_blocks", 0))
            trial_available += resident
            trial_freed += resident
            previous = accepted_by_index.get(victim_index)
            if previous is not None:
                trial_slots -= int(previous["slots"])
                trial_tokens -= int(previous["tokens"])
                trial_blocks -= int(previous["required_blocks"])
            trial_planned, trial_reason = admission(
                unit, slots, requested_tokens, required, trial_slots,
                trial_tokens, trial_blocks, trial_available)
            if trial_reason is None:
                break

        if trial_reason is not None:
            deferred_by_index[index] = (ids, trial_reason, "waiting_or_swapped")
            accounts[index] = {
                "ids": ids, "required_blocks": required,
                "requested_tokens": requested_tokens,
                "disposition": "waiting_or_swapped", "reason": trial_reason,
                "available_after": available,
            }
            continue

        for victim_index in candidate_victims:
            victim = units[victim_index]
            victim_running = list(victim.get("running", []))
            resident = int(victim.get("resident_blocks", 0))
            previous = accepted_by_index.pop(victim_index, None)
            if previous is not None:
                accepted.remove(previous)
                used_slots -= int(previous["slots"])
                used_tokens -= int(previous["tokens"])
                used_blocks -= int(previous["required_blocks"])
            available += resident
            planned_freed += resident
            swaps_remaining -= len(victim_running)
            evicted.add(victim_index)
            evictions.append((victim, victim_running, reason))
            accounts[victim_index] = {
                "ids": victim["ids"],
                "required_blocks": victim.get("required_blocks"),
                "disposition": "evict", "resident_freed": resident,
                "trigger": reason, "available_after": available,
            }

        planned_tokens, committed_reason = admission(
            unit, slots, requested_tokens, required, used_slots,
            used_tokens, used_blocks, available)
        if committed_reason is not None:
            raise RuntimeError("first-step eviction commit was not conservative")
        unit.update(requested_tokens=requested_tokens, tokens=planned_tokens)
        accepted.append(unit)
        accepted_by_index[index] = unit
        used_slots += slots
        used_tokens += planned_tokens
        used_blocks += required
        available -= required
        accounts[index] = {
            "ids": ids, "required_blocks": required,
            "requested_tokens": requested_tokens,
            "planned_tokens": planned_tokens,
            "disposition": "accepted", "available_after": available,
            "evictions": [units[victim]["ids"] for victim in candidate_victims],
        }

    leftover_running = []
    deferred = []
    for index, unit in enumerate(units):
        if index in evicted or index in accepted_by_index:
            continue
        running = list(unit.get("running", []))
        if running:
            leftover_running.extend(running)
        if index in deferred_by_index:
            deferred.append(deferred_by_index[index])
        else:
            deferred.append((unit["ids"], unit.get("reason") or "resource",
                             "retained" if running else "waiting_or_swapped"))

    return {
        "accepted": accepted,
        "evictions": evictions,
        "deferred": deferred,
        "leftover_running": leftover_running,
        "unit_accounts": [accounts.get(index, {
            "ids": unit["ids"], "disposition": "evict" if index in evicted else "deferred"
        }) for index, unit in enumerate(units)],
        "used_slots": used_slots,
        "used_blocks": used_blocks,
        "used_tokens": used_tokens,
        "planned_freed_blocks": planned_freed,
        "remaining_available_blocks": available,
        "swaps_remaining": swaps_remaining,
    }
