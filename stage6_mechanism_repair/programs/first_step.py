def plan_first_step(units, available_blocks, batch_slots, max_swaps,
                    policy_preemption, token_budget=None):
    accepted = []
    evictions = []
    deferred = []
    leftover_running = []
    unit_accounts = []
    used_slots = 0
    used_blocks = 0
    used_tokens = 0
    planned_freed = 0
    swaps_remaining = max(0, int(max_swaps))
    available = available_blocks

    for unit in units:
        ids = unit["ids"]
        slots = unit["slots"]
        tokens = unit.get("tokens", 0)
        required = unit.get("required_blocks")
        reason = unit.get("reason")
        if reason is None:
            if slots > batch_slots:
                reason = "compound_slot_limit"
            elif used_slots + slots > batch_slots:
                reason = "capacity"
            elif token_budget is not None and used_tokens + tokens > token_budget:
                reason = "token_budget"
            elif required > available:
                reason = "kv"
        if reason is None:
            accepted.append(unit)
            used_slots += slots
            used_tokens += tokens
            used_blocks += required
            available -= required
            unit_accounts.append({"ids": ids, "required_blocks": required,
                                  "disposition": "accepted", "available_after": available})
            continue

        reason = reason or "resource"
        running = unit.get("running", [])
        allowed = policy_preemption or reason not in ("capacity", "no_tokens")
        protected = unit.get("protected", False)
        if running and allowed and protected:
            leftover_running.extend(running)
            detail = "residency_guard"
        elif running and allowed and len(running) > swaps_remaining:
            leftover_running.extend(running)
            detail = "swap_limit"
        elif running and allowed:
            resident = unit.get("resident_blocks", 0)
            evictions.append((unit, running, reason))
            planned_freed += resident
            available += resident
            swaps_remaining -= len(running)
            unit_accounts.append({"ids": ids, "required_blocks": required,
                                  "disposition": "evict", "resident_freed": resident,
                                  "available_after": available})
            continue
        else:
            if running:
                leftover_running.extend(running)
            detail = "stable_batch" if running else "waiting_or_swapped"
        deferred.append((ids, reason, detail))
        unit_accounts.append({"ids": ids, "required_blocks": required,
                              "disposition": detail, "available_after": available})

    return {
        "accepted": accepted,
        "evictions": evictions,
        "deferred": deferred,
        "leftover_running": leftover_running,
        "unit_accounts": unit_accounts,
        "used_slots": used_slots,
        "used_blocks": used_blocks,
        "used_tokens": used_tokens,
        "planned_freed_blocks": planned_freed,
        "remaining_available_blocks": available,
        "swaps_remaining": swaps_remaining,
    }
