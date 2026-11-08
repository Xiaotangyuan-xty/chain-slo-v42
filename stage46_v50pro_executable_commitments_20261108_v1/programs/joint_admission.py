"""Mixed-only arbitration around the unchanged v45 chain selector."""
from chain_admission import blocks, select


def select_joint(goals, foreground, capacity, max_seqs, block_size=16, protected=()):
    """Foreground is already in native reward-density order.

    The half-pool/half-slot target is a coexistence policy, not a hard fairness
    guarantee. Existing v45 chain leases and its oversized fallback are retained.
    No virtual/unsubmitted member is ever passed to the physical selector.
    """
    if not any(goal['probability'] > 0 for goal in goals):
        _, rows = select(goals, capacity, max_seqs, block_size, protected)
        return None, rows, set()

    chain_members = {m['request']: m for goal in goals for m in goal['members']}
    protected = set(protected) & chain_members.keys()
    protected_cost = sum(blocks(chain_members[r], block_size) for r in protected)
    available = max(0, capacity - protected_cost)
    available_slots = max(0, max_seqs - len(protected))
    target = min(capacity // 2, available)
    target_slots = min(max(1, max_seqs // 2), available_slots)
    fg_selected = set()
    used = 0
    for member in foreground:
        charge = blocks(member, block_size)
        if used + charge <= target and len(fg_selected) < target_slots:
            fg_selected.add(member['request'])
            used += charge
    # A large foreground context may borrow the idle part of the chain target.
    if not fg_selected and available_slots:
        for member in foreground:
            charge = blocks(member, block_size)
            if charge <= available:
                fg_selected.add(member['request'])
                used = charge
                break

    remaining_slots = max_seqs - len(fg_selected)
    if capacity - used > 0 and remaining_slots > 0:
        chain_selected, rows = select(goals, capacity - used, remaining_slots,
                                      block_size, protected)
    else:
        chain_selected, rows = set(protected), []
    selected = fg_selected | chain_selected
    used += sum(blocks(chain_members[r], block_size) for r in chain_selected)
    for member in foreground:
        if member['request'] in selected:
            continue
        charge = blocks(member, block_size)
        if used + charge <= capacity and len(selected) < max_seqs:
            selected.add(member['request'])
            fg_selected.add(member['request'])
            used += charge
    return selected, rows, fg_selected
