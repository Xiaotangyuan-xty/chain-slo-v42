"""Frozen v47 pure path with explicit mixed-mode state isolation."""
import time

from frozen_v47.native_chain import ChainCompletionPolicy as FrozenPolicy
from frozen_v47.native_chain import MODEL
from frozen_v47.native_chain import NativeChainScheduler as FrozenScheduler


def _groups(scheduler):
    return list(scheduler.running) + list(scheduler.waiting) + list(scheduler.swapped)


def _reset_transition_state(scheduler):
    scheduler._clear_plan()
    scheduler.last_members = None
    scheduler.last_recorded_selected = None
    scheduler.fast_expired_drain = False
    scheduler.drain_key = None
    scheduler.scheduler_config.max_num_seqs = scheduler.base_max_num_seqs


def _rebuild_policy(scheduler, policy_type):
    policy = policy_type()
    policy.service_model = scheduler.cost_model
    for group in _groups(scheduler):
        policy.add_seq_group(group)
    scheduler.policy = policy


def enter_pure(scheduler):
    if getattr(scheduler, '_pure_core_active', False):
        return
    _reset_transition_state(scheduler)
    _rebuild_policy(scheduler, FrozenPolicy)
    scheduler._pure_core_active = True
    scheduler.pure_core_transitions = getattr(scheduler, 'pure_core_transitions', 0) + 1


def leave_pure(scheduler):
    if not getattr(scheduler, '_pure_core_active', False):
        return
    from mixed_joint import MixedPolicy
    _reset_transition_state(scheduler)
    _rebuild_policy(scheduler, MixedPolicy)
    scheduler._pure_core_active = False
    scheduler.mode = None


def schedule_pure(scheduler):
    enter_pure(scheduler)
    groups = [g for g in _groups(scheduler) if not g.is_finished()]
    now = time.perf_counter()
    mode = 'expired_drain' if groups and not scheduler.policy.has_live_credit(now) else 'pure'
    if scheduler.mode != mode:
        previous = scheduler.mode
        scheduler.mode = mode
        scheduler.mode_epoch += 1
        scheduler.v47_counts['mode_changes'] += 1
        scheduler._emit({'kind': 'mode_change', 'previous': previous, 'mode': mode})
    scheduler._planned_this_step = False
    scheduler._execution_flags = {'decision_id': scheduler.decision_id,
        'mode_epoch': scheduler.mode_epoch, 'mode': scheduler.mode,
        'candidate_filter': True, 'mask_id': None}
    previous_plan = scheduler.last_plan
    scheduler.v47_counts['pure_core_calls'] += 1
    result = FrozenScheduler._schedule(scheduler)
    if scheduler.last_plan != previous_plan:
        scheduler.v47_counts['plans'] += 1
        scheduler.decision_id += 1
        scheduler._emit({'kind': 'plan', 'mode': scheduler.mode,
            'tick': scheduler.last_plan, 'reasons': ['frozen_v47_replan'],
            'next_due': None, 'selected': sorted(scheduler.selected) if scheduler.selected is not None else None,
            'rows': scheduler.rows, 'running': [g.request_id for g in scheduler.running],
            'waiting': len(scheduler.waiting), 'swapped': len(scheduler.swapped)})
    return result
