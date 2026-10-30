import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import unittest
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from first_step import plan_first_step


@contextmanager
def patched_modules(modules):
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def module(name, **attributes):
    result = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


def load_module(name, path, modules):
    with patched_modules(modules):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
    return loaded


class Flexible:
    def __init__(self, *args, **kwargs):
        if args:
            names = ("now", "goals", "cache", "chain_virtual", "declared_inputs",
                     "compound_members")
            for name, value in zip(names, args):
                setattr(self, name, value)
        self.__dict__.update(kwargs)


class ExecutionView(Flexible):
    @property
    def calls(self):
        return tuple(goal.ready for goal in self.goals if goal.ready is not None)

    def to_dict(self):
        return {"goals": [goal.ready.key for goal in self.goals if goal.ready is not None]}


class FakeChainState:
    def __init__(self, stage_counts):
        self.current = {}
        self.meta = {}

    def add(self, group, now, active_collections):
        self.current[group.request_id] = {
            "level": 0, "quantum": 0.0, "ready_since": now,
            "service": 0.0, "wait": 0.0,
        }
        self.meta[group.collection_id] = {
            "arrival": now, "deadline": now + 120, "virtual": 0.0,
            "history": [], "completed_stages": set(), "declared_history": {},
            "tool_entered": None,
        }
        return False, 0

    def away_goals(self, now, ready_collections):
        return []

    def after_step(self, *args):
        return None

    def snapshot(self):
        return {}


class FakeSupportState:
    def __init__(self, pred):
        self.supported = True

    def check_goals(self, goals, decode_seconds, compound_members, declared):
        return SimpleNamespace(supported=self.supported, unsupported_keys=[],
                               denominator=len(goals), results=[])

    def fallback_action(self, calls, eligible, decode_seconds):
        key = sorted(eligible)[0]
        return (key, 4 * decode_seconds), {
            "accepted": False, "fallback": "official", "actions": [], "seconds": 0.0,
        }

    def observe_audit(self, audit, *args):
        return audit.get("fallback_kind", "official")

    def snapshot(self):
        return {}


class FakeSLOPolicy:
    def __init__(self, schedule_interval=50, penalty_factor=100):
        self.schedule_interval = schedule_interval
        self.penalty_factor = penalty_factor
        self.num_schedule_count = 0
        self.seq_group_dict = {}
        self.seq_group_slo_dict = {}
        self.interval_time = None

    def get_priority(self, group):
        return 0, group.arrival_time

    def add_seq_group(self, group):
        self.seq_group_dict.setdefault(group.collection_id, []).append(group)

    def delete_seq_group(self, group):
        self.seq_group_dict.get(group.collection_id, []).remove(group)


def load_policy():
    modules = {
        "contracts": module("contracts", CacheEntry=Flexible, CallView=Flexible),
        "pool_engine": module("pool_engine", PoolCache=Flexible),
        "support_state": module("support_state", SupportState=FakeSupportState),
        "execution_view": module("execution_view", ExecutionView=ExecutionView),
        "v2_contracts": module("v2_contracts", GoalView=Flexible),
        "v2_state": module("v2_state", ChainState=FakeChainState),
        "jitserve": module("jitserve"),
        "jitserve.scheduler": module("jitserve.scheduler"),
        "jitserve.scheduler.policy": module("jitserve.scheduler.policy", SLOPolicy=FakeSLOPolicy),
    }
    return load_module("_stage7_policy_under_test", "v2_compound_policy.py", modules)


class FakeSeq:
    def __init__(self, prompt=5540):
        self.prompt = prompt

    def is_finished(self):
        return False

    def get_prompt_len(self):
        return self.prompt

    def get_output_len(self):
        return 0


class FakeGroup:
    def __init__(self, request_id, collection=1, stage=4, prompt=5540):
        self.request_id = request_id
        self.collection_id = collection
        self.arrival_time = 0.0
        self.request_info = SimpleNamespace(stage_id=stage, input_len=prompt)
        self.first_seq = FakeSeq(prompt)

    def is_prefill(self):
        return True


class RuntimeGroup(FakeGroup):
    def __init__(self, request_id, collection, stage, demand, required, resident):
        super().__init__(request_id, collection, stage, 5540 if demand > 1 else 16)
        self.demand = demand
        self.required = required
        self.resident = resident

    def get_max_num_running_seqs(self):
        return 1


class FakeLedger:
    gpu_capacity = 8192
    host_capacity = 8192

    def __init__(self, groups):
        self.entries = {group.request_id: SimpleNamespace(
            gpu=0, host=0, gpu_touch=0, host_touch=0, pending=0) for group in groups}

    def view(self):
        return self.entries


def policy_with_groups(groups):
    loaded = load_policy()
    policy = loaded.V2CompoundPolicy(SimpleNamespace(decode_seconds=0.01), None,
                                     {group.collection_id: 6 for group in groups})
    now = time.perf_counter()
    for group in groups:
        policy.current[group.request_id] = {
            "level": 0, "quantum": 0.0, "ready_since": now,
            "service": 0.0, "wait": 0.0,
        }
        policy.chain_meta[group.collection_id] = {
            "arrival": now, "deadline": now + 120, "virtual": 0.0,
            "history": [], "completed_stages": set(), "declared_history": {},
            "tool_entered": None,
        }
    policy.running_ids = set()
    policy.swapped_ids = set()
    policy.custom_forced_enabled = True
    policy.resource_provider = lambda current: {
        "units": {}, "free_kv_blocks": 10, "watermark_blocks": 0,
        "max_swaps_per_iter": 3, "token_budget": 2048,
    }
    return policy, now, FakeLedger(groups)


class FakeBudget:
    def __init__(self, token_budget=2048, max_num_seqs=4):
        self.token_budget = token_budget
        self.max_num_seqs = max_num_seqs
        self._tokens = 0
        self._seqs = 0

    def remaining_token_budget(self):
        return self.token_budget - self._tokens

    @property
    def num_batched_tokens(self):
        return self._tokens

    @property
    def num_curr_seqs(self):
        return self._seqs

    def can_schedule(self, num_new_tokens, num_new_seqs):
        return (self._tokens + num_new_tokens <= self.token_budget
                and self._seqs + num_new_seqs <= self.max_num_seqs)

    def add_num_batched_tokens(self, request_id, value):
        self._tokens += value

    def add_num_seqs(self, request_id, value):
        self._seqs += value


class FakeOfficialScheduler:
    def _adjust_batch_limits(self, process_list, min_tbt_required, budget):
        budget.max_num_seqs = max(4, len(process_list))

    def _test_enough_swap_space(self):
        return getattr(self, "enough_swap_space", True)

    def _schedule_slo(self):
        self.official_path = "official_50_tick"
        return [], SimpleNamespace(scheduled_seq_groups=[]), False

    def _schedule_chunked_prefill(self):
        self.official_path = "official_chunked_prefill"
        return [], SimpleNamespace(scheduled_seq_groups=[]), False

    def schedule(self):
        return self._schedule()

    def _schedule_process_list(self, *args):
        return "official_process_list"

    def _schedule_running(self, *args):
        return "official_running"

    def _schedule_prefills(self, *args):
        return "official_prefills"

    def _schedule_swapped(self, *args):
        return "official_swapped"

    def _build_process_list(self, *args):
        return "official_build_process_list"

    def _preempt_pending(self, *args):
        return "official_preempt_pending"


def load_scheduler():
    class FakeRecomputeInvalidatingScheduler:
        pass

    preemption = SimpleNamespace(SWAP="SWAP", RECOMPUTE="RECOMPUTE")
    status = SimpleNamespace(WAITING="WAITING", SWAPPED="SWAPPED", RUNNING="RUNNING")
    alloc = SimpleNamespace(OK="OK", LATER="LATER", NEVER="NEVER")

    class Empty:
        def __init__(self):
            self.seq_groups = []
            self.ignored_seq_groups = []
            self.infeasible_seq_groups = []
            self.decode_seq_groups = []
            self.prefill_seq_groups = []
            self.blocks_to_swap_in = []
            self.blocks_to_copy = []
            self.blocks_to_swap_out = []
            self.leftover_running = []
            self.preempted = []
            self.swapped_out = []
            self.prefill_seq_groups_list = []
            self.decode_seq_groups_list = []
            self.num_lookahead_slots = 0

        @classmethod
        def create_empty(cls):
            return cls()

    class SchedulerOutputs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class HostCapacityDefer(Exception):
        pass

    modules = {
        "jitserve": module("jitserve"),
        "jitserve.scheduler": module("jitserve.scheduler"),
        "jitserve.scheduler.slo_scheduler": module(
            "jitserve.scheduler.slo_scheduler", SLOScheduler=FakeOfficialScheduler),
        "scheduler_recompute": module(
            "scheduler_recompute", RecomputeInvalidatingScheduler=FakeRecomputeInvalidatingScheduler,
            HostCapacityDefer=HostCapacityDefer),
        "v2_compound_policy": module("v2_compound_policy", V2CompoundPolicy=object),
        "vllm": module("vllm"),
        "vllm.core": module("vllm.core"),
        "vllm.core.scheduler": module(
            "vllm.core.scheduler", PreemptionMode=preemption, SchedulingBudget=FakeBudget,
            SchedulerOutputs=SchedulerOutputs, SchedulerRunningOutputs=Empty,
            SchedulerPrefillOutputs=Empty, SchedulerSwappedInOutputs=Empty),
        "vllm.core.interfaces": module("vllm.core.interfaces", AllocStatus=alloc),
        "vllm.sequence": module("vllm.sequence", SequenceStatus=status),
    }
    return load_module("_stage7_scheduler_under_test", "v2_slo_scheduler.py", modules)


def run_trace_frame(infeasible=False, expected_first=None, resident_set=None, already_scheduled=False):
    loaded = load_scheduler()
    scheduler = object.__new__(loaded.V2SLOScheduler)
    target = RuntimeGroup("chain1-stage4", 1, 4, 2048, 347, 0)
    free = [10]
    if infeasible:
        retained = RuntimeGroup("protected", 2, 0, 0, 0, 502)
        lows = []
    else:
        retained = RuntimeGroup("retained", 4, 0, 1, 0, 142)
        lows = [RuntimeGroup("low-a", 2, 0, 1, 0, 180),
                RuntimeGroup("low-b", 3, 0, 1, 0, 180)]
    scheduler.running = deque(lows + [retained])
    scheduler.waiting = deque([target])
    scheduler.swapped = deque()
    scheduler.scheduler_config = SimpleNamespace(max_num_batched_tokens=2048)
    scheduler.max_batch_size = 4
    scheduler.max_swaps_per_iter = 3
    scheduler.compound_events = []
    scheduler.policy = SimpleNamespace(
        forced="1:stage:4", model_action=False, compound_members={}, batch_slots=4)
    scheduler.policy.policy_preemption_allowed = lambda: True
    scheduler.policy.residency_protected = lambda group: group is retained
    if expected_first is not None:
        scheduler.policy.model_action = True
        scheduler.policy.pending_action = {"expected_first": expected_first,
                                           "scheduled": already_scheduled}
        scheduler.policy.resident_order = list(expected_first)
    if resident_set is not None:
        scheduler.policy.resident_set = set(resident_set)
        scheduler.policy.compound_members = {
            "1:stage:4": [target.request_id],
            "2:stage:0": [lows[0].request_id] if lows else [],
            "3:stage:0": [lows[1].request_id] if len(lows) > 1 else [],
            "4:stage:0": [retained.request_id],
        }
    scheduler.policy.current = {group.request_id: {"level": 0, "ready_since": 0.0}
                                for group in [target] + lows + [retained]}
    scheduler.cache_id = 0
    scheduler._scheduled_seq_group_cache = {
        0: SimpleNamespace(get_object=lambda: SimpleNamespace(seq_group=None, token_chunk_size=0))
    }
    scheduler._get_num_new_tokens = lambda group, status, chunking, budget: group.demand
    scheduler._get_num_lookahead_slots = lambda prefill, chunking: 0
    scheduler._get_prompt_limit = lambda group: 8192
    scheduler._append_blocks_required = lambda group, lookahead: 0
    scheduler._can_append_slots = lambda group, chunking: True
    scheduler._append_slots = lambda group, blocks, chunking: None
    manager = SimpleNamespace(num_total_gpu_blocks=512)
    manager.get_num_free_gpu_blocks = lambda: free[0]
    manager.get_max_gpu_num_blocks_number = lambda: 512
    manager.can_allocate = lambda group, lookahead: loaded.AllocStatus.OK
    manager.get_num_blocks_number = lambda group, status, lookahead: group.required
    manager.compute_running_seq_group_blocks = lambda groups: groups[0].resident
    scheduler.block_manager = manager

    def preempt(group, blocks, mode):
        free[0] += group.resident
        return loaded.PreemptionMode.SWAP

    def schedule_prefill(group, tokens, lookahead, chunking):
        free[0] -= group.required
        scheduled = SimpleNamespace(seq_group=group, token_chunk_size=tokens)
        return SimpleNamespace(seq_group=scheduled)

    scheduler._preempt = preempt
    scheduler._schedule_slo_prefills = schedule_prefill
    running_result = loaded.SchedulerRunningOutputs.create_empty()
    prefill_result = loaded.SchedulerPrefillOutputs.create_empty()
    swapped_in_result = loaded.SchedulerSwappedInOutputs.create_empty()
    budget = loaded.SchedulingBudget(2048, 4)
    process = [target] + lows + [retained]
    feasibility = (scheduler._action_feasibility("1:stage:4")
                   if resident_set is not None else None)
    scheduler._schedule_process_list(process, running_result, prefill_result,
                                     swapped_in_result, budget, True)
    after_schedule = free[0]
    for item in prefill_result.seq_groups:
        free[0] += item.seq_group.required
    free[0] += retained.resident
    return {
        "feasibility": feasibility,
        "chunks": [item.token_chunk_size for item in prefill_result.seq_groups],
        "scheduled": [item.seq_group.request_id for item in prefill_result.seq_groups],
        "preempted": [group.request_id for group in running_result.swapped_out],
        "leftover": [group.request_id for group in running_result.leftover_running],
        "after_schedule_free": after_schedule,
        "final_free": free[0],
        "event": scheduler._dispatch_event,
        "budget_tokens": budget.num_batched_tokens,
    }


def planner_unit(ids, slots=1, tokens=1, required=0, running=(), resident=0,
                 protected=False, minimum=None, reason=None, state=None):
    return {
        "ids": list(ids), "slots": slots, "tokens": tokens,
        "min_tokens": tokens if minimum is None else minimum,
        "required_blocks": required, "running": list(running),
        "resident_blocks": resident, "protected": protected,
        "reason": reason, "state": state,
    }


class FirstStepTests(unittest.TestCase):
    def test_trace_5540_forced_execution_and_no_kv_leak(self):
        target = planner_unit(["chain1-stage4"], tokens=2048, minimum=1,
                              required=(5540 + 15) // 16, state="WAITING")
        low_a = planner_unit(["low-a"], running=["low-a"], resident=180)
        low_b = planner_unit(["low-b"], running=["low-b"], resident=180)
        plan = plan_first_step([target, low_a, low_b], 10, 4, 3, True, 2048)
        self.assertEqual(plan["accepted"][0]["tokens"], 2048)
        self.assertEqual(plan["accepted"][0]["requested_tokens"], 2048)
        self.assertEqual(plan["used_blocks"], 347)
        self.assertEqual(plan["planned_freed_blocks"], 360)
        self.assertEqual(plan["remaining_available_blocks"], 23)
        self.assertEqual({group for _, running, _ in plan["evictions"] for group in running},
                         {"low-a", "low-b"})
        free = 10
        for _, running, _ in plan["evictions"]:
            for request_id in running:
                free += 180 if request_id == "low-a" else 180
        for entry in plan["accepted"]:
            free -= entry["required_blocks"]
        self.assertEqual(free, 23)
        free += 347
        free += 142
        self.assertEqual(free, 512)

    def test_trace_5540_infeasible_does_not_evict_or_spin(self):
        target = planner_unit(["chain1-stage4"], tokens=2048, minimum=1, required=347)
        protected = planner_unit(["protected"], running=["protected"], resident=360,
                                 protected=True)
        plan = plan_first_step([target, protected], 10, 4, 3, True, 2048)
        self.assertFalse(plan["accepted"][0]["ids"] == ["chain1-stage4"])
        self.assertFalse(any(["chain1-stage4"] == entry["ids"] for entry in plan["accepted"]))
        self.assertEqual(plan["evictions"], [])
        self.assertEqual(plan["deferred"][0][1], "kv")
        self.assertEqual(plan["remaining_available_blocks"], 10)

    def test_running_kv_pressure_evicts_lower_priority_running(self):
        high = planner_unit(["high"], tokens=1, required=1,
                            running=["high"], resident=217)
        low = planner_unit(["low"], tokens=1, required=1,
                           running=["low"], resident=290)
        plan = plan_first_step([high, low], 0, 2, 3, False, 2048)
        self.assertEqual([entry["ids"] for entry in plan["accepted"]], [["high"]])
        self.assertEqual([unit["ids"] for unit, _, _ in plan["evictions"]], [["low"]])
        self.assertEqual(plan["remaining_available_blocks"], 289)

    def test_compound_eviction_and_slot_failure_are_atomic(self):
        compound = planner_unit(["a", "b"], slots=2, tokens=200, minimum=2, required=30)
        low = planner_unit(["low-a", "low-b"], slots=2, tokens=2,
                           running=["low-a", "low-b"], resident=20)
        plan = plan_first_step([compound, low], 10, 2, 2, True, 2048)
        self.assertEqual(plan["accepted"][0]["ids"], ["a", "b"])
        self.assertEqual(len(plan["evictions"]), 1)
        self.assertEqual(plan["swaps_remaining"], 0)
        too_wide = planner_unit(["a", "b", "c"], slots=3, tokens=3, required=1)
        blocked = plan_first_step([too_wide, low], 100, 2, 3, True, 2048)
        self.assertEqual(blocked["evictions"], [])
        self.assertEqual(blocked["deferred"][0][1], "compound_slot_limit")

    def test_retained_waiting_swapped_and_swap_limit(self):
        retained = planner_unit(["retained"], tokens=2048, minimum=1, required=50,
                                running=["retained"], resident=50)
        plan = plan_first_step([retained], 10, 4, 3, True, 2048)
        self.assertEqual(plan["leftover_running"], ["retained"])
        self.assertEqual(plan["evictions"], [])
        for state in ("WAITING", "SWAPPED"):
            target = planner_unit([state], tokens=10, required=20, state=state)
            low = planner_unit(["low"], running=["low"], resident=20)
            plan = plan_first_step([target, low], 0, 4, 1, True, 2048)
            self.assertEqual(plan["accepted"][0]["state"], state)
            self.assertEqual(plan["evictions"][0][1], ["low"])
        target = planner_unit(["target"], tokens=10, required=40)
        low_a = planner_unit(["low-a"], running=["low-a"], resident=20)
        low_b = planner_unit(["low-b"], running=["low-b"], resident=20)
        limited = plan_first_step([target, low_a, low_b], 0, 4, 1, True, 2048)
        self.assertEqual(limited["evictions"], [])
        self.assertEqual(limited["deferred"][0][1], "kv")

    def test_prefill_shrinks_at_planning_and_never_expands_afterwards(self):
        target = planner_unit(["long"], tokens=2048, minimum=1, required=347)
        plan = plan_first_step([target], 500, 4, 3, False, 1000)
        self.assertEqual(plan["accepted"][0]["requested_tokens"], 2048)
        self.assertEqual(plan["accepted"][0]["tokens"], 1000)
        loaded = load_scheduler()
        scheduler = object.__new__(loaded.V2SLOScheduler)
        group = SimpleNamespace(request_id="long")
        scheduler._token_demands = lambda unit, chunking: {"long": 2048}
        allocation = scheduler._allocate_unit_tokens([group], 1000, True)
        self.assertEqual(allocation, {"long": 1000})
        first = SimpleNamespace(request_id="a")
        second = SimpleNamespace(request_id="b")
        scheduler._token_demands = lambda unit, chunking: {"a": 2048, "b": 2048}
        allocation = scheduler._allocate_unit_tokens([first, second], 2048, True)
        self.assertEqual(allocation, {"a": 1024, "b": 1024})


class PolicyTests(unittest.TestCase):
    def test_forced_has_scheduled_and_executed_witness(self):
        group = FakeGroup("chain1-stage4")
        policy, now, ledger = policy_with_groups([group])
        policy.current[group.request_id]["ready_since"] = now - 10
        policy.frame_pending = True
        policy.feasibility_provider = lambda key: {"feasible": True, "target": [group.request_id]}
        policy.before_schedule([group], now, ledger)
        self.assertEqual(policy.forced, "1:stage:4")
        policy.note_selection([group], now, scheduled_details=[
            {"request_id": group.request_id, "token_chunk_size": 2048, "prefill": True}])
        self.assertTrue(policy.forced_action["scheduled"])
        self.assertEqual(policy.effective_controller, "forced")
        policy.after_model_step(0.01, now + 0.01)
        self.assertTrue(policy.forced_action["executed"])

    def test_infeasible_or_nonframe_forced_does_not_take_over(self):
        for frame in (True, False):
            group = FakeGroup("chain1-stage4")
            policy, now, ledger = policy_with_groups([group])
            policy.current[group.request_id]["ready_since"] = now - 10
            policy.frame_pending = frame
            policy.feasibility_provider = lambda key: {"feasible": False, "reason": "kv"}
            policy.before_schedule([group], now, ledger)
            self.assertIsNone(policy.forced)
            self.assertFalse(policy.policy_preemption_allowed())
            self.assertFalse(any(event["kind"] == "v2_forced_accepted"
                                 for event in policy.audit_events))

    def test_attainable_deadline_overrides_virtual_eligibility(self):
        group = FakeGroup("chain1-stage4")
        policy, now, ledger = policy_with_groups([group])
        policy.chain_meta[group.collection_id]["virtual"] = 100.0
        policy.chain_meta[group.collection_id]["deadline"] = now + 200.0
        policy.pred = SimpleNamespace(estimate=lambda *args, **kwargs: {"work": 100.0})
        policy.frame_pending = True
        policy.before_schedule([group], now, ledger)
        candidate = next(event for event in policy.audit_events
                         if event["kind"] == "v2_candidate")
        self.assertIn("1:stage:4", candidate["eligible"])

    def test_forced_override_of_model_lease_is_audited(self):
        group = FakeGroup("chain1-stage4")
        policy, now, ledger = policy_with_groups([group])
        policy.current[group.request_id]["ready_since"] = now - 10
        policy.frame_pending = True
        policy.model_action = True
        policy.anchor = "other"
        policy.remaining = 0.03
        policy.pending_action = {"id": 7, "scheduled": False, "executed": False}
        policy.feasibility_provider = lambda key: {"feasible": True}
        policy.before_schedule([group], now, ledger)
        override = [event for event in policy.audit_events
                    if event["kind"] == "v2_forced_lease_override"]
        self.assertEqual(len(override), 1)
        self.assertEqual(override[0]["previous_action_id"], 7)
        self.assertIsNone(policy.pending_action)

    def test_pending_action_uses_trace_and_selector_signature_is_dynamic(self):
        group = FakeGroup("chain1-stage4")
        policy, now, ledger = policy_with_groups([group])
        policy.frame_pending = True
        calls = {}

        def selector(view, instance, eligible, pred, perf, owners, **kwargs):
            calls.update(kwargs)
            key = "1:stage:4"
            return (key, 0.04), {
                "accepted": True, "fallback": None, "seconds": 0.0,
                "actions": [{"key": key, "h": 0.04, "batch_trace": [
                    {"keys": [key], "slots": 1, "lease_after": 0.03}]}],
            }

        policy.candidate_selector = selector
        policy.batch_slots = 7
        policy.feasibility_provider = lambda key: {"feasible": True}
        policy.before_schedule([group], now, ledger)
        self.assertEqual(calls["batch_slots"], 7)
        self.assertIn("compound_members", calls)
        self.assertEqual(policy.pending_action["expected_first"], ["1:stage:4"])
        policy.note_selection([group], now, scheduled_details=[
            {"request_id": group.request_id, "token_chunk_size": 2048}])
        policy.after_model_step(0.01, now + 0.01)
        self.assertTrue(policy.pending_action["scheduled"])
        self.assertTrue(policy.pending_action["executed"])
        self.assertEqual(policy.effective_controller, "model")
        for step in range(1, 4):
            policy.after_model_step(0.01, now + 0.01 + step * 0.01)
        self.assertTrue(policy.pending_action["lease_first_matches"])
        self.assertTrue(policy.pending_action["lease_completed"])
        self.assertEqual(policy.pending_action["decode_steps"], 4)
        self.assertEqual(sum(event["kind"] == "v2_lease_completed"
                             for event in policy.audit_events), 1)

    def test_active_model_lease_executes_and_decrements_off_frame(self):
        selected = FakeGroup("selected", collection=1, stage=4)
        parked = FakeGroup("parked", collection=2, stage=4)
        policy, now, ledger = policy_with_groups([selected, parked])
        policy.frame_pending = True

        def selector(view, instance, eligible, pred, perf, owners, **kwargs):
            return ("1:stage:4", 0.04), {
                "accepted": True, "fallback": None, "seconds": 0.0,
                "actions": [{"key": "1:stage:4", "h": 0.04, "batch_trace": [
                    {"keys": ["1:stage:4"], "slots": 1, "lease_after": 0.03}]}],
            }

        policy.candidate_selector = selector
        policy.feasibility_provider = lambda key: {"feasible": True}
        policy.before_schedule([selected, parked], now, ledger)
        policy.note_selection([selected, parked], now)
        policy.after_model_step(0.01, now + 0.01)
        for step in range(1, 4):
            policy.frame_pending = False
            policy.before_schedule([selected, parked], now + step * 0.01, ledger)
            self.assertTrue(policy.custom_dispatch_active())
            self.assertEqual(policy.get_priority(selected)[0], -float("inf"))
            self.assertEqual(policy.get_priority(parked), (0, parked.arrival_time))
            policy.note_selection([selected, parked], now + step * 0.01)
            policy.after_model_step(0.01, now + (step + 1) * 0.01)
            self.assertEqual(policy.effective_controller, "model")
        self.assertAlmostEqual(policy.remaining, 0.0)
        self.assertTrue(policy.pending_action["lease_completed"])
        self.assertEqual(policy.pending_action["decode_steps"], 4)

    def test_completed_model_lease_renews_without_replan(self):
        selected = FakeGroup("selected", collection=1, stage=4)
        parked = FakeGroup("parked", collection=2, stage=4)
        policy, now, ledger = policy_with_groups([selected, parked])
        calls = []

        def selector(view, instance, eligible, pred, perf, owners, **kwargs):
            calls.append(view)
            return ("1:stage:4", 0.01), {
                "accepted": True, "fallback": None, "seconds": 0.0,
                "actions": [{"key": "1:stage:4", "h": 0.01, "batch_trace": [
                    {"keys": ["1:stage:4"], "slots": 1, "lease_after": 0.0}]}],
            }

        policy.candidate_selector = selector
        policy.feasibility_provider = lambda key: {"feasible": True}
        policy.frame_pending = True
        policy.before_schedule([selected, parked], now, ledger)
        policy.note_selection([selected, parked], now)
        policy.after_model_step(0.01, now + 0.01)
        self.assertTrue(policy.pending_action["lease_completed"])
        residents = set(policy.resident_set)
        policy.num_schedule_count = 50
        policy.frame_pending = True
        policy.before_schedule([selected, parked], now + 0.02, ledger)
        self.assertEqual(len(calls), 1)
        self.assertEqual(policy.resident_set, residents)
        self.assertAlmostEqual(policy.remaining, 0.5)
        renewals = [event for event in policy.audit_events
                    if event["kind"] == "v2_lease_renewed"]
        self.assertEqual(len(renewals), 1)
        self.assertEqual(renewals[0]["id"], 1)

    def test_lease_yields_to_more_urgent_attainable_chain(self):
        first = FakeGroup("first", collection=1, stage=4)
        second = FakeGroup("second", collection=2, stage=4)
        policy, now, ledger = policy_with_groups([first, second])
        policy.defer_safe_later_enabled = False  # Isolate the explicit lease-yield feature.
        policy.chain_meta[1]["deadline"] = now + 100.0
        policy.chain_meta[2]["deadline"] = now + 50.0
        policy.pred = SimpleNamespace(
            estimate=lambda goal, *args, **kwargs: {"work": 10.0})
        policy.stage_counts[1] = 4
        policy.lease_yield_enabled = True
        policy.model_action = True
        policy.anchor = "1:stage:4"
        policy.remaining = 0.0
        policy.resident_set = {"1:stage:4"}
        policy.pending_action = {"id": 99, "lease_completed": True}
        calls = []

        def selector(view, instance, eligible, pred, perf, owners, **kwargs):
            calls.append(kwargs.get("incumbent"))
            return ("2:stage:4", 0.04), {
                "accepted": True, "fallback": None, "seconds": 0.0,
                "actions": [{"key": "2:stage:4", "h": 0.04, "batch_trace": [
                    {"keys": ["2:stage:4"], "slots": 1, "lease_after": 0.03}]}],
            }

        policy.candidate_selector = selector
        policy.feasibility_provider = lambda key: {"feasible": True}
        policy.frame_pending = True
        policy.before_schedule([first, second], now, ledger)
        self.assertEqual(policy.anchor, "2:stage:4")
        self.assertEqual(calls, [("1:stage:4", 0.0)])
        self.assertFalse(any(event["kind"] == "v2_lease_renewed"
                             for event in policy.audit_events))
        yields = [event for event in policy.audit_events if event["kind"] == "v2_lease_yield"]
        self.assertEqual(len(yields), 1)
        self.assertEqual(yields[0]["next_anchor"], "2:stage:4")

    def test_nonfinal_anchor_does_not_yield_lease(self):
        first = FakeGroup("first", collection=1, stage=4)
        second = FakeGroup("second", collection=2, stage=4)
        policy, now, ledger = policy_with_groups([first, second])
        policy.defer_safe_later_enabled = False  # Cohort admission has its own boundary test.
        policy.chain_meta[1]["deadline"] = now + 100.0
        policy.chain_meta[2]["deadline"] = now + 50.0
        policy.pred = SimpleNamespace(estimate=lambda *args, **kwargs: {"work": 10.0})
        policy.model_action = True
        policy.anchor = "1:stage:4"
        policy.remaining = 0.0
        policy.resident_set = {"1:stage:4"}
        policy.pending_action = {"id": 99, "lease_completed": True}
        policy.candidate_selector = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("non-final anchor must renew without reevaluation"))
        policy.frame_pending = True
        policy.before_schedule([first, second], now, ledger)
        self.assertEqual(policy.anchor, "1:stage:4")
        self.assertAlmostEqual(policy.remaining, 0.5)
        renewals = [event for event in policy.audit_events
                    if event["kind"] == "v2_lease_renewed"]
        self.assertEqual(len(renewals), 1)
        self.assertFalse(any(event["kind"] == "v2_lease_yield"
                             for event in policy.audit_events))

    def test_fallback_anchor_is_reevaluated_on_every_official_frame(self):
        groups = [FakeGroup("first", collection=1, stage=4),
                  FakeGroup("second", collection=2, stage=4)]
        policy, now, ledger = policy_with_groups(groups)
        calls = []

        def selector(view, instance, eligible, pred, perf, owners, **kwargs):
            calls.append(dict(view=view, eligible=eligible))
            return ("1:stage:4", 0.04), {
                "accepted": False, "fallback": "no_model_gain", "actions": [], "seconds": 0.0,
            }

        policy.candidate_selector = selector
        for _ in range(2):
            policy.frame_pending = True
            policy.before_schedule(groups, now, ledger)
            self.assertFalse(policy.model_action)
            self.assertEqual(policy.anchor, "1:stage:4")
            self.assertGreater(policy.remaining, 0.0)
        self.assertEqual(len(calls), 2)
        self.assertIn("1:stage:4", calls[-1]["eligible"])

    def test_nonframe_no_model_skips_expensive_view(self):
        group = FakeGroup("chain1-stage4")
        policy, now, ledger = policy_with_groups([group])
        policy.frame_pending = False
        calls = []
        policy.resource_provider = lambda groups: calls.append(groups) or {}
        policy.before_schedule([group], now, ledger)
        self.assertEqual(calls, [])
        self.assertIsNone(policy.last_view)
        self.assertEqual(policy.effective_controller, "official")
        policy.note_selection([group], now)
        policy.after_model_step(0.01, now + 0.01)
        self.assertFalse(any(event["kind"] in {
            "v2_selected", "v2_after_step", "v2_effective_controller"}
            for event in policy.audit_events))

    def test_resident_set_has_real_priority_effect(self):
        selected = FakeGroup("selected", collection=1, stage=4)
        parked = FakeGroup("parked", collection=2, stage=4)
        policy, now, ledger = policy_with_groups([selected, parked])
        policy.frame_pending = True

        def selector(view, instance, eligible, pred, perf, owners, **kwargs):
            return ("1:stage:4", 0.04), {
                "accepted": True, "fallback": None, "seconds": 0.0,
                "actions": [{"key": "1:stage:4", "h": 0.04, "batch_trace": [
                    {"keys": ["1:stage:4"], "slots": 1, "lease_after": 0.03}]}],
            }

        policy.candidate_selector = selector
        policy.feasibility_provider = lambda key: {"feasible": True}
        policy.before_schedule([selected, parked], now, ledger)
        self.assertEqual(policy.resident_set, {"1:stage:4"})
        self.assertEqual(policy.get_priority(selected)[0], -float("inf"))
        self.assertEqual(policy.get_priority(parked), (0, parked.arrival_time))

    def test_recovery_metadata_does_not_replace_native_priority(self):
        anchor = FakeGroup("anchor", collection=1, stage=4)
        urgent = FakeGroup("urgent", collection=2, stage=4)
        later = FakeGroup("later", collection=3, stage=4)
        parked = FakeGroup("parked", collection=4, stage=4)
        groups = [anchor, urgent, later, parked]
        policy, now, ledger = policy_with_groups(groups)
        policy.chain_meta[2]["deadline"] = now + 200.0
        policy.chain_meta[3]["deadline"] = now + 300.0

        def selector(view, instance, eligible, pred, perf, owners, **kwargs):
            return ("1:stage:4", 0.04), {
                "accepted": True, "fallback": None, "seconds": 0.0,
                "evicted_attainable": {"2:stage:4": {}, "3:stage:4": {}},
                "actions": [{"key": "1:stage:4", "h": 0.04, "batch_trace": [
                    {"keys": ["1:stage:4"], "slots": 1, "lease_after": 0.03}]}],
            }

        policy.candidate_selector = selector
        policy.feasibility_provider = lambda key: {"feasible": True}
        policy.frame_pending = True
        policy.before_schedule(groups, now, ledger)
        self.assertEqual(policy.recovery_set, {"2:stage:4", "3:stage:4"})
        self.assertEqual(policy.get_priority(anchor)[0], -float("inf"))
        self.assertEqual(policy.get_priority(urgent), (0, urgent.arrival_time))
        self.assertEqual(policy.get_priority(later), (0, later.arrival_time))
        self.assertEqual(policy.get_priority(parked), (0, parked.arrival_time))

    def test_far_deadline_chain_is_deferred_behind_native_urgent_chain(self):
        anchor = FakeGroup("anchor", collection=1, stage=4)
        urgent = FakeGroup("urgent", collection=2, stage=4)
        far = FakeGroup("far", collection=3, stage=4)
        attainable_far = FakeGroup("attainable_far", collection=4, stage=4)
        policy, now, ledger = policy_with_groups([anchor, urgent, far, attainable_far])
        policy.chain_meta[2]["deadline"] = now + 120.01
        policy.chain_meta[3]["deadline"] = now + 1000.0
        policy.chain_meta[4]["deadline"] = now + 1000.0
        policy.pred = SimpleNamespace(
            estimate=lambda goal, *args, **kwargs: {"work": 1001.0 if goal.program == 3 else 1.0})

        def selector(view, instance, eligible, pred, perf, owners, **kwargs):
            return ("1:stage:4", 0.04), {
                "accepted": True, "fallback": None, "seconds": 0.0,
                "actions": [{"key": "1:stage:4", "h": 0.04, "batch_trace": [
                    {"keys": ["1:stage:4"], "slots": 1, "lease_after": 0.03}]}],
            }

        policy.candidate_selector = selector
        policy.feasibility_provider = lambda key: {"feasible": True}
        policy.frame_pending = True
        policy.before_schedule([anchor, urgent, far, attainable_far], now, ledger)
        self.assertEqual(policy.deadline_deferred, {"3:stage:4"})
        self.assertNotIn("4:stage:4", policy.deadline_deferred)
        self.assertEqual(policy.get_priority(anchor)[0], -float("inf"))
        self.assertEqual(policy.get_priority(urgent), (0, urgent.arrival_time))
        self.assertEqual(policy.get_priority(attainable_far), (0, attainable_far.arrival_time))
        self.assertEqual(policy.get_priority(far)[0], float("inf"))
        policy.model_action = False
        self.assertTrue(policy.custom_dispatch_active())
        self.assertEqual(policy.get_priority(far)[0], float("inf"))
        policy.running_ids = {far.request_id}
        self.assertFalse(policy.residency_protected(far))
        policy.frame_pending = False
        policy.note_selection([far], now)
        self.assertEqual(policy.effective_controller, "deadline_class")

    def test_deadline_class_preserves_native_order_without_model_action(self):
        urgent = FakeGroup("urgent", collection=1)
        later = FakeGroup("later", collection=2)
        late = FakeGroup("late", collection=3)
        urgent.arrival_time, later.arrival_time = 1.0, 2.0
        policy, now, ledger = policy_with_groups([urgent, later, late])
        policy.deadline_deferred = {"3:stage:4"}
        policy.priorities = {urgent.request_id: (9, 1), later.request_id: (1, 2)}
        self.assertLess(policy.get_priority(urgent), policy.get_priority(later))
        self.assertEqual(policy.get_priority(urgent), (0, 1.0))
        policy.frame_pending = False
        policy.before_schedule([urgent, later, late], now, ledger)
        self.assertEqual(policy.get_priority(urgent), (0, 1.0))
        self.assertEqual(policy.get_priority(later), (0, 2.0))
        self.assertEqual(policy.get_priority(late)[0], float("inf"))

    def test_hopeless_admission_pauses_then_drains_every_call(self):
        viable = FakeGroup("viable", collection=1)
        background = FakeGroup("background", collection=2)
        policy, now, ledger = policy_with_groups([viable, background])
        policy.pred = SimpleNamespace(estimate=lambda *args, **kwargs: {"work": 10.})
        policy.chain_meta[1]["deadline"] = now + 100.
        policy.chain_meta[2]["deadline"] = now + 1.
        policy.frame_pending = True
        policy.before_schedule([viable, background], now, ledger)
        self.assertEqual(policy.admission_paused, {"2:stage:4"})
        policy.before_schedule([background], now, ledger)
        self.assertEqual(policy.admission_paused, set())
        self.assertEqual(policy.audit_events[-2]["kind"], "v2_admission_pause")

    def test_admission_pause_preserves_an_unfinished_bounded_lease(self):
        anchor = FakeGroup("anchor", collection=1)
        viable = FakeGroup("viable", collection=2)
        policy, now, ledger = policy_with_groups([anchor, viable])
        policy.pred = SimpleNamespace(estimate=lambda *args, **kwargs: {"work": 10.})
        policy.chain_meta[1]["deadline"] = now + 1.
        policy.chain_meta[2]["deadline"] = now + 100.
        policy.model_action = True
        policy.anchor = "1:stage:4"
        policy.remaining = 0.02
        policy.frame_pending = False
        policy.before_schedule([anchor, viable], now, ledger)
        self.assertNotIn(policy.anchor, policy.admission_paused)

    def test_safe_later_chain_is_held_until_urgent_cohort_completes(self):
        urgent = FakeGroup("urgent", collection=1)
        later = FakeGroup("later", collection=2)
        policy, now, ledger = policy_with_groups([urgent, later])
        policy.pred = SimpleNamespace(estimate=lambda *args, **kwargs: {"work": 10.})
        policy.chain_meta[1]["deadline"] = now + 20.
        policy.chain_meta[2]["deadline"] = now + 100.
        policy.frame_pending = True
        policy.before_schedule([urgent, later], now, ledger)
        self.assertEqual(policy.deadline_deferred, set())
        self.assertEqual(policy.admission_paused, {"2:stage:4"})
        self.assertTrue(policy.custom_dispatch_active())
        policy.running_ids = {later.request_id}
        self.assertFalse(policy.residency_protected(later))
        policy.before_schedule([later], now, ledger)
        self.assertEqual(policy.admission_paused, set())

    def test_safe_cohort_releases_incumbent_only_at_completed_lease_boundary(self):
        urgent = FakeGroup("urgent", collection=1)
        later = FakeGroup("later", collection=2)
        policy, now, ledger = policy_with_groups([urgent, later])
        policy.pred = SimpleNamespace(estimate=lambda *args, **kwargs: {"work": 10.})
        policy.chain_meta[1]["deadline"] = now + 20.
        policy.chain_meta[2]["deadline"] = now + 100.
        policy.model_action = True
        policy.anchor = "2:stage:4"
        policy.remaining = 0.
        policy.pending_action = {"id": 1, "lease_completed": True}
        policy.frame_pending = True
        policy.before_schedule([urgent, later], now, ledger)
        self.assertEqual(policy.admission_paused, {"2:stage:4"})
        self.assertFalse(policy.model_action)
        self.assertFalse(any(e["kind"] == "v2_lease_renewed" for e in policy.audit_events))

    def test_no_model_compound_does_not_activate_permanent_custom_path(self):
        groups = [FakeGroup("a", collection=2, stage=1),
                  FakeGroup("b", collection=2, stage=1)]
        policy, now, ledger = policy_with_groups(groups)
        policy.frame_pending = True
        policy.candidate_selector = None
        policy.before_schedule(groups, now, ledger)
        self.assertTrue(any(len(members) == 2 for members in policy.compound_members.values()))
        self.assertFalse(policy.model_action)
        self.assertFalse(policy.custom_dispatch_active())

    def test_running_protection_progress_is_initialized(self):
        group = FakeGroup("protected")
        policy, _, _ = policy_with_groups([group])
        policy.add_seq_group(group)
        self.assertEqual(policy.decode_progress[group.request_id], 0)
        policy.running_ids = {group.request_id}
        self.assertTrue(policy.residency_protected(group))
        policy.decode_progress[group.request_id] = 4
        group.is_prefill = lambda: False
        self.assertFalse(policy.residency_protected(group))


class OfficialPathTests(unittest.TestCase):
    def test_collective_allocation_matches_official_sum(self):
        policy, _, _ = policy_with_groups([])
        policy._get_interval_time = lambda: 0.01
        peers = [
            SimpleNamespace(predict_output_length=100,
                            seqs=[SimpleNamespace(get_decode_len=lambda: 10)]),
            SimpleNamespace(predict_output_length=200,
                            seqs=[SimpleNamespace(get_decode_len=lambda: 20)]),
            SimpleNamespace(predict_output_length=150,
                            seqs=[SimpleNamespace(get_decode_len=lambda: 30)]),
        ]
        owner = SimpleNamespace(collection_id=7)
        policy.seq_group_dict[7] = peers
        self.assertAlmostEqual(policy.allocation_control(owner), 3.9)

    def test_host_full_preemption_defers_without_recompute_or_swap(self):
        mode = SimpleNamespace(SWAP="SWAP", RECOMPUTE="RECOMPUTE")
        loaded = load_module("_host_recompute_gate", "scheduler_recompute.py", {
            "vllm": module("vllm"), "vllm.core": module("vllm.core"),
            "vllm.core.scheduler": module("vllm.core.scheduler", PreemptionMode=mode),
        })
        class Base:
            def _preempt(self, group, blocks, preemption_mode):
                self.selected_mode = preemption_mode
                return preemption_mode
        class Scheduler(loaded.RecomputeInvalidatingScheduler, Base):
            pass
        scheduler = Scheduler()
        scheduler.block_manager = SimpleNamespace(
            compute_running_seq_group_blocks=lambda groups: 100,
            get_num_free_cpu_blocks=lambda: 0)
        scheduler.cache_config = SimpleNamespace(block_size=16)
        scheduler.v2_ledger = SimpleNamespace(host_capacity=1600, entries={})
        invalidated = []
        scheduler.recompute_invalidate = invalidated.append
        with self.assertRaises(loaded.HostCapacityDefer):
            scheduler._preempt(SimpleNamespace(request_id="hot"), [], mode.SWAP)
        self.assertEqual(invalidated, [])
        scheduler.block_manager.get_num_free_cpu_blocks = lambda: 100
        scheduler.v2_ledger.entries["occupied"] = SimpleNamespace(host=1600)
        with self.assertRaises(loaded.HostCapacityDefer):
            scheduler._preempt(SimpleNamespace(request_id="ledger-hot"), [], mode.SWAP)
        self.assertEqual(invalidated, [])
        scheduler.v2_ledger.entries.clear()
        result = scheduler._preempt(SimpleNamespace(request_id="fits"), [], mode.SWAP)
        self.assertEqual(result, mode.SWAP)
        self.assertEqual(invalidated, [])

    def test_host_defer_returns_empty_outputs_and_restores_official(self):
        loaded = load_scheduler()
        scheduler = object.__new__(loaded.V2SLOScheduler)
        scheduler.running = []
        scheduler.compound_events = []
        events = []
        policy = SimpleNamespace(
            model_action=True, pending_action={"id": 1}, resident_set={"a"},
            resident_order=["a"], recovery_set={"b"}, deadline_deferred={"c"},
            anchor="a", remaining=1.0, frame_pending=True, forced=None,
            forced_action=None, effective_controller="model", last_selected=["a"])
        policy.record = lambda kind, **fields: events.append({"kind": kind, **fields})
        def note_selection(selected, now, scheduled_details=None):
            policy.last_selected = list(selected)
            policy.effective_controller = "official"
        policy.note_selection = note_selection
        scheduler.policy = policy
        metadata, outputs, async_output = scheduler._host_defer_outputs(
            SimpleNamespace(request_id="hot", free_blocks=0, free_tokens=0))
        self.assertEqual(metadata, [])
        self.assertFalse(async_output)
        self.assertEqual(outputs.scheduled_seq_groups, [])
        self.assertEqual(outputs.num_batched_tokens, 0)
        self.assertEqual(outputs.blocks_to_swap_out, [])
        self.assertFalse(policy.model_action)
        self.assertEqual(policy.deadline_deferred, set())
        self.assertEqual(policy.last_selected, [])
        self.assertEqual(policy.effective_controller, "official")
        self.assertEqual(events[0]["kind"], "v2_host_capacity_deferred")
        self.assertEqual(scheduler.compound_events[-1]["kind"], "host_capacity_deferred")

    def test_mixed_queue_compound_is_deferred_to_official_frame(self):
        loaded = load_scheduler()
        scheduler = object.__new__(loaded.V2SLOScheduler)
        running = FakeGroup("a", collection=2, stage=1)
        waiting = FakeGroup("b", collection=2, stage=1)
        scheduler.running = deque([running])
        scheduler.waiting = deque([waiting])
        scheduler.swapped = deque()
        scheduler.policy = SimpleNamespace(compound_members={"2:stage:1": ["a", "b"]})
        self.assertTrue(scheduler._has_compound([running]))
        unit = scheduler._compound_units([running], include_missing=False)[0]
        self.assertEqual([group.request_id for group in unit], ["a"])
        self.assertTrue(scheduler._mixed_queue_unit(unit))

    def test_paused_compound_remains_a_kv_victim_without_execution(self):
        loaded = load_scheduler()
        scheduler = object.__new__(loaded.V2SLOScheduler)
        a = FakeGroup("a", collection=2, stage=1)
        b = FakeGroup("b", collection=2, stage=1)
        scheduler.running = [a, b]
        scheduler.policy = SimpleNamespace(admission_paused={"2:stage:1"},
                                           residency_protected=lambda group: False)
        scheduler._unit_resources = lambda *args: ((2, 2, 0), None)
        scheduler._token_demands = lambda *args: {"a": 1, "b": 1}
        scheduler._resident_blocks = lambda group: 10
        paused = scheduler._planner_units([[a,b]])[0]
        plan = plan_first_step([planner_unit(["viable"], required=20), paused],
                               0, 4, 3, True, 2048)
        self.assertEqual([u["ids"] for u in plan["accepted"]], [["viable"]])
        self.assertEqual(plan["evictions"][0][1], [a,b])
        self.assertEqual(plan["remaining_available_blocks"], 0)

    def test_waiting_compound_enters_process_list_without_pending_swap(self):
        loaded = load_scheduler()
        scheduler = object.__new__(loaded.V2SLOScheduler)
        groups = [FakeGroup("a", collection=2, stage=1),
                  FakeGroup("b", collection=2, stage=1)]
        scheduler.running = deque()
        scheduler.waiting = deque(groups)
        scheduler.swapped = deque()
        scheduler.policy = SimpleNamespace(
            compound_members={"2:stage:1": ["a", "b"]},
            policy_preemption_allowed=lambda: True,
            dispatch_order=lambda current: current)
        process, _, to_activate, _ = scheduler._build_process_list(None, 0)
        self.assertEqual([group.request_id for group in process], ["a", "b"])
        self.assertEqual(len(to_activate), 2)

    def test_trace_5540_forced_frame_executes_and_releases_kv(self):
        result = run_trace_frame()
        self.assertEqual(result["scheduled"], ["chain1-stage4"])
        self.assertEqual(result["chunks"], [2048])
        self.assertEqual(result["budget_tokens"], 2048)
        self.assertEqual(result["preempted"], ["low-b", "low-a"])
        self.assertEqual(result["leftover"], ["retained"])
        self.assertEqual(result["after_schedule_free"], 23)
        self.assertEqual(result["final_free"], 512)
        self.assertEqual(result["event"]["executed_tokens"], [("chain1-stage4", 2048)])

    def test_resident_set_order_reaches_real_planner(self):
        result = run_trace_frame(expected_first=["3:stage:0", "1:stage:4"])
        self.assertEqual(result["event"]["units"][:2],
                         [["low-b"], ["chain1-stage4"]])

    def test_first_dispatch_witness_does_not_override_later_live_order(self):
        result = run_trace_frame(expected_first=["3:stage:0", "1:stage:4"],
                                 already_scheduled=True)
        self.assertEqual(result["event"]["units"][0], ["chain1-stage4"])
        self.assertEqual(result["scheduled"], ["chain1-stage4"])

    def test_resident_set_goal_keys_map_to_request_ids(self):
        result = run_trace_frame(resident_set=["1:stage:4", "4:stage:0"])
        self.assertTrue(result["feasibility"]["feasible"])
        self.assertEqual(set(result["feasibility"]["target"]),
                         {"chain1-stage4", "retained"})

    def test_feasibility_uses_the_same_resident_order_as_execution(self):
        result = run_trace_frame(expected_first=["4:stage:0", "1:stage:4"],
                                 resident_set=["1:stage:4", "4:stage:0"])
        self.assertTrue(result["feasibility"]["feasible"])
        self.assertEqual(result["event"]["units"][:2],
                         [["retained"], ["chain1-stage4"]])

    def test_resident_set_feasibility_requires_every_member(self):
        result = run_trace_frame(resident_set=["chain1-stage4", "missing-member"])
        self.assertFalse(result["feasibility"]["feasible"])

    def test_trace_5540_forced_frame_explicitly_infeasible(self):
        result = run_trace_frame(infeasible=True)
        self.assertEqual(result["scheduled"], [])
        self.assertEqual(result["preempted"], [])
        self.assertEqual(result["budget_tokens"], 0)
        self.assertEqual(result["after_schedule_free"], 10)
        self.assertEqual(result["final_free"], 512)
        self.assertEqual(result["event"]["deferred"][0][1], "kv")

    def test_custom_forced_is_disabled_by_default(self):
        loaded = load_policy()
        policy = loaded.V2CompoundPolicy(SimpleNamespace(decode_seconds=0.01), None, {})
        self.assertFalse(policy.custom_forced_enabled)
        self.assertIsNone(policy._forced_candidate([], None, time.perf_counter()))

    def test_no_model_compound_uses_official_all_paths(self):
        loaded = load_scheduler()
        scheduler = object.__new__(loaded.V2SLOScheduler)
        scheduler.lora_enabled = False
        scheduler.policy = SimpleNamespace(policy_preemption_allowed=lambda: False)
        self.assertEqual(scheduler._schedule_process_list(None, None, None, None, None, False),
                         "official_process_list")
        self.assertEqual(scheduler._schedule_running(None, None), "official_running")
        self.assertEqual(scheduler._schedule_prefills(None, None), "official_prefills")
        self.assertEqual(scheduler._schedule_swapped(None, None), "official_swapped")
        self.assertEqual(scheduler._build_process_list(None, 0), "official_build_process_list")
        self.assertEqual(scheduler._preempt_pending(None), "official_preempt_pending")

    def test_official_nonframe_frame_and_dynamic_adjust_are_restored(self):
        loaded = load_scheduler()
        self.assertIs(loaded.V2SLOScheduler._adjust_batch_limits,
                      loaded.SLOScheduler._adjust_batch_limits)
        scheduler = object.__new__(loaded.V2SLOScheduler)
        scheduler.running = deque()
        scheduler.waiting = deque()
        scheduler.swapped = deque()
        scheduler.compound_events = []
        scheduler.v2_ledger = None
        policy = SimpleNamespace(schedule_interval=50, num_schedule_count=48)
        policy.update_schedule_count = lambda: update_count(policy)
        policy.before_schedule = lambda *args: None
        policy.note_selection = lambda *args: None
        scheduler.policy = policy
        scheduler.schedule()
        self.assertEqual(scheduler.official_path, "official_chunked_prefill")
        self.assertFalse(policy.frame_pending)
        scheduler.schedule()
        self.assertEqual(scheduler.official_path, "official_50_tick")
        self.assertTrue(policy.frame_pending)
        scheduler.schedule()
        scheduler.enough_swap_space = False
        scheduler.schedule()
        self.assertEqual(scheduler.official_path, "official_chunked_prefill")
        self.assertFalse(policy.frame_pending)
        policy.num_schedule_count = 49
        scheduler.enough_swap_space = True
        scheduler.schedule()
        self.assertEqual(scheduler.official_path, "official_50_tick")
        self.assertTrue(policy.frame_pending)

    def test_dynamic_batch_slots_are_not_fixed_four(self):
        loaded = load_scheduler()
        scheduler = object.__new__(loaded.V2SLOScheduler)
        scheduler.max_batch_size = 4
        scheduler.max_swaps_per_iter = 3
        scheduler.running = deque(range(6))
        scheduler.waiting = deque([1, 2])
        scheduler.swapped = deque([3])
        scheduler.policy = SimpleNamespace(compound_members={}, running_ids=set())
        self.assertEqual(scheduler._planning_batch_slots(), 9)


def update_count(policy):
    policy.num_schedule_count += 1
    return policy.num_schedule_count % policy.schedule_interval == 0


def run_cpu_tests():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    status = run_cpu_tests()
    print(json.dumps({"gate": "PASS" if status == 0 else "FAIL",
                      "gpu_used": False, "runtime_vllm": False}))
    raise SystemExit(status)
