import copy
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "programs", ROOT / "v2_src", ROOT / "backend_src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from candidate_evaluator import candidate_action
from conditional_predictor import ConditionalPredictor
from contracts import CacheView, CallView
from execution_reference import forecasts
from deadline_admission import admission_pauses
from execution_view import ExecutionView
from lease_reference import first_plan_for_action, project
from pool_policies import Performance
from v2_contracts import GoalView, ObservedCall


class ToyPerformance:
    decode_seconds = 0.01
    copy_seconds_per_token = 0.01

    def iteration(self, batch_slots, context):
        return self.decode_seconds

    def prefill(self, new, cached=0):
        return max(0, new) * 0.001

    def recovery(self, entry, needed):
        raise ValueError("exercise safe recovery")


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.perf = ToyPerformance()

    @staticmethod
    def predictor(prompt=256, outputs=4, slots=1, future=()):
        record = {"graph_id": "g", "ordinal": 0, "prompt": prompt,
                  "outputs": outputs, "slots": slots, "future": tuple(future)}
        return ConditionalPredictor((record,), 2.0, 0.0, 0.0, ())

    @staticmethod
    def view(count=2, deadlines=None, prompt=256, **kwargs):
        deadlines = deadlines or [1000.0 + index for index in range(count)]
        goals = []
        members = {}
        for index in range(count):
            key = f"{index}:stage:0"
            call = CallView(key, index, 0, prompt, 0, 0, 0, 0.0, 0.0,
                            0.0, 0.0, False, 0.0)
            goals.append(GoalView(index, 0.0, deadlines[index], "READY", (), call))
            members[key] = [key]
        return ExecutionView(0.0, tuple(goals), CacheView(65536, 65536, ()), **kwargs), members

    @staticmethod
    def job(key, compute=0.02, slots=1, **kwargs):
        value = {"key": key, "state": "READY", "deadline": 10.0,
                 "instance": 0, "context": 256, "current": compute,
                 "phases": (("compute", compute),), "slots": slots,
                 "level": 0, "ready_wait": 0.0, "virtual": 0.0,
                 "running": False}
        value.update(kwargs)
        return value

    def test_first_step_uses_native_order_and_defers_hopeless_before_mlfq(self):
        jobs = [self.job("anchor", tokens=1, kv_required=0),
                self.job("urgent", tokens=1, kv_required=0, native_priority=(0, 1), level=3),
                self.job("later", tokens=1, kv_required=0, native_priority=(1, 2), level=0),
                self.job("hopeless", tokens=1, kv_required=0, native_priority=(-1, 0),
                         level=0, deadline_attainable=False)]
        for job in jobs:
            job["custom_forced_enabled"] = False
        plan = first_plan_for_action(jobs, ("anchor", 0.5), 0., self.perf, 0, 8,
                                     {"free_blocks": 100, "watermark_blocks": 0,
                                      "block_size": 16, "max_swaps": 3, "token_budget": 2048})
        self.assertEqual([u["ids"][0] for u in plan["plan"]["accepted"]],
                         ["anchor", "urgent", "later", "hopeless"])

    def test_first_step_pauses_background_but_can_reclaim_its_kv(self):
        jobs = [self.job("viable", tokens=1, kv_required=20),
                self.job("background", tokens=1, kv_required=0, running=True,
                         resident_blocks=20, admission_paused=True, gpu_tokens=320)]
        for job in jobs:
            job["custom_forced_enabled"] = False
        plan = first_plan_for_action(jobs, ("viable", .5), 0., self.perf, 0, 4,
                                     {"free_blocks": 0, "watermark_blocks": 0,
                                      "block_size": 16, "max_swaps": 3, "token_budget": 2048})
        self.assertEqual([u["ids"][0] for u in plan["plan"]["accepted"]], ["viable"])
        self.assertEqual([job["key"] for job in plan["evicted_states"]], ["background"])

    def test_later_deadline_deferral_requires_frame_and_recovery_reserve(self):
        risks = {"urgent": 5., "later": 10.}
        deadlines = {"urgent": 20., "later": 31.}
        kwargs = dict(defer_safe_later=True, frame_reserve=1.,
                      recovery_costs={"later": 1.})
        self.assertEqual(admission_pauses(risks, deadlines, 0., risks, **kwargs), set())
        deadlines["later"] = 32.
        self.assertEqual(admission_pauses(risks, deadlines, 0., risks, **kwargs), {"later"})

    def test_cohort_drains_when_all_urgent_chains_are_no_longer_attainable(self):
        risks = {"urgent": 5., "later": 10.}
        deadlines = {"urgent": 20., "later": 100.}
        self.assertEqual(admission_pauses(risks, deadlines, 30., risks,
                                         defer_hopeless=True, defer_safe_later=True), {"urgent"})
        self.assertEqual(admission_pauses(risks, deadlines, 101., risks,
                                         defer_hopeless=True, defer_safe_later=True), set())

    def test_current_and_future_prefill_enter_timeline(self):
        pred = self.predictor(prompt=3000, outputs=4, future=(
            {"prompt": 1500, "outputs": 6, "slots": 2},))
        view, members = self.view(1, prompt=3000, compound_members=(("0:stage:0", ["m"]),))
        job = forecasts(view, pred, self.perf, conformal=False)[0]
        self.assertEqual(job["prefill_tokens"], 3000)
        self.assertEqual(job["recovery"], 3.0)
        self.assertGreater(job["phases"][0][1], 3.0)
        future = [meta for meta in job["phase_metadata"]
                  if meta["kind"] == "compute" and meta["ordinal"] == 1][0]
        self.assertEqual((future["width"], future["prefill_tokens"]), (2, 1500))
        job.update(instance=0, context=3000, slots=1, running=False)
        score = project([job], ("0:stage:0", 0.04), 0.0, self.perf, 0)
        self.assertAlmostEqual(score["elapsed"], 4.58)
        self.assertGreaterEqual(score["batch_trace"][0]["duration"], 3.0)
        self.assertTrue(any(row["duration"] >= 1.5 for row in score["batch_trace"]))

    def test_future_long_prefill_respects_token_budget(self):
        job = self.job("a", compute=0.01, tokens=1, kv_required=1,
                       resident_blocks=1, gpu_tokens=16, running=True)
        job["phases"] = (("compute", 0.01), ("away", 0.01), ("compute", 3.02))
        job["phase_metadata"] = (
            {"kind": "compute", "width": 1, "prefill_tokens": 0},
            {"kind": "away", "width": 1},
            {"kind": "compute", "width": 1, "prefill_tokens": 3000,
             "prefill_seconds": 3.0, "prompt_tokens": 3000},
        )
        resource = {"free_blocks": 500, "watermark_blocks": 0, "block_size": 16,
                    "max_swaps": 3, "token_budget": 2048, "host_free_blocks": 100}
        score = project([job], ("a", 0.04), 0.0, self.perf, 0,
                        batch_slots=1, resource=resource)
        self.assertEqual([row["tokens"] for row in score["batch_trace"]],
                         [1, 2048, 952, 1, 1])
        self.assertAlmostEqual(score["elapsed"], 3.04)

    def test_away_releases_previous_stage_before_future_prefill(self):
        job = self.job("a", compute=0.01, tokens=1, kv_required=1,
                       resident_blocks=259, gpu_tokens=4129, host_tokens=4129, running=True)
        job["phases"] = (("compute", 0.01), ("away", 0.01), ("compute", 4.156))
        job["phase_metadata"] = (
            {"kind": "compute", "width": 1, "prefill_tokens": 0},
            {"kind": "away", "width": 1},
            {"kind": "compute", "width": 1, "prefill_tokens": 4146,
             "prefill_seconds": 4.146, "prompt_tokens": 4146},
        )
        resource = {"free_blocks": 253, "watermark_blocks": 0, "block_size": 16,
                    "max_swaps": 3, "token_budget": 2048, "host_free_blocks": 100}
        score = project([job], ("a", 0.04), 0.0, self.perf, 0,
                        batch_slots=1, resource=resource)
        self.assertEqual([row["tokens"] for row in score["batch_trace"]],
                         [1, 2048, 2048, 50, 1])
        self.assertTrue(all(0 <= row["free_blocks"] <= 512 for row in score["kv_trace"]))
        self.assertAlmostEqual(score["elapsed"], 4.176)

    def test_compound_member_progress_and_future_width(self):
        pred = self.predictor(prompt=600, outputs=12, slots=2, future=(
            {"prompt": 900, "outputs": 15, "slots": 3},))
        current, _, tail, _, mode = pred.ready_parts(
            600, 0, 2, 7, self.perf, member_progress={"a": 2, "b": 5})
        self.assertEqual(mode, "exact")
        self.assertAlmostEqual(current, 4.0)
        self.assertAlmostEqual(tail[1], 4.0 + 0.9 / 0.01)
        call = CallView("c", 0, 0, 600, 7, 0, 0, 0.0, 0.0,
                        0.0, 0.0, True, 0.0)
        goal = GoalView(0, 0.0, 1000.0, "READY", (), call)
        view = ExecutionView(0.0, (goal,), CacheView(65536, 65536, ()),
                             compound_members=(("c", ["a", "b"]),),
                             compound_progress=(("c", (("a", 2), ("b", 5))),))
        job = forecasts(view, pred, self.perf, conformal=False)[0]
        self.assertEqual(job["support"]["member_progress"], "observed")
        self.assertEqual(job["phase_metadata"][-1]["width"], 3)
        self.assertEqual(job["phase_metadata"][-1]["ordinal"], 1)

    def test_prediction_modes_and_away_alignment(self):
        records = tuple({"graph_id": str(index), "ordinal": 0, "prompt": prompt,
                         "outputs": 8, "slots": 1,
                         "future": ({"prompt": 800, "outputs": 12, "slots": 2},)}
                        for index, prompt in enumerate((500, 700)))
        pred = ConditionalPredictor(records, 3.0, 0.0, 0.0, ())
        self.assertEqual(pred.ready_parts(600, 0, 1, 0, self.perf)[-1], "backoff")
        self.assertEqual(pred.ready_parts(900, 0, 1, 0, self.perf)[-1], "extrapolated")
        history = (ObservedCall(0, 500, 8, 1.0, 1.0, 0.0),)
        goal = GoalView(0, 0.0, 1000.0, "AWAY", history,
                        tool_entered=1.0, tool_elapsed=0.01)
        estimate = pred.estimate(goal, self.perf, conformal=False)
        self.assertEqual(estimate["support"]["phase"], "AWAY")
        self.assertEqual(estimate["phase_metadata"][0]["ordinal"], 1)
        self.assertEqual(estimate["phase_metadata"][0]["width"], 2)
        self.assertEqual(estimate["phase_metadata"][1]["prefill_tokens"], 800)
        view = ExecutionView(1.0, (goal,), CacheView(65536, 65536, ()))
        job = forecasts(view, pred, self.perf, conformal=False)[0]
        self.assertEqual(job["recovery"], 0.0)

    def test_observed_member_outputs_use_critical_path(self):
        record = {"graph_id": "members", "ordinal": 0, "prompt": 600,
                  "outputs": 12, "member_outputs": (3, 9), "slots": 2, "future": ()}
        pred = ConditionalPredictor((record,), 2.0, 0.0, 0.0, ())
        current, _, _, _, _ = pred.ready_parts(
            600, 0, 2, 0, self.perf, member_progress={"a": 0, "b": 0})
        self.assertEqual(current, 9.0)

    def test_baseline_timeout_is_safe_and_not_repeated(self):
        view, members = self.view(2)
        calls = []

        def timeout(*args, **kwargs):
            calls.append(1)
            raise TimeoutError("baseline")

        cache = {}
        action, audit = candidate_action(view, 0, set(members), self.predictor(), self.perf,
                                         {0: 0, 1: 0}, compound_members=members,
                                         project_fn=timeout, evaluation_cache=cache)
        self.assertEqual(action[0], audit["pi0"])
        self.assertEqual(audit["fallback"], "baseline_evaluation_timeout")
        self.assertFalse(audit["actions"])
        _, repeated = candidate_action(view, 0, set(members), self.predictor(), self.perf,
                                       {0: 0, 1: 0}, compound_members=members,
                                       project_fn=timeout, evaluation_cache=cache)
        self.assertEqual(repeated["fallback"], "baseline_timeout_cache")
        self.assertEqual(len(calls), 1)

    def test_dynamic_slots_swap_resume_and_member_progress(self):
        jobs = [
            self.job("a", running=True, tokens=1, kv_required=0,
                     resident_blocks=16, gpu_tokens=256, host_tokens=0),
            self.job("c", compute=0.01, tokens=1, kv_required=20,
                     resident_blocks=0, gpu_tokens=0, host_tokens=0),
        ]
        resource = {"free_blocks": 4, "watermark_blocks": 0, "block_size": 16,
                    "max_swaps": 3, "token_budget": 2048, "host_free_blocks": 100}
        score = project(jobs, ("c", 0.04), 0.0, self.perf, 0,
                        batch_slots=1, resource=resource)
        self.assertAlmostEqual(score["switch_cost"], 2.56)
        self.assertAlmostEqual(score["resume_cost"], 2.56)
        self.assertGreaterEqual(score["finishes"]["c"], 2.57)
        self.assertEqual(score["final_progress"]["a"]["decode"], 2)
        self.assertTrue(score["kv_trace"])
        self.assertTrue(all(state["gpu"] == 0 and state["host"] == 0
                            for state in score["kv_trace"][-1]["residency"].values()
                            if state["blocks"] == 0))

        compound = self.job("wide", slots=2,
                            member_progress=(("m0", 0), ("m1", 0)))
        score = project([self.job("a", compute=0.01), compound],
                        ("a", 0.04), 0.0, self.perf, 0, batch_slots=[1, 2])
        self.assertEqual(score["max_batch_slots"], 2)
        self.assertEqual(score["final_progress"]["wide"]["members"], {"m0": 2, "m1": 2})
        required = {"keys", "slots", "duration", "lease_after"}
        self.assertTrue(all(required <= row.keys() for row in score["batch_trace"]))
        self.assertTrue(all(row["slots"] <= row["slot_limit"] for row in score["batch_trace"]))

    def test_first_plan_prefill_can_shrink_around_decode_members(self):
        jobs = [
            self.job("a", running=True, tokens=1, kv_required=1,
                     resident_blocks=10, native_priority=(0, 0)),
            self.job("b", running=True, tokens=1, kv_required=1,
                     resident_blocks=10, native_priority=(1, 0)),
            self.job("p", running=False, tokens=2048, kv_required=170,
                     resident_blocks=0, native_priority=(2, 0)),
        ]
        resource = {"free_blocks": 500, "watermark_blocks": 0, "block_size": 16,
                    "max_swaps": 3, "token_budget": 2048, "host_free_blocks": 100}
        result = first_plan_for_action(jobs, ("a", 0.04), 0.0, self.perf, 0, 3, resource)
        accepted = {unit["ids"][0]: unit for unit in result["plan"]["accepted"]}
        self.assertEqual(set(accepted), {"a", "b", "p"})
        self.assertEqual(accepted["p"]["tokens"], 2046)
        self.assertEqual(result["plan"]["used_tokens"], 2048)

    def test_model_eviction_moves_resident_and_recomputes_when_host_is_full(self):
        def jobs():
            return [
                self.job("a", running=True, tokens=1, kv_required=1,
                         resident_blocks=100, gpu_tokens=1600, native_priority=(1, 0)),
                self.job("b", running=True, tokens=1, kv_required=1,
                         resident_blocks=100, gpu_tokens=1600, native_priority=(0, 0)),
            ]
        resource = {"free_blocks": 300, "watermark_blocks": 0, "block_size": 16,
                    "max_swaps": 3, "token_budget": 2048, "host_free_blocks": 400}
        score = project(jobs(), ("a", 0.04), 0.0, self.perf, 0,
                        batch_slots=1, resource=resource)
        self.assertEqual(score["kv_trace"][0]["residency"]["b"]["blocks"], 0)
        self.assertEqual(score["kv_trace"][0]["residency"]["b"]["host"], 1600)
        self.assertAlmostEqual(score["batch_trace"][0]["switch_cost"], 16.0)
        resource["host_free_blocks"] = 0
        recovered = project(jobs(), ("a", 0.04), 0.0, self.perf, 0,
                            batch_slots=1, resource=resource)
        self.assertGreater(recovered["recompute_seconds"], 0.0)
        self.assertTrue(all(row["free_blocks"] >= 0 for row in recovered["kv_trace"]))
        self.assertTrue(any(row["tokens"] >= 256 for row in recovered["batch_trace"]))

    def test_deadline_rescue_prefers_attainable_chain_without_rollout(self):
        view, members = self.view(2, deadlines=[0.005, 1.0])
        resources = {
            goal.ready.key: {"slots": 1, "tokens": 1, "required_blocks": 1,
                             "resident_blocks": 1, "status": "RUNNING", "gpu_tokens": 16}
            for goal in view.goals
        }
        view = replace(view, resource_units=tuple(resources.items()),
                       free_kv_blocks=100, watermark_blocks=0,
                       native_priorities=tuple((key, (0, index))
                                               for index, key in enumerate(sorted(resources))))
        def forbidden(*args, **kwargs):
            raise AssertionError("deadline rescue must not run full rollout")
        action, audit = candidate_action(view, 0, set(members), self.predictor(), self.perf,
                                         {0: 0, 1: 0}, compound_members=members,
                                         project_fn=forbidden, deadline_rescue=True)
        self.assertEqual(action[0], "1:stage:0")
        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["heuristic"], "deadline_rescue")
        self.assertEqual(audit["actions"][0]["good"], 1)
        self.assertIn("1:stage:0", audit["actions"][0]["batch_trace"][0]["keys"])

    def test_rescue_only_cap_does_not_run_rollout_without_rescue(self):
        view, members = self.view(2, deadlines=[1.0, 1.0])
        resources = {
            goal.ready.key: {"slots": 1, "tokens": 1, "required_blocks": 1,
                             "resident_blocks": 1, "status": "RUNNING", "gpu_tokens": 16}
            for goal in view.goals
        }
        view = replace(view, resource_units=tuple(resources.items()),
                       free_kv_blocks=100, watermark_blocks=0,
                       native_priorities=tuple((key, (0, index))
                                               for index, key in enumerate(sorted(resources))))
        def forbidden(*args, **kwargs):
            raise AssertionError("rescue-only cap must not run full rollout")
        action, audit = candidate_action(view, 0, set(members), self.predictor(), self.perf,
                                         {0: 0, 1: 0}, compound_members=members,
                                         project_fn=forbidden, deadline_rescue=True,
                                         deadline_rescue_only=True)
        self.assertEqual(action[0], "0:stage:0")
        self.assertFalse(audit["accepted"])
        self.assertEqual(audit["fallback"], "conservative_evaluation_cap")
        self.assertEqual(audit["project_calls"], 0)

    def test_calibrated_rescue_risk_margin_and_frame_lease(self):
        view, members = self.view(2, deadlines=[0.005, 0.5], schedule_count=101,
                                  official_frame=True)
        resources = {
            goal.ready.key: {"slots": 1, "tokens": 1, "required_blocks": 1,
                             "resident_blocks": 1, "status": "RUNNING", "gpu_tokens": 16}
            for goal in view.goals
        }
        view = replace(view, resource_units=tuple(resources.items()),
                       free_kv_blocks=100, watermark_blocks=0,
                       native_priorities=tuple((key, (0, index))
                                               for index, key in enumerate(sorted(resources))))
        pred = replace(self.predictor(), risk_q=1.0)
        def forbidden(*args, **kwargs):
            raise AssertionError("calibrated rescue must not run full rollout")
        action, audit = candidate_action(view, 0, set(members), pred, self.perf,
                                         {0: 0, 1: 0}, compound_members=members,
                                         project_fn=forbidden, deadline_rescue=True)
        self.assertEqual(action[0], "1:stage:0")
        self.assertAlmostEqual(action[1], 0.49)
        self.assertEqual(audit["risk_q"], 1.0)
        self.assertEqual(audit["lease_steps"], 49)
        self.assertAlmostEqual(audit["actions"][0]["batch_trace"][0]["lease_after"], 0.48)
        risky = replace(view, goals=(view.goals[0], replace(view.goals[1], deadline=0.3)))
        action, rejected = candidate_action(risky, 0, set(members), pred, self.perf,
                                            {0: 0, 1: 0}, compound_members=members,
                                            project_fn=forbidden, deadline_rescue=True,
                                            deadline_rescue_only=True)
        self.assertEqual(action[0], "0:stage:0")
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["project_calls"], 0)

    def test_rescue_uses_earliest_deadline_instead_of_shortest_work(self):
        class Predictor:
            risk_q = 0.0
            def estimate(self, goal, profile, conformal, **kwargs):
                work = 2.0 if goal.program == 1 else 1.0
                return {"work": work, "point": work, "current": work,
                        "phases": (("compute", work),), "phase_metadata": (),
                        "support": {}}
        view, members = self.view(3, deadlines=[0.005, 10.0, 11.0], official_frame=True)
        resources = {
            goal.ready.key: {"slots": 1, "tokens": 1, "required_blocks": 1,
                             "resident_blocks": 1, "status": "RUNNING", "gpu_tokens": 16}
            for goal in view.goals
        }
        view = replace(view, resource_units=tuple(resources.items()),
                       free_kv_blocks=100, watermark_blocks=0,
                       native_priorities=tuple((key, (0, index))
                                               for index, key in enumerate(sorted(resources))))
        action, audit = candidate_action(view, 0, set(members), Predictor(), self.perf,
                                         {index: 0 for index in range(3)},
                                         compound_members=members, deadline_rescue=True)
        self.assertEqual(action[0], "1:stage:0")
        self.assertEqual(audit["selection_rule"], "earliest_deadline_first")
        self.assertTrue(audit["accepted"])

    def test_incumbent_rescue_uses_least_laxity(self):
        class Predictor:
            risk_q = 0.0
            def estimate(self, goal, profile, conformal, **kwargs):
                work = 1.0 if goal.program == 1 else 2.0
                return {"work": work, "point": work, "current": work,
                        "phases": (("compute", work),), "phase_metadata": (),
                        "support": {}}
        view, members = self.view(3, deadlines=[0.005, 10.0, 11.0], official_frame=True,
                                  incumbent_action=("0:stage:0", 0.0))
        resources = {
            goal.ready.key: {"slots": 1, "tokens": 1, "required_blocks": 1,
                             "resident_blocks": 1, "status": "RUNNING", "gpu_tokens": 16}
            for goal in view.goals
        }
        view = replace(view, resource_units=tuple(resources.items()),
                       free_kv_blocks=100, watermark_blocks=0,
                       native_priorities=tuple((key, (0, index))
                                               for index, key in enumerate(sorted(resources))))
        action, audit = candidate_action(view, 0, set(members), Predictor(), self.perf,
                                         {index: 0 for index in range(3)},
                                         compound_members=members, deadline_rescue=True)
        self.assertEqual(action[0], "2:stage:0")
        self.assertEqual(audit["selection_rule"], "incumbent_least_laxity")
        self.assertTrue(audit["accepted"])

    def test_rescue_allows_eviction_of_hopeless_chain(self):
        view, members = self.view(3, deadlines=[0.005, 10.0, 11.0], official_frame=True)
        goals = tuple(replace(goal, ready=replace(goal.ready, running=True))
                      for goal in view.goals)
        resources = {
            goal.ready.key: {"slots": 1, "tokens": 1, "required_blocks": 1,
                             "resident_blocks": 100, "status": "RUNNING", "gpu_tokens": 1600}
            for goal in goals
        }
        view = replace(view, goals=goals, resource_units=tuple(resources.items()),
                       free_kv_blocks=0, watermark_blocks=0,
                       native_priorities=tuple((key, (0, index))
                                               for index, key in enumerate(sorted(resources))))
        action, audit = candidate_action(view, 0, set(members), self.predictor(), self.perf,
                                         {index: 0 for index in range(3)}, compound_members=members,
                                         deadline_rescue=True, deadline_rescue_only=True)
        self.assertEqual(action[0], "1:stage:0")
        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["evicted_hopeless"], ["0:stage:0"])
        self.assertFalse(audit["resource_skips"])

    def test_rescue_rejects_attainable_chain_eviction(self):
        view, members = self.view(3, deadlines=[0.005, 10.0, 11.0], official_frame=True)
        goals = tuple(replace(goal, ready=replace(goal.ready, running=True))
                      for goal in view.goals)
        resources = {
            goal.ready.key: {"slots": 1, "tokens": 1, "required_blocks": 1,
                             "resident_blocks": 100, "status": "RUNNING", "gpu_tokens": 1600,
                             "protected": goal.program == 0}
            for goal in goals
        }
        view = replace(view, goals=goals, resource_units=tuple(resources.items()),
                       free_kv_blocks=0, watermark_blocks=0,
                       native_priorities=tuple((key, (0, index))
                                               for index, key in enumerate(sorted(resources))))
        action, audit = candidate_action(view, 0, set(members), self.predictor(), self.perf,
                                         {index: 0 for index in range(3)}, compound_members=members,
                                         deadline_rescue=True, deadline_rescue_only=True)
        self.assertEqual(action[0], "0:stage:0")
        self.assertFalse(audit["accepted"])
        self.assertEqual(audit["resource_skips"][0]["reason"], "rescue_evicts_attainable_chain")
        self.assertEqual(audit["resource_skips"][0]["protected_evictions"], ["2:stage:0"])

    def test_rescue_allows_safe_later_deadline_eviction(self):
        view, members = self.view(3, deadlines=[0.005, 10.0, 11.0], official_frame=True)
        goals = tuple(replace(goal, ready=replace(goal.ready, running=True))
                      for goal in view.goals)
        resources = {
            goal.ready.key: {"slots": 1, "tokens": 1, "required_blocks": 1,
                             "resident_blocks": 100, "status": "RUNNING",
                             "gpu_tokens": 1 if goal.program == 2 else 1600,
                             "protected": goal.program == 0}
            for goal in goals
        }
        view = replace(view, goals=goals, resource_units=tuple(resources.items()),
                       free_kv_blocks=0, watermark_blocks=0,
                       native_priorities=tuple((key, (0, index))
                                               for index, key in enumerate(sorted(resources))))
        action, audit = candidate_action(view, 0, set(members), self.predictor(), self.perf,
                                         {index: 0 for index in range(3)}, compound_members=members,
                                         deadline_rescue=True, deadline_rescue_only=True)
        self.assertEqual(action[0], "1:stage:0")
        self.assertTrue(audit["accepted"])
        self.assertIn("2:stage:0", audit["evicted_attainable"])
        self.assertFalse(audit["resource_skips"])

    def test_rescue_allows_earlier_deadline_eviction_when_recovery_is_safe(self):
        class Predictor:
            risk_q = 0.0
            def estimate(self, goal, profile, conformal, **kwargs):
                work = 1.0 if goal.program == 1 else 2.0
                return {"work": work, "point": work, "current": work,
                        "phases": (("compute", work),), "phase_metadata": (),
                        "support": {}}
        view, members = self.view(3, deadlines=[0.005, 10.0, 11.0], official_frame=True,
                                  incumbent_action=("0:stage:0", 0.0))
        goals = tuple(replace(goal, ready=replace(goal.ready, running=True))
                      for goal in view.goals)
        resources = {
            goal.ready.key: {"slots": 1, "tokens": 1, "required_blocks": 1,
                             "resident_blocks": 100, "status": "RUNNING",
                             "gpu_tokens": 1 if goal.program == 1 else 1600,
                             "protected": goal.program == 0}
            for goal in goals
        }
        view = replace(view, goals=goals, resource_units=tuple(resources.items()),
                       free_kv_blocks=0, watermark_blocks=0,
                       native_priorities=tuple((key, (0, index))
                                               for index, key in enumerate(sorted(resources))))
        action, audit = candidate_action(view, 0, set(members), Predictor(), self.perf,
                                         {index: 0 for index in range(3)}, compound_members=members,
                                         deadline_rescue=True, deadline_rescue_only=True)
        self.assertEqual(action[0], "2:stage:0")
        self.assertTrue(audit["accepted"])
        self.assertIn("1:stage:0", audit["evicted_attainable"])
        self.assertFalse(audit["resource_skips"])

    def test_dynamic_performance_observation_updates_large_batch_factor(self):
        profile = Performance({
            "decode_anchor": 0.03, "copy_seconds_per_token": 0.00001,
            "prefill_coefficients": [0.0, 0.001, 0.0, 0.0, 0.0, 0.0],
            "decode_coefficients": [0.01, 0.001, 0.00001],
        })
        before = profile.iteration(11, 1000)
        observation = profile.observe(11, 1000, before * 1.5)
        self.assertEqual(observation["B"], 11)
        self.assertGreater(observation["new_factor"], 1.0)
        self.assertGreater(profile.iteration(11, 1000), before)

    def test_rollout_respects_custom_forced_disablement(self):
        def jobs(enabled):
            rows = [
                self.job("a", running=True, tokens=1, kv_required=1,
                         resident_blocks=10, native_priority=(0, 0), ready_wait=100.0),
                self.job("b", running=True, tokens=1, kv_required=1,
                         resident_blocks=10, native_priority=(1, 0), ready_wait=100.0),
            ]
            for row in rows:
                row["custom_forced_enabled"] = enabled
            return rows
        resource = {"free_blocks": 300, "watermark_blocks": 0, "block_size": 16,
                    "max_swaps": 3, "token_budget": 2048, "host_free_blocks": 400}
        disabled = project(jobs(False), ("a", 0.04), 0.0, self.perf, 0,
                           batch_slots=1, resource=resource)
        self.assertIsNone(disabled["batch_trace"][0]["forced"])
        enabled = project(jobs(True), ("a", 0.04), 0.0, self.perf, 0,
                          batch_slots=1, resource=resource)
        self.assertEqual(enabled["batch_trace"][0]["forced"], "b")

    def test_evaluation_cost_changes_finish_and_good(self):
        jobs = [self.job("a", deadline=0.03), self.job("b", compute=0.01)]
        free = project(jobs, ("a", 0.04), 0.0, self.perf, 0, batch_slots=1)
        charged = project(jobs, ("a", 0.04), 0.0, self.perf, 0,
                          batch_slots=1, evaluation_cost=0.05)
        self.assertEqual(free["good"], 2)
        self.assertEqual(charged["good"], 1)
        for key in free["finishes"]:
            self.assertAlmostEqual(charged["finishes"][key], free["finishes"][key] + 0.05)
        self.assertAlmostEqual(charged["elapsed"], free["elapsed"] + 0.05)

    def test_baseline_on_time_chain_is_protected(self):
        view, members = self.view(2, deadlines=[1.5, 2.5])
        pred = self.predictor()

        def synthetic(jobs, action, *args, **kwargs):
            baseline = action[0] == "0:stage:0"
            finishes = {job["key"]: (1.0 if job["key"] == "0:stage:0" else 4.0)
                        for job in jobs} if baseline else {
                            job["key"]: 2.0 for job in jobs}
            return {"good": 1, "tardiness": 0.0, "mean": 2.0,
                    "finishes": finishes, "switch_cost": 0.0,
                    "batch_trace": [{"keys": [action[0]], "slots": 1,
                                     "duration": 0.01, "lease_after": 0.03}]}

        action, audit = candidate_action(view, 0, set(members), pred, self.perf,
                                         {0: 0, 1: 0}, compound_members=members,
                                         project_fn=synthetic, synchronous_evaluation=False)
        self.assertEqual(action[0], "0:stage:0")
        self.assertFalse(audit["accepted"])
        self.assertEqual(audit["fallback_diagnosis"], "baseline_on_time_protection")
        self.assertTrue(any(item["baseline_protection_violation"]
                            for item in audit["actions"][1:]))

    def test_best_of_all_evaluated_and_partial_timeout(self):
        view, members = self.view(3)
        pred = self.predictor()

        def synthetic(jobs, action, *args, **kwargs):
            index = int(action[0].split(":")[0])
            finish = (100.0, 90.0, 80.0)[index]
            return {"good": 0, "tardiness": 0.0, "mean": finish,
                    "finishes": {job["key"]: finish for job in jobs},
                    "switch_cost": 0.0,
                    "batch_trace": [{"keys": [action[0]], "slots": 1,
                                     "duration": 0.01, "lease_after": 0.03}]}

        action, audit = candidate_action(view, 0, set(members), pred, self.perf,
                                         {index: 0 for index in range(3)},
                                         compound_members=members, project_fn=synthetic,
                                         synchronous_evaluation=False)
        self.assertEqual(action[0], "2:stage:0")
        self.assertTrue(audit["accepted"])

        calls = []

        def flaky(jobs, action, *args, **kwargs):
            calls.append(action)
            if len(calls) == 2:
                raise TimeoutError("one candidate")
            return synthetic(jobs, action, *args, **kwargs)

        cache = {}
        _, partial = candidate_action(view, 0, set(members), pred, self.perf,
                                      {index: 0 for index in range(3)},
                                      compound_members=members, project_fn=flaky,
                                      evaluation_seconds=10.0, evaluation_cache=cache,
                                      synchronous_evaluation=False)
        first_calls = len(calls)
        self.assertEqual(partial["timeouts"], 1)
        self.assertGreaterEqual(len(partial["actions"]), 2)
        self.assertEqual(partial["selection_scope"], "evaluated_actions_partial")
        _, repeated = candidate_action(view, 0, set(members), pred, self.perf,
                                       {index: 0 for index in range(3)},
                                       compound_members=members, project_fn=flaky,
                                       evaluation_seconds=10.0, evaluation_cache=cache,
                                       synchronous_evaluation=False)
        self.assertEqual(len(calls), first_calls)
        self.assertEqual(repeated["timeout_cache_hits"], 1)

    def test_full_state_cache_and_no_admit_degenerate(self):
        view, members = self.view(2)
        pred = self.predictor()
        calls = []

        def synthetic(jobs, action, *args, **kwargs):
            calls.append(action)
            return {"good": 2, "tardiness": 0.0, "mean": 1.0,
                    "finishes": {job["key"]: 1.0 for job in jobs},
                    "switch_cost": 0.0,
                    "batch_trace": [{"keys": [action[0]], "slots": 1,
                                     "duration": 0.01, "lease_after": 0.03}]}

        cache = {}
        _, first = candidate_action(view, 0, set(members), pred, self.perf,
                                    {0: 0, 1: 0}, compound_members=members,
                                    project_fn=synthetic, evaluation_cache=cache,
                                    synchronous_evaluation=False)
        count = len(calls)
        self.assertGreater(count, 0)
        self.assertNotEqual(first["preflight_reason"], "all_ready_units_fit_fixed_batch")
        _, second = candidate_action(view, 0, set(members), pred, self.perf,
                                     {0: 0, 1: 0}, compound_members=members,
                                     project_fn=synthetic, evaluation_cache=cache,
                                     synchronous_evaluation=False)
        self.assertEqual(len(calls), count)
        self.assertGreater(second["cache_hits"], 0)
        changed = copy.deepcopy(view)
        object.__setattr__(changed, "now", 0.5)
        candidate_action(changed, 0, set(members), pred, self.perf,
                         {0: 0, 1: 0}, compound_members=members,
                         project_fn=synthetic, evaluation_cache=cache,
                         synchronous_evaluation=False)
        self.assertGreater(len(calls), count)

    def test_official_frame_gate(self):
        view, members = self.view(2, schedule_count=1, schedule_interval=50)
        pred = self.predictor()

        def forbidden(*args, **kwargs):
            raise AssertionError("non-frame rollout")

        action, audit = candidate_action(view, 0, set(members), pred, self.perf,
                                         {0: 0, 1: 0}, compound_members=members,
                                         project_fn=forbidden)
        self.assertEqual(action[0], audit["pi0"])
        self.assertEqual(audit["fallback"], "non_official_frame")
        official = ExecutionView(view.now, view.goals, view.cache,
                                 schedule_count=50, schedule_interval=50)
        calls = []

        def synthetic(jobs, action, *args, **kwargs):
            calls.append(action)
            return {"good": 2, "tardiness": 0.0, "mean": 1.0,
                    "finishes": {job["key"]: 1.0 for job in jobs},
                    "batch_trace": [{"keys": [action[0]], "slots": 1,
                                     "duration": 0.01, "lease_after": 0.03}]}

        _, audit = candidate_action(official, 0, set(members), pred, self.perf,
                                    {0: 0, 1: 0}, compound_members=members,
                                    project_fn=synthetic, synchronous_evaluation=False)
        self.assertNotEqual(audit["fallback"], "non_official_frame")
        self.assertTrue(calls)
        self.assertTrue(all(horizon <= self.perf.decode_seconds
                            or horizon == 4 * self.perf.decode_seconds
                            for _, horizon in calls))
        self.assertEqual(audit["frame_lease_cap"], self.perf.decode_seconds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
