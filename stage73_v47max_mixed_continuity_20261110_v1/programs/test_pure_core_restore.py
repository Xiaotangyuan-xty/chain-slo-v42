"""Native trace equivalence between restored pure mode and frozen v47."""
from copy import copy
import unittest
from unittest.mock import patch

from vllm.config import CacheConfig
from vllm.sequence import SequenceStatus
from frozen_v47.native_chain import NativeChainScheduler as FrozenV47
from mixed_joint import MixedScheduler
from jitserve.request_info import RequestInfo, RequestType
from vllm import SamplingParams
from vllm.sequence import Sequence, SequenceGroup, Logprob


class PureCoreTraceTests(unittest.TestCase):
    def setUp(self):
        self.wall = patch('time.time', return_value=100.)
        self.clock = patch('time.perf_counter', return_value=1000.)
        self.wall.start()
        self.clock.start()
        self.addCleanup(self.wall.stop)
        self.addCleanup(self.clock.stop)

    def scheduler(self, blocks=64, slots=4):
        from vllm.config import SchedulerConfig
        self.shared = SchedulerConfig('generate', 64, slots, 512,
            enable_chunked_prefill=True, policy='jitserve', top_k_selection=3,
            max_swaps_per_iter=3)
        cache = CacheConfig(16, .9, 1, 'auto')
        cache.num_gpu_blocks = blocks
        cache.num_cpu_blocks = 64
        return MixedScheduler(self.shared, cache, None)

    def group(self, rid, kind=2, prompt=64, cid=None, stage=0, width=1, deadline=2000.):
        info = RequestInfo(RequestType(kind), (2., .1, 20.), 0,
            rid if cid is None else cid, 20, prompt, 100,
            stage_id=stage, request_id=0)
        if kind == 2:
            info.chain_deadline = deadline
            info.chain_stage_count = stage + 1
            info.chain_stage_widths = [1] * stage + [width]
        seq = Sequence(rid, inputs={'prompt_token_ids': list(range(prompt)),
                                   'prompt': 'CPU'}, block_size=16)
        return SequenceGroup(str(rid), [seq], 100., info,
                             SamplingParams(temperature=0, max_tokens=100))

    def step(self, scheduler):
        _, out, _ = scheduler.schedule()
        for item in out.scheduled_seq_groups:
            group = item.seq_group
            group.update_num_computed_tokens(item.token_chunk_size)
            if not group.is_prefill():
                group.first_seq.append_token_id(1, {1: Logprob(0.)})
        return out

    def cleanup(self, scheduler):
        scheduler.abort_seq_group([g.request_id for queue in
            (scheduler.running, scheduler.waiting, scheduler.swapped) for g in queue])
        self.assertEqual(scheduler.block_manager.get_num_free_gpu_blocks(),
                         scheduler.block_manager.num_total_gpu_blocks)

    def reference(self, blocks=64, slots=4):
        cache = CacheConfig(16, .9, 1, 'auto')
        cache.num_gpu_blocks = blocks
        cache.num_cpu_blocks = 64
        return FrozenV47(copy(self.shared), cache, None)

    def add_stage(self, scheduler, groups, cid, widths, sid, *, deadline=2000.,
                  prompt=64, application='dr'):
        width = widths[sid]
        for member in range(width):
            rid = f'{cid}:{sid}:{member}'
            group = self.group(rid, 2, prompt=prompt, cid=cid, stage=sid,
                               width=width, deadline=deadline)
            info = group.request_info
            info.request_id = member
            info.chain_stage_count = len(widths)
            info.chain_stage_widths = widths
            info.application = application
            info.chain_future_max_tokens = 256 if application == 'tot_native' else 1024
            scheduler.add_seq_group(group)
            groups[rid] = group

    def compare(self, reference, candidate, groups_ref, groups_candidate, out_ref, out_candidate):
        self.assertEqual(reference.selected, candidate.selected)
        self.assertEqual(reference.leases, candidate.leases)
        self.assertEqual(reference.policy.draining, candidate.policy.draining)
        self.assertEqual(reference.last_plan, candidate.last_plan)
        self.assertEqual(reference.fast_expired_drain, candidate.fast_expired_drain)
        self.assertEqual(reference.scheduler_config.max_num_seqs,
                         candidate.scheduler_config.max_num_seqs)
        for a, b in [(reference.running, candidate.running),
                     (reference.waiting, candidate.waiting),
                     (reference.swapped, candidate.swapped)]:
            self.assertEqual([g.request_id for g in a], [g.request_id for g in b])
        self.assertEqual(out_ref.preempted, out_candidate.preempted)
        self.assertEqual([x.seq_group.request_id for x in out_ref.scheduled_seq_groups],
                         [x.seq_group.request_id for x in out_candidate.scheduled_seq_groups])
        self.assertEqual([x.token_chunk_size for x in out_ref.scheduled_seq_groups],
                         [x.token_chunk_size for x in out_candidate.scheduled_seq_groups])
        for field in ['blocks_to_swap_in', 'blocks_to_swap_out']:
            a, b = getattr(out_ref, field), getattr(out_candidate, field)
            self.assertEqual(len(a), len(b))
        for rid in groups_ref:
            a, b = groups_ref[rid], groups_candidate[rid]
            self.assertEqual(a.first_seq.get_output_len(), b.first_seq.get_output_len())
            self.assertEqual(a.first_seq.data.get_num_computed_tokens(),
                             b.first_seq.data.get_num_computed_tokens())
            self.assertEqual(a.first_seq.status, b.first_seq.status)

    def run_pair(self, candidate, reference, widths_by_root, *, prompt=64,
                 output_target=6, drain_after=12):
        groups_ref, groups_candidate = {}, {}
        for cid, widths in enumerate(widths_by_root):
            self.add_stage(reference, groups_ref, cid, widths, 0, prompt=prompt,
                           application='tot_native' if len(widths) == 4 else 'dr')
            self.add_stage(candidate, groups_candidate, cid, widths, 0, prompt=prompt,
                           application='tot_native' if len(widths) == 4 else 'dr')
        added_ref = {cid: 0 for cid in range(len(widths_by_root))}
        added_candidate = dict(added_ref)
        for tick in range(240):
            out_ref = self.step(reference)
            out_candidate = self.step(candidate)
            self.compare(reference, candidate, groups_ref, groups_candidate, out_ref, out_candidate)
            for groups, scheduler, added in [(groups_ref, reference, added_ref),
                                             (groups_candidate, candidate, added_candidate)]:
                for group in groups.values():
                    if group.first_seq.get_output_len() >= output_target:
                        group.first_seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
                scheduler.free_finished_seq_groups()
                for cid, widths in enumerate(widths_by_root):
                    live = [g for g in scheduler.running + scheduler.waiting + scheduler.swapped
                            if g.collection_id == cid]
                    current_stage = added[cid]
                    if not live and current_stage + 1 < len(widths):
                        self.add_stage(scheduler, groups, cid, widths, current_stage + 1,
                                       prompt=prompt,
                                       application='tot_native' if len(widths) == 4 else 'dr')
                        added[cid] += 1
            if tick + 1 == drain_after:
                deadline_patch = patch('time.perf_counter', return_value=3001.)
                deadline_patch.start()
                self.addCleanup(deadline_patch.stop)
            if not (reference.running or reference.waiting or reference.swapped):
                if not (candidate.running or candidate.waiting or candidate.swapped):
                    break
        self.compare(reference, candidate, groups_ref, groups_candidate, out_ref, out_candidate)
        self.assertFalse(reference.running or reference.waiting or reference.swapped)
        self.assertFalse(candidate.running or candidate.waiting or candidate.swapped)
        self.cleanup(reference)
        self.cleanup(candidate)
        self.assertEqual(reference.block_manager.get_num_free_cpu_blocks(), 64)
        self.assertEqual(candidate.block_manager.get_num_free_cpu_blocks(), 64)
        self.assertGreaterEqual(tick, drain_after)

    def test_pure_dr_trace_equivalence_with_swap_recompute_and_drain(self):
        candidate = self.scheduler(blocks=20, slots=2)
        reference = self.reference(blocks=20, slots=2)
        self.run_pair(candidate, reference, [[1, 2], [1, 1], [1, 2]],
                      prompt=160, output_target=6, drain_after=12)

    def test_pure_tot_trace_equivalence_when_supported(self):
        candidate = self.scheduler(blocks=64, slots=8)
        reference = self.reference(blocks=64, slots=8)
        self.run_pair(candidate, reference, [[3, 1, 3, 1]],
                      prompt=64, output_target=4, drain_after=16)

    def test_new_root_before_fifty_ticks_triggers_original_plan(self):
        candidate = self.scheduler()
        groups = {}
        self.add_stage(candidate, groups, 0, [1], 0)
        self.step(candidate)
        before = candidate.last_plan
        self.add_stage(candidate, groups, 1, [1], 0)
        self.step(candidate)
        self.assertGreater(candidate.last_plan, before)
        self.assertLess(candidate.last_plan, before + 50)
        self.cleanup(candidate)

    def test_new_stage_sibling_before_fifty_ticks_triggers_original_plan(self):
        candidate = self.scheduler()
        groups = {}
        self.add_stage(candidate, groups, 0, [1, 1], 0)
        self.step(candidate)
        before = candidate.last_plan
        groups['0:0:0'].first_seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
        candidate.free_finished_seq_groups()
        self.add_stage(candidate, groups, 0, [1, 1], 1)
        self.step(candidate)
        self.assertGreater(candidate.last_plan, before)
        self.assertLess(candidate.last_plan, before + 50)
        self.cleanup(candidate)

    def test_no_new_arrival_trace_remains_frozen_v47_equal(self):
        candidate = self.scheduler()
        reference = self.reference()
        self.run_pair(candidate, reference, [[1], [1]],
                      prompt=64, output_target=4, drain_after=0)

    def test_mixed_to_pure_clears_stale_selection_lease_and_drain(self):
        candidate = self.scheduler()
        foreground = self.group(99, 0)
        candidate.add_seq_group(foreground)
        self.step(candidate)
        self.step(candidate)
        candidate.abort_seq_group('99')
        candidate.selected = {'stale'}
        candidate.desired_selected = {'stale'}
        candidate.leases = {'stale': {'generated': 0, 'started': 1000.}}
        candidate.fast_expired_drain = True
        candidate.scheduler_config.max_num_seqs = 99
        reference = self.reference()
        groups_ref, groups_candidate = {}, {}
        self.add_stage(reference, groups_ref, 0, [1, 1], 0)
        self.add_stage(candidate, groups_candidate, 0, [1, 1], 0)
        out_ref = self.step(reference)
        out_candidate = self.step(candidate)
        self.compare(reference, candidate, groups_ref, groups_candidate, out_ref, out_candidate)
        self.assertNotIn('stale', candidate.selected)
        self.assertNotIn('stale', candidate.leases)
        self.assertEqual(candidate.scheduler_config.max_num_seqs,
                         candidate.base_max_num_seqs)
        self.cleanup(reference)
        self.cleanup(candidate)


if __name__ == '__main__':
    unittest.main(verbosity=2)
