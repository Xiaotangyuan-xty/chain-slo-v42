"""Real native scheduler, allocator, swap/recompute and queue integration; CPU only."""
import unittest
from unittest.mock import patch
from vllm.config import CacheConfig,SchedulerConfig
from vllm.sequence import Sequence,SequenceGroup,SequenceStatus,Logprob
from vllm import SamplingParams
from vllm.core.scheduler import PreemptionMode
from jitserve.request_info import RequestInfo,RequestType
from mixed_joint import MixedScheduler
from native_chain import NativeChainScheduler


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.wall=patch('time.time',return_value=100.);self.clock=patch('time.perf_counter',return_value=1000.)
        self.wall.start();self.clock.start();self.addCleanup(self.wall.stop);self.addCleanup(self.clock.stop)
    def scheduler(self,blocks=64,slots=4,cpu=64):
        self.shared=SchedulerConfig('generate',64,slots,512,enable_chunked_prefill=True,policy='jitserve',top_k_selection=3,max_swaps_per_iter=3)
        cache=CacheConfig(16,.9,1,'auto');cache.num_gpu_blocks=blocks;cache.num_cpu_blocks=cpu
        return MixedScheduler(self.shared,cache,None)
    def group(self,rid,kind=2,prompt=64,cid=None,stage=0,width=1,deadline=2000.):
        info=RequestInfo(RequestType(kind),(2.,.1,20.),0,rid if cid is None else cid,20,prompt,100,stage_id=stage,request_id=0)
        if kind==2:info.chain_deadline=deadline;info.chain_stage_count=stage+1;info.chain_stage_widths=[1]*stage+[width]
        seq=Sequence(rid,inputs={'prompt_token_ids':list(range(prompt)),'prompt':'CPU'},block_size=16)
        return SequenceGroup(str(rid),[seq],100.,info,SamplingParams(temperature=0,max_tokens=100))
    def step(self,s):
        _,out,_=s.schedule()
        for item in out.scheduled_seq_groups:
            g=item.seq_group;g.update_num_computed_tokens(item.token_chunk_size)
            if not g.is_prefill():g.first_seq.append_token_id(1,{1:Logprob(0.)})
        return out
    def process(self,s,groups,token_budget=64,max_seqs=4):
        from vllm.core.scheduler import SchedulingBudget,SchedulerPrefillOutputs,SchedulerSwappedInOutputs
        prefills,swapped,running=s._init_preemption_outputs(True,SchedulerPrefillOutputs,SchedulerSwappedInOutputs)
        s._schedule_process_list(groups,running,prefills,swapped,SchedulingBudget(token_budget,max_seqs),True)
        return prefills,swapped,running
    def cleanup(self,s):
        s.abort_seq_group([g.request_id for g in list(s.running)+list(s.waiting)+list(s.swapped)])
        self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),s.block_manager.num_total_gpu_blocks)
    def test_new_independent_chain_and_foreground_replan_before_fifty_ticks(self):
        s=self.scheduler();s.add_seq_group(self.group(1));s.add_seq_group(self.group(9,0));self.step(s);n=s.v47_counts['plans']
        s.add_seq_group(self.group(2));self.step(s);self.assertGreater(s.v47_counts['plans'],n)
        n=s.v47_counts['plans'];s.add_seq_group(self.group(3,0));self.step(s)
        self.assertGreater(s.v47_counts['plans'],n);self.assertEqual(s.mode,'joint');self.cleanup(s)
    def test_queue_sorting_is_not_a_physical_state_change(self):
        from collections import deque
        s=self.scheduler();s.add_seq_group(self.group(1));s.add_seq_group(self.group(2));s.add_seq_group(self.group(9,0))
        for _ in range(4):self.step(s)
        self.assertEqual(len(s.running),3)
        first=s._event_state(list(s.running),1000.)[0]['physical_state']
        s.running=deque(reversed(s.running))
        second=s._event_state(list(s.running),1000.)[0]['physical_state']
        self.assertEqual(first,second);self.cleanup(s)

    def test_partial_completion_with_surviving_selected_invalidates(self):
        s=self.scheduler();a=self.group(1);b=self.group(2)
        s.add_seq_group(a);s.add_seq_group(b);s.add_seq_group(self.group(9,0));self.step(s);n=s.v47_counts['plans']
        a.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED
        self.step(s);self.assertGreater(s.v47_counts['plans'],n)
        self.assertIn('membership_stage',s.events.last_reasons);self.cleanup(s)
    def test_abort_with_other_selected_alive_replans(self):
        s=self.scheduler();s.add_seq_group(self.group(1));s.add_seq_group(self.group(2));s.add_seq_group(self.group(9,0));self.step(s)
        count=s.v47_counts['plans'];s.abort_seq_group('1');self.step(s)
        self.assertGreater(s.v47_counts['plans'],count);self.assertIn('membership_stage',s.events.last_reasons);self.cleanup(s)

    def test_new_stage_during_barrier_replans(self):
        s=self.scheduler();g=self.group(1);g.request_info.chain_stage_count=2;g.request_info.chain_stage_widths=[1,1]
        s.add_seq_group(g);s.add_seq_group(self.group(2));s.add_seq_group(self.group(9,0));self.step(s)
        g.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED;s.free_finished_seq_groups()
        s.add_seq_group(self.group(3,cid=1,stage=1));self.step(s)
        self.assertIn('membership_stage',s.events.last_reasons);self.assertEqual(s.policy.chains[1]['stages'],2);self.cleanup(s)

    def test_prediction_revision_invalidates(self):
        s=self.scheduler();g=self.group(1);s.add_seq_group(g);s.add_seq_group(self.group(9,0));self.step(s)
        g.request_info.prediction_revision=1
        with patch('time.perf_counter',return_value=1001.):self.step(s)
        self.assertIn('prediction_revision',s.events.last_reasons);self.cleanup(s)
    def test_foreground_prediction_revision_coalesces_then_replans(self):
        s=self.scheduler();g=self.group(1,0);s.add_seq_group(g);self.step(s)
        g.request_info.prediction_revision=1;self.step(s)
        self.assertEqual(s.events.counts['prediction_revision'],0)
        with patch('time.perf_counter',return_value=1001.):self.step(s)
        self.assertEqual(s.events.counts['prediction_revision'],1);self.cleanup(s)
    def test_configuration_invalidates_but_preserves_initial_cap(self):
        s=self.scheduler(slots=2);s.add_seq_group(self.group(1));s.add_seq_group(self.group(9,0));self.step(s)
        s.scheduler_config.max_num_batched_tokens=128;self.step(s)
        self.assertIn('configuration',s.events.last_reasons);self.assertEqual(s.base_max_num_seqs,2);self.cleanup(s)
    def test_hidden_deadline_expiry_replans_without_membership_change(self):
        s=self.scheduler();s.add_seq_group(self.group(1,deadline=1000.4));s.add_seq_group(self.group(2));s.add_seq_group(self.group(9,0))
        self.step(s);n=s.v47_counts['plans']
        with patch('time.perf_counter',return_value=1000.6):self.step(s)
        self.assertGreater(s.v47_counts['plans'],n);self.assertIn('deadline_wait_or_quantum',s.events.last_reasons);self.cleanup(s)
    def test_foreground_ttft_deadline_is_event(self):
        s=self.scheduler();s.add_seq_group(self.group(1,0,prompt=256));self.step(s)
        self.assertIsNotNone(s.events.next_due)
        with patch('time.perf_counter',return_value=1002.1):self.step(s)
        self.assertGreater(s.events.counts['deadline_wait_or_quantum'],0);self.cleanup(s)
    def test_wall_quantum_expires_and_foreground_has_same_lease(self):
        s=self.scheduler();a=self.group(1,0);b=self.group(2)
        s.add_seq_group(a);s.add_seq_group(b)
        for _ in range(5):self.step(s)
        self.assertIn(a.request_id,s.leases);self.assertIn(b.request_id,s.leases)
        with patch('time.perf_counter',return_value=1001.6):
            self.step(s);self.assertFalse(s._protected(1001.6))
        self.cleanup(s)
    def test_lease_protected_deselected_resident_is_not_evicted(self):
        s=self.scheduler();g=self.group(1);s.add_seq_group(g);self.step(s)
        self.assertIn(g,s.running);s.last_protected={'1'}
        n=s.v47_counts['all_preemptions'];out=s._execute_selected(set())
        self.assertEqual(s.v47_counts['all_preemptions'],n);self.assertTrue(out.scheduled_seq_groups);self.cleanup(s)
    def test_deselected_residents_have_bounded_turnover_without_duplicates(self):
        from collections import deque
        s=self.scheduler(slots=6);groups=[self.group(i)for i in range(1,7)]
        for g in groups:s.add_seq_group(g)
        prefills,_,_=self.process(s,groups,max_seqs=6)
        s.running.extend(x.seq_group for x in prefills.seq_groups);self.assertEqual(len(s.running),6)
        for item in prefills.seq_groups:
            g=item.seq_group;g.update_num_computed_tokens(item.token_chunk_size)
            g.first_seq.append_token_id(1,{1:Logprob(0.)})
        running_ids={g.request_id for g in s.running}
        s.waiting=deque(g for g in s.waiting if g.request_id not in running_ids)
        s.leases={rid:{'generated':1,'started':1000.}for rid in running_ids}
        s.last_protected={'2'};s.policy.draining=False
        with patch.object(s,'_choose_preemption',return_value=PreemptionMode.RECOMPUTE):
            out=s._execute_selected({'1','6'})
        self.assertEqual(out.preempted,3);self.assertEqual(s.v47_counts['all_preemptions'],3)
        ids=[g.request_id for queue in (s.running,s.waiting,s.swapped)for g in queue]
        self.assertCountEqual(ids,[str(i)for i in range(1,7)]);self.assertEqual(len(ids),len(set(ids)))
        running_ids={g.request_id for g in s.running}
        self.assertIn('1',running_ids);self.assertIn('2',running_ids)
        self.assertTrue(all(g.request_id not in {'3','4','5'}for g in s.running));self.cleanup(s)
    def test_resident_execution_mask_is_declared_before_execution(self):
        s=self.scheduler();g=self.group(1);s.add_seq_group(g);self.step(s);s.last_protected={'1'}
        out=s._execute_selected(set())
        self.assertIn('1',s.last_recorded_selected)
        self.assertTrue(all(x.seq_group.request_id in s.last_recorded_selected for x in out.scheduled_seq_groups));self.cleanup(s)

    def test_foreground_native_mode_clears_candidate_filter(self):
        s=self.scheduler();s.add_seq_group(self.group(1,0));self.step(s)
        self.assertEqual(s.mode,'foreground');self.assertTrue(s.policy.draining)
        self.assertIsNone(s.selected);self.assertIsNone(s.last_recorded_selected);self.cleanup(s)

    def test_recompute_frees_kv_and_schedules_same_round(self):
        s=self.scheduler(blocks=20,slots=2,cpu=1);a=self.group(1,prompt=160);s.add_seq_group(a)
        for _ in range(4):self.step(s)
        prefix=a.first_seq.get_output_len();b=self.group(2,prompt=160);s.add_seq_group(b);s.last_protected=set()
        out=s._execute_selected({'2'})
        self.assertTrue(out.scheduled_seq_groups);self.assertFalse(out.blocks_to_swap_out)
        self.assertGreater(s.v47_counts['same_round_after_recompute'],0)
        self.assertEqual(a.first_seq.get_output_len(),prefix);self.assertNotIn('1',s.leases);self.cleanup(s)
    def test_swap_dependency_uses_empty_round_and_preserves_output(self):
        s=self.scheduler(blocks=20,slots=2);a=self.group(1,prompt=160);s.add_seq_group(a)
        for _ in range(4):self.step(s)
        prefix=a.first_seq.get_output_len();s.add_seq_group(self.group(2,prompt=160));s.last_protected=set()
        with patch.object(s,'_choose_preemption',return_value=PreemptionMode.SWAP):out=s._execute_selected({'2'})
        self.assertTrue(out.blocks_to_swap_out);self.assertFalse(out.blocks_to_swap_in);self.assertFalse(out.scheduled_seq_groups)
        self.assertEqual(a.first_seq.get_output_len(),prefix);self.cleanup(s)
    def test_native_physical_preemption_invalidates_lease_and_plan(self):
        s=self.scheduler();g=self.group(1);s.add_seq_group(g)
        for _ in range(3):self.step(s)
        old=g.first_seq.get_output_len();s._preempt(g,[],PreemptionMode.RECOMPUTE)
        s.running.remove(g);s.waiting.append(g)
        self.assertNotIn('1',s.leases);self.assertIn('physical_preemption',s.events.invalidations)
        for _ in range(5):self.step(s)
        self.assertGreaterEqual(s.leases['1'],old);self.cleanup(s)
    def test_mixed_foreground_pure_mode_epoch_clears_native_bins_and_old_leases(self):
        s=self.scheduler();fg=self.group(1,0);c=self.group(2)
        s.add_seq_group(fg);s.add_seq_group(c);self.step(s);epoch=s.mode_epoch
        s.abort_seq_group('2');self.step(s);self.assertEqual(s.mode,'foreground')
        s._pending_swap_out_queue.append(fg);s._current_bin_queue.append(fg)
        s.abort_seq_group('1');s.add_seq_group(self.group(3));self.step(s)
        self.assertEqual(s.mode,'pure');self.assertGreater(s.mode_epoch,epoch)
        self.assertFalse(s._pending_swap_out_queue);self.assertFalse(s._current_bin_queue)
        self.assertNotIn('1',s.leases);self.assertNotIn('2',s.leases);self.cleanup(s)
    def test_foreground_recovery_uses_physical_status_not_stale_queue_snapshot(self):
        from jitserve.scheduler.policy import SLOPolicy
        s=self.scheduler();g=self.group(1,0);s.policy.physical_running=[g]
        def recovery(view):
            self.assertFalse(view['resident']);return 1.
        with patch.object(s.cost_model,'recovery',side_effect=recovery),patch.object(SLOPolicy,'reward_estimation',return_value=(100.,10.)):
            s.policy.get_priority(g)

    def test_native_adjustment_changes_only_budget(self):
        from vllm.core.scheduler import SchedulingBudget
        s=self.scheduler(slots=2);budget=SchedulingBudget(64,2)
        s.policy.interval_time=.5;s._adjust_batch_limits([None]*10,.1,budget)
        self.assertEqual(self.shared.max_num_seqs,2);self.assertEqual(s.scheduler_config.max_num_seqs,2)
        self.assertEqual(s.base_max_num_seqs,2);self.assertGreater(budget.max_num_seqs,2)
    def test_expired_drain_limit_is_local_and_reversible(self):
        s=self.scheduler(slots=2);s.add_seq_group(self.group(1,deadline=900.));self.step(s)
        self.assertEqual(s.mode,'expired_drain');self.assertEqual(s.scheduler_config.max_num_seqs,2)
        s.add_seq_group(self.group(2,0));self.step(s)
        self.assertEqual(s.mode,'joint');self.assertEqual(self.shared.max_num_seqs,2);self.cleanup(s)
    def test_idle_gap_is_not_observed_as_gpu_service_time(self):
        s=self.scheduler();g=self.group(1);s.add_seq_group(g);self.step(s)
        g.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED
        with patch('time.perf_counter',return_value=1100.):
            s.add_seq_group(self.group(2));self.step(s)
        self.assertLess(s.cost_model.last_round,1.);self.cleanup(s)
    def test_pure_and_mixed_do_not_call_native_root_scoring_or_native_pack(self):
        from jitserve.scheduler.policy import SLOPolicy
        with patch.object(SLOPolicy,'get_priority',side_effect=AssertionError('native root score')),patch.object(MixedScheduler,'_schedule_slo',side_effect=AssertionError('native pack')):
            s=self.scheduler();s.add_seq_group(self.group(1));self.step(s)
            s.add_seq_group(self.group(2,0));self.step(s);self.cleanup(s)

    def test_native_idle_guard_does_not_abort_pending_requests(self):
        s=self.scheduler();g=self.group(1);s.add_seq_group(g)
        s.dull_iteration=100000
        self.assertEqual(s._stop_dull_scheduling(),[]);self.assertIn(g,s.waiting)
        self.assertFalse(g.is_finished());self.cleanup(s)

    def test_low_kv_finite_mixed_replay_drains_all_requests_without_prefix_loss(self):
        s=self.scheduler(blocks=20,slots=2);groups=[self.group(i,i%3,prompt=64+(i%2)*64,deadline=900.) for i in range(1,13)]
        for g in groups:s.add_seq_group(g)
        previous={g.request_id:0 for g in groups}
        for tick in range(2000):
            out=self.step(s)
            for g in groups:
                n=g.first_seq.get_output_len();self.assertGreaterEqual(n,previous[g.request_id]);previous[g.request_id]=n
                if n>=20:g.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED
            s.free_finished_seq_groups()
            if not(s.running or s.waiting or s.swapped):break
        self.assertLess(tick,1999);self.assertTrue(all(v>=20 for v in previous.values()))
        self.cleanup(s);self.assertEqual(s.block_manager.get_num_free_cpu_blocks(),64)

    def test_native_arguments_and_swap_recovery(self):
        s=self.scheduler();g=self.group(1,0);s.add_seq_group(g)
        lookahead=s._get_num_lookahead_slots(True,True)
        with patch.object(s,'_schedule_slo_prefills',wraps=s._schedule_slo_prefills) as prefill:
            prefills,_,_=self.process(s,[g])
        prefill.assert_called_once();args=prefill.call_args.args
        self.assertEqual(args[0],g);self.assertEqual(args[1],64)
        self.assertEqual(args[2],lookahead);self.assertTrue(args[3])
        s.running.extend(x.seq_group for x in prefills.seq_groups)
        g.update_num_computed_tokens(prefills.seq_groups[0].token_chunk_size)
        g.first_seq.append_token_id(1,{1:Logprob(0.)});prefix=g.first_seq.get_output_len()
        swaps=[]
        self.assertEqual(s._preempt(g,swaps,PreemptionMode.SWAP),PreemptionMode.SWAP)
        s.running.remove(g);s.swapped.append(g)
        lookahead=s._get_num_lookahead_slots(False,True)
        with patch.object(s,'_schedule_slo_swapped',wraps=s._schedule_slo_swapped) as recovery:
            _,swapped,_=self.process(s,[g])
        recovery.assert_called_once_with(g,1,lookahead,True)
        self.assertTrue(swapped.blocks_to_swap_in)
        self.assertEqual([x.seq_group for x in swapped.decode_seq_groups],[g])
        self.assertNotIn(g,s.swapped);self.assertEqual(g.first_seq.get_output_len(),prefix)
        s.running.extend(x.seq_group for x in swapped.decode_seq_groups);self.cleanup(s)

    def test_native_never_is_dequeued_and_finally_drained(self):
        from vllm.core.interfaces import AllocStatus
        s=self.scheduler();g=self.group(1,0);s.add_seq_group(g)
        with patch.object(s.block_manager,'can_allocate',return_value=AllocStatus.NEVER):
            prefills,_,_=self.process(s,[g])
        self.assertEqual(prefills.ignored_seq_groups,[g]);self.assertNotIn(g,s.waiting)
        self.assertFalse(s.running or s.waiting or s.swapped)
        out=s._schedule_slo();self.assertFalse(out.ignored_seq_groups)
        self.assertFalse(s.running or s.waiting or s.swapped);self.cleanup(s)

    def test_low_budget_running_is_conserved_and_recovers_monotonically(self):
        s=self.scheduler();groups=[self.group(i,0,prompt=2) for i in range(1,3)]
        for g in groups:s.add_seq_group(g)
        prefills,_,_=self.process(s,groups)
        s.running.extend(x.seq_group for x in prefills.seq_groups)
        for item in prefills.seq_groups:
            g=item.seq_group;g.update_num_computed_tokens(item.token_chunk_size)
            g.first_seq.append_token_id(1,{1:Logprob(0.)})
        previous={g.request_id:g.first_seq.get_output_len() for g in groups}
        s.scheduler_config.max_num_batched_tokens=1;out=s._schedule_slo()
        for item in out.scheduled_seq_groups:
            g=item.seq_group;g.update_num_computed_tokens(item.token_chunk_size)
            if not g.is_prefill():g.first_seq.append_token_id(1,{1:Logprob(0.)})
        ids=[g.request_id for queue in (s.running,s.waiting,s.swapped) for g in queue]
        self.assertCountEqual(ids,['1','2']);self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(s.running),2);self.assertEqual(out.num_batched_tokens,1)
        self.assertEqual(len(out.scheduled_seq_groups),1)
        for g in groups:
            n=g.first_seq.get_output_len();self.assertGreaterEqual(n,previous[g.request_id]);previous[g.request_id]=n
        s.scheduler_config.max_num_batched_tokens=64
        for tick in range(10):
            out=s._schedule_slo()
            for item in out.scheduled_seq_groups:
                g=item.seq_group;g.update_num_computed_tokens(item.token_chunk_size)
                if not g.is_prefill():g.first_seq.append_token_id(1,{1:Logprob(0.)})
            for g in groups:
                n=g.first_seq.get_output_len();self.assertGreaterEqual(n,previous[g.request_id]);previous[g.request_id]=n
                if n>=4:g.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED
            ids=[g.request_id for queue in (s.running,s.waiting,s.swapped) for g in queue]
            self.assertEqual(len(ids),len(set(ids)));s.free_finished_seq_groups()
            if not(s.running or s.waiting or s.swapped):break
        self.assertLess(tick,9);self.assertTrue(all(v>=4 for v in previous.values()));self.cleanup(s)

    def test_quantum_without_protection_change_skips_full_plan(self):
        s=self.scheduler();g=self.group(1);s.add_seq_group(g);s.add_seq_group(self.group(9,0));self.step(s);self.step(s)
        for running in s.running:s.leases.setdefault(running.request_id,{'generated':running.first_seq.get_output_len(),'started':1000.})
        s.last_protected=s._protected(1000.)
        groups=list(s.running)+list(s.waiting)+list(s.swapped);plans=s.v47_counts['plans']
        with patch.object(s.events,'reasons',return_value=['service_quantum']),patch.object(s.policy,'goals',side_effect=AssertionError('full forecast called')):
            s._plan_loop(groups,1000.,joint=True)
        self.assertEqual(s.v47_counts['plans'],plans)
        self.assertEqual(s.events.counts['quantum_without_protection_change'],1);self.cleanup(s)

    def test_no_positive_choice_uses_required_fallback_with_protected_resident(self):
        from chain_admission import reprice
        goal={'chain':1,'stage':0,'deadline':100.,'arrival':0.,'probability':0.,'score':0.,
              'expected_credited_tokens':0.,'members':[{'request':'1','prompt':1,'generated':0,'max_tokens':100}]}
        class Zero:
            def revalue(self,goal,first,background):return {**goal,'probability':0.,'expected_credited_tokens':0.,'score':0.}
        selected,rows=reprice([goal],{'1'},{'1'},Zero())
        self.assertIsNone(selected);self.assertEqual(rows[0]['probability'],0.)

    def test_diagnostic_joint_disables_revalue_and_pure_bypasses_pro_selector(self):
        import joint_admission
        import native_chain
        with patch.object(native_chain.MODEL,'revalue',side_effect=AssertionError('diagnostic revalue called')):
            s=self.scheduler();s.add_seq_group(self.group(1));s.add_seq_group(self.group(2,0))
            with patch('joint_admission.select_joint',wraps=joint_admission.select_joint)as selector:self.step(s)
            self.assertIsNone(selector.call_args.kwargs['forecaster']);self.cleanup(s)
            s=self.scheduler();s.add_seq_group(self.group(1))
            with patch('native_chain.select',side_effect=AssertionError('pro pure selector called')):self.step(s)
            self.assertTrue(s._pure_core_active);self.cleanup(s)


if __name__=='__main__':unittest.main(verbosity=2)
