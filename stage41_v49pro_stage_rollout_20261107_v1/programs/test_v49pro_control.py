from collections import Counter
import math
import time
import unittest
from chain_forecast import StageForecaster
from stage_scenarios import Stage,Scenario,Scenarios,ScenarioCache,forecast_scenarios
from step_cost import StepCost
from mixed_control import search_sets,plan,eligible_decode_count
from paired_rollout import simulate,PairedRollout


class Costs:
    def __init__(self,points=None):self.points=points or {1:.02,2:.03,32:.072}
    def seconds(self,n,context_tokens=0,prefill_tokens=0,transition=False):
        if n<=0:return 0.
        if n in self.points:return self.points[n]*(1+prefill_tokens/2048)
        return self.points[max(self.points)]*max(1,n/max(self.points))
    def transition_seconds(self,n,context=0):return n*.005
    def describe(self):return {'test_fixture':True}


def action(aid,width=1,decode=20,credit=1000,slack=10):
    return {'id':aid,'kind':'compound','members':[aid+str(i)for i in range(width)],
            'arrival':0.,'native_priority':(0.,), 'continuation_priority':(0.,),
            'cheap_credit':credit,'cheap_work':decode*width,
            'scenarios':Scenarios([Scenario((Stage(width,decode,0,128*width),),credit)],slack)}


class StageService(unittest.TestCase):
    def test_width1_then32_rejects_flat_current_width_false_positive(self):
        a=Scenarios([Scenario((Stage(1,50,0,0),Stage(32,50,0,0)),10000)],3)
        v=a.isolated(Costs(),current_members=1)
        self.assertAlmostEqual(v['scenario_completion_seconds'][0],4.6)
        self.assertEqual(v['probability'],0)
        j=action('x');j['scenarios']=a
        r=simulate([j],{'x'},Costs(),100000,32,1.)
        self.assertEqual(r['actions']['x']['credit'],0)

    def test_current_congestion_is_not_applied_to_narrow_future_stage(self):
        a=Scenarios([Scenario((Stage(32,50,0,0),Stage(1,50,0,0)),10000)],5)
        v=a.isolated(Costs(),current_members=32)
        self.assertAlmostEqual(v['scenario_completion_seconds'][0],4.6)
        self.assertEqual(v['probability'],1)

    def test_each_donor_uses_own_completion_and_terminal_credit(self):
        bundle=Scenarios([Scenario((Stage(1,10,0,0),),100),Scenario((Stage(1,100,0,0),),1000)],1)
        v=bundle.isolated(Costs());self.assertEqual(v['probability'],.5);self.assertEqual(v['expected_credit'],50)

    def test_future_prefill_has_service_cost_and_width_larger_than_slots_uses_waves(self):
        a=action('wide',width=4,decode=10);a['scenarios']=Scenarios([Scenario((Stage(4,10,4096,512),),100)],10)
        fast=simulate([a],{'wide'},Costs(),10000,4,1.)
        slow=simulate([a],{'wide'},Costs(),10000,1,1.)
        self.assertGreater(slow['actions']['wide']['finish'],fast['actions']['wide']['finish'])
        self.assertGreater(fast['actions']['wide']['finish'],.2)

    def test_current_stage_uses_mixed_batch_then_reduces_width_after_peer_finishes(self):
        a=action('a',decode=50);b=action('b',decode=10)
        r=simulate([a,b],{'a','b'},Costs({1:.02,2:.04}),10000,2,5.)
        self.assertAlmostEqual(r['actions']['b']['finish'],.4)
        self.assertAlmostEqual(r['actions']['a']['finish'],1.2)

    def test_full_prefix_is_conditional_not_paid_before_terminal(self):
        a=action('root',decode=100,credit=100000,slack=1)
        self.assertEqual(simulate([a],{'root'},Costs(),10000,1,1.)['credit'],0)

    def test_small_prefill_still_requires_one_executor_iteration(self):
        a=Scenarios([Scenario((Stage(1,0,1,1),),100)],1)
        self.assertGreaterEqual(a.isolated(Costs())['scenario_completion_seconds'][0],.02)

    def test_completed_selected_root_allows_common_rolling_backfill(self):
        a=action('fast',decode=1);b=action('background',decode=1)
        v=simulate([a,b],{'fast'},Costs({1:.02}),10000,1,10.)
        self.assertLess(v['actions']['background']['finish'],.1)

    def test_planning_delay_consumes_deadline_slack(self):
        a=action('a',decode=50,slack=1.001)
        self.assertEqual(simulate([a],{'a'},Costs(),10000,1,1.)['credit'],1000)
        self.assertEqual(simulate([a],{'a'},Costs(),10000,1,1.,planning_seconds=.01)['credit'],0)

    def test_paired_window_scenario_continuation_and_action_deltas(self):
        a=action('resident',decode=30,slack=.7);a['resident']=True
        b=action('new',decode=10,credit=2000,slack=.3)
        p=PairedRollout([a,b],{'resident'},Costs({1:.02,2:.04}),10000,1,.2)
        v=p.paired({'new'})
        self.assertEqual(v['same_window_seconds'],.2)
        self.assertGreater(v['candidate_gain'],0)
        self.assertEqual(v['delta_credit'],sum(v['paired_action_delta'].values()))
        self.assertTrue(v['complete_envelope'])

    def test_switch_charge_only_displaced_actions_not_entire_incumbent(self):
        acts=[action(x,decode=100)for x in ['a','b','c','new']]
        for a in acts:a['resident']=a['id']!='new'
        v=simulate(acts,{'a','b','new'},Costs(),10000,3,1.)
        self.assertEqual(v['affected_members'],1);self.assertAlmostEqual(v['switch_seconds'],.005)

    def test_event_limit_is_unknown_not_silent_zero_probability(self):
        a=action('x');v=simulate([a],{'x'},Costs(),10000,1,1.,event_limit=0)
        self.assertFalse(v['complete_envelope'])

    def test_latency_settlement_excludes_already_earned_prefix_and_first_output(self):
        a={'id':'f','kind':'foreground','members':['f'],'arrival':0.,'foreground':(0,10000,50,100,1024,0,0.,(2.,.1,20.))}
        v=simulate([a],{'f'},Costs(),10000,1,10.)
        self.assertEqual(v['credit'],8*50)
        a['foreground']=(0,100,0,10,1024,0,0.,(2.,.1,20.))
        self.assertEqual(simulate([a],{'f'},Costs(),10000,1,10.)['credit'],100+8*9)

    def test_latency_late_ttft_can_recover_cumulative_deadlines(self):
        a={'id':'f','kind':'foreground','members':['f'],'arrival':0.,'foreground':(0,100,0,100,1024,0,3.,(2.,.1,20.))}
        v=simulate([a],{'f'},Costs({1:.02}),10000,1,10.)
        self.assertGreater(v['credit'],0);self.assertLess(v['credit'],8*99)


class SetSearch(unittest.TestCase):
    def test_pair_add_escapes_three_way_batching_trap(self):
        value=lambda ids: {0:0,1:500,2:487.8,3:731.7}[len(ids)]
        best,d=search_sets(['a','b','c'],[set()],lambda ids:True,value,max_evaluations=1000)
        self.assertEqual(best,{'a','b','c'});self.assertTrue(d['exact_pool_enumeration'])

    def test_one_for_two_escapes_KV_density_trap(self):
        need={'a':6,'b':4,'c':4};value={'a':500,'b':450,'c':450}
        best,d=search_sets(list(need),[{'a'},set()],lambda ids:sum(need[x]for x in ids)<=8,lambda ids:sum(value[x]for x in ids),max_evaluations=1000)
        self.assertEqual(best,{'b','c'})

    def test_empty_and_incumbent_are_explicit_and_budget_bounded(self):
        seen=[]
        best,d=search_sets(list('abcdefghij'),[set(),{'a','b'}],lambda ids:True,lambda ids:seen.append(ids)or len(ids),max_evaluations=7)
        self.assertIn(frozenset(),seen);self.assertIn(frozenset({'a','b'}),seen)
        self.assertLessEqual(len(seen),7);self.assertTrue(d['search_exhausted_budget'])

    def test_cheap_pruning_precedes_expensive_factories_256_candidates(self):
        called=[];acts=[];ms=[]
        for i in range(256):
            a=action(str(i));a['members']=[str(i)]
            a['scenario_factory']=lambda i=i,a=a:called.append(i)or a['scenarios']
            acts.append(a);ms.append({'request':str(i),'prompt':128,'generated':0,'max_tokens':1024,'prefill_tokens':0})
        selected,d=plan(acts,ms,set(),10000,8,.02,costs=Costs(),pool_limit=12,max_seconds=None,max_evaluations=3)
        self.assertLessEqual(len(called),12);self.assertEqual(d['all_actions'],256)

    def test_integrated_paired_rollout_replaces_one_KV_heavy_resident_with_two(self):
        acts=[action('a',decode=50,credit=500,slack=1.1),action('b',decode=10,credit=450,slack=.5),action('c',decode=10,credit=450,slack=.5)]
        for a in acts:a['members']=[a['id']]
        ms=[{'request':'a','prompt':46,'generated':0,'max_tokens':100},
            {'request':'b','prompt':14,'generated':0,'max_tokens':100},
            {'request':'c','prompt':14,'generated':0,'max_tokens':100}]
        selected,d=plan(acts,ms,{'a'},8,3,.02,costs=Costs({1:.02,2:.02,3:.02}),previous={'a'},max_seconds=None,max_evaluations=1000)
        self.assertEqual(selected,{'b','c'})
        self.assertGreater(d['paired']['delta_credit'],0)
        self.assertGreater(d['paired']['resident_loss'],0)
        self.assertEqual(d['selected_charge'],8)

    def test_missing_reference_model_falls_back_native_not_zero_gain(self):
        a=action('a');a['scenarios']=Scenarios([],10,supported=False)
        selected,d=plan([a],[{'request':'a0','prompt':128,'generated':0,'max_tokens':100}],set(),1000,1,.02,max_seconds=None)
        self.assertIsNone(selected);self.assertIn('unsupported',d['fallback_reason'])

    def test_reference_over_pool_capacity_never_omits_resident_losses(self):
        acts=[action(str(i))for i in range(20)];ms=[{'request':str(i)+'0','prompt':128,'generated':0,'max_tokens':100}for i in range(20)]
        selected,d=plan(acts,ms,{m['request']for m in ms},10000,32,.02,max_seconds=None)
        self.assertIsNone(selected);self.assertEqual(d['fallback_reason'],'resident_reference_exceeds_model_pool')

    def test_zero_or_overrun_budget_returns_native_and_preserves_requests(self):
        a=action('a');m={'request':'a0','prompt':128,'generated':0,'max_tokens':100}
        selected,d=plan([a],[m],set(),1000,1,.02,max_seconds=0.)
        self.assertIsNone(selected);self.assertEqual(a['members'],['a0'])


class ObservationModel(unittest.TestCase):
    def test_long_prefill_swap_preempted_ignored_planned_samples_are_retained(self):
        m=StepCost();f={'members':8,'context_tokens':32000,'prefill_tokens':2048,'transition':True,'preempted':3,'ignored':2,'planning_seconds':.2}
        self.assertTrue(m.observe_execution(f,2.5))
        for k in ['long_over1s','prefill_samples','transition_samples','planned_samples','preempted_samples','ignored_samples']:self.assertEqual(m.stats[k],1)

    def test_strata_and_samples_have_hard_storage_caps(self):
        m=StepCost(capacity=2,strata=3)
        for i in range(10):
            for _ in range(5):m.observe_execution({'members':i+1},.02)
        self.assertEqual(len(m.samples),3);self.assertTrue(all(len(v)<=2 for v in m.samples.values()))
        self.assertGreater(m.stats['strata_evicted'],0)

    def test_context_prefill_are_not_collapsed_into_member_count(self):
        m=StepCost()
        for f,t in [({'members':1,'context_tokens':512},.02),({'members':1,'context_tokens':8192},.08),({'members':1,'context_tokens':512,'prefill_tokens':2048},.1)]:m.observe_execution(f,t)
        s=m.snapshot(.02,1)
        self.assertAlmostEqual(s.seconds(1,512),.02);self.assertAlmostEqual(s.seconds(1,8192),.08)
        self.assertAlmostEqual(s.seconds(1,512,2048),.1)
        self.assertEqual(len(s._cache),3)

    def test_invalid_observations_only_are_rejected(self):
        m=StepCost()
        for n,t in [(1,float('nan')),(1,0),(-1,2)]:self.assertFalse(m.observe(n,t))
        self.assertTrue(m.observe(1,10.))


if __name__=='__main__':unittest.main(verbosity=2)
