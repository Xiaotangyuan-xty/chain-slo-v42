import unittest
import numpy as np
from stage_scenarios import Stage,Scenario,Scenarios
from window_projection import WindowProjector,weighted_cdf,ProjectionBudgetExceeded
from window_control import plan
class Cost:
 def seconds(self,n,context_tokens=0,prefill_tokens=0,transition=False):return .1 if prefill_tokens else .02 if n<=1 else .072 if n>=32 else .02+.052*(n-1)/31
 def transition_seconds(self,*args):return 0.
 def describe(self):return {}
def action(aid,stages,credit=100,slack=100,ids=None):
 return {'id':aid,'kind':'compound','members':ids or[aid],'arrival':0.,'cheap_credit':credit,'cheap_work':sum(s.width*s.decode for s in stages),'scenarios':Scenarios([Scenario(tuple(stages),credit)],slack)}
def views(actions):return {r:{'request':r,'prompt':0,'generated':0,'max_tokens':1024,'state':'running'}for a in actions for r in a['members']}
def projector(aa,ref,**kw):return WindowProjector(aa,ref,Cost(),100000,32,views(aa),1.5,0.,**kw)
class ProjectionContract(unittest.TestCase):
 def test_empirical_cdf_does_not_move_late_donor_credit_forward(self):
  c,p=weighted_cdf(np.array([1.,2.]),np.array([100.,100.]),1.9);self.assertEqual(c,50.);self.assertEqual(p,.5)
 def test_stage_width_uses_future_width_4p6_seconds(self):
  a=action('r',[Stage(1,50,0,0),Stage(32,50,0,0)],slack=3.)
  v=projector([a],{'r'}).evaluate({'r'});self.assertAlmostEqual(v['completion_seconds_by_action']['r'][0],4.6,places=5);self.assertEqual(v['credit'],0)
 def test_partial_running_root_is_not_expanded_to_waiting_sibling(self):
  a=action('r',[Stage(2,10,0,0)],ids=['x','y']);v=projector([a],{'x'}).evaluate({'x'})
  self.assertGreater(v['completion_seconds_by_action']['r'][0],1.5)
 def test_short_member_completion_reduces_batch_cost(self):
  a=action('r',[Stage(2,100,0,0,0.,(1.,100.),(0.,0.),(0.,0.),('x','y'))],ids=['x','y'])
  v=projector([a],{'x','y'}).evaluate({'x','y'});self.assertLess(v['completion_seconds_by_action']['r'][0],100*Cost().seconds(2))
 def test_proposal_changes_background_and_future_tail(self):
  root=action('r',[Stage(1,10,0,0),Stage(1,60,0,0)]);peer=action('p',[Stage(1,100,0,0)])
  p=projector([root,peer],{'r','p'});both=p.evaluate({'r','p'});focus=p.evaluate({'r'})
  self.assertNotEqual(both['completion_seconds_by_action']['r'],focus['completion_seconds_by_action']['r'])
 def test_small_prefill_is_at_least_one_call(self):
  a=action('r',[Stage(1,1,1,0)]);v=projector([a],{'r'}).evaluate({'r'});self.assertGreaterEqual(v['completion_seconds_by_action']['r'][0],.12-1e-8)
 def test_wide_prefill_is_charged_per_physical_wave(self):
  a=action('r',[Stage(64,1,64,0)],ids=['x']);p=WindowProjector([a],{'x'},Cost(),100000,8,views([a]),.01,0.)
  v=p.evaluate({'x'});self.assertGreaterEqual(v['completion_seconds_by_action']['r'][0],.8)
 def test_budget_does_not_cache_an_incomplete_evaluation(self):
  a=action('r',[Stage(1,100,0,0)]);p=projector([a],{'r'},expired=lambda:True)
  with self.assertRaises(ProjectionBudgetExceeded):p.evaluate({'r'})
  self.assertFalse(p.memo)
 def test_production_plan_reference_is_exact_physical_set(self):
  a=action('root',[Stage(2,10,0,0)],ids=['x','y']);v=views([a]);v['y']['state']='waiting'
  selected,d=plan([a],list(v.values()),{'x'},10000,32,.02,costs=Cost(),max_seconds=None)
  self.assertEqual(d['reference_members'],['x']);self.assertEqual(d['reference_semantics'],'exact_physical_running')
 def test_production_controller_can_choose_a_complete_positive_gain(self):
  aa=[action('r',[Stage(1,100,0,0)],credit=1000,slack=2.05),action('expired',[Stage(1,1000,0,0)],credit=0,slack=-1)]
  v=views(aa);selected,d=plan(aa,list(v.values()),{'r','expired'},10000,2,.02,costs=Cost(),max_seconds=None)
  self.assertEqual(selected,{'r'});self.assertGreater(d['paired']['delta_credit'],0);self.assertGreater(d['complete_sets'],1)
 def test_unsupported_live_factory_is_not_silently_zero_loss(self):
  a=action('r',[Stage(1,100,0,0)]);a.pop('scenarios');a['cheap_credit']=0.;a['slack']=10.;a['scenario_factory']=lambda:Scenarios([],10.,supported=False)
  v=views([a]);selected,d=plan([a],list(v.values()),{'r'},10000,2,.02,costs=Cost(),max_seconds=None)
  self.assertEqual(selected,{'r'});self.assertEqual(d['fallback_reason'],'unsupported_live_scenario')
 def test_integrated_one_for_two_KV_exchange_survives_single_action_losses(self):
  aa=[action('a',[Stage(1,50,0,0)],credit=500,slack=1.1),action('b',[Stage(1,10,0,0)],credit=450,slack=.5),action('c',[Stage(1,10,0,0)],credit=450,slack=.5)]
  for a in aa:a['cheap_work']=50.
  v=views(aa);v['a']['prompt']=46;v['b']['prompt']=14;v['c']['prompt']=14;v['b']['state']='waiting';v['c']['state']='waiting'
  class Flat(Cost):
   def seconds(self,n,*a,**k):return .02
  selected,d=plan(aa,list(v.values()),{'a'},8,3,.02,costs=Flat(),max_seconds=None,max_evaluations=5)
  self.assertEqual(selected,{'b','c'});self.assertGreater(d['joint_candidates_generated'],0);self.assertGreater(d['paired']['delta_credit'],0)

 def test_joint_CPU_reservation_uses_runtime_eviction_order(self):
  aa=[action(r,[Stage(1,1,0,0)])for r in ['a','b','n']];v=views(aa)
  for r in ['a','b']:v[r].update(prompt=160,generated=16,can_swap=True)
  v['a']['native_priority']=(1.,);v['b']['native_priority']=(2.,);v['n']['state']='waiting'
  p=WindowProjector(aa,{'a','b'},Cost(),10000,3,v,1.5,0.,swap_capacity_blocks=11)
  out=p.evaluate({'n'});self.assertEqual(out['eviction_order'],['b','a']);self.assertEqual(out['recompute_members'],['a'])

 def test_mean_context_is_not_used_as_start_then_grown_a_second_time(self):
  a=action('r',[Stage(1,100,0,150)])
  class Spy(Cost):
   def __init__(self):self.contexts=[]
   def seconds(self,n,context_tokens=0.,*a,**k):self.contexts.append(context_tokens);return .02
  c=Spy();p=WindowProjector([a],{'r'},c,10000,1,views([a]),1.5,0.);p.evaluate({'r'})
  self.assertEqual(c.contexts[0],100.);self.assertLessEqual(max(c.contexts),200.)

 def test_production_lease_remaining_prevents_eviction(self):
  aa=[action('r',[Stage(1,100,0,0)],slack=.1),action('new',[Stage(1,1,0,0)],credit=1000,slack=.1)]
  v=views(aa);v['new']['state']='waiting'
  selected,d=plan(aa,list(v.values()),{'r'},10000,1,.02,costs=Cost(),lease_remaining={'r':50},max_seconds=None)
  self.assertIn('r',selected)
if __name__=='__main__':unittest.main(verbosity=2)
