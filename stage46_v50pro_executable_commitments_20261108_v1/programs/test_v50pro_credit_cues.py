import unittest
from credit_cues import root_cue,foreground_cue,compute_target
from stage_scenarios import Stage,Scenario,Scenarios
from compute_admission import choose
class Cost:
 def seconds(self,n,*a,**k):return .02
 def prefill_seconds(self,tokens,*a,**k):return .1 if tokens else 0.
class Cues(unittest.TestCase):
 def test_root_uses_terminal_credit_and_all_future_stage_service(self):
  b=Scenarios([Scenario((Stage(3,50,0,0),Stage(1,50,0,0)),1000)],3)
  credit,seconds,d=root_cue(b,Cost());self.assertEqual(credit,1000);self.assertAlmostEqual(seconds,2.)
 def test_prefix_credit_is_zero_if_terminal_root_not_feasible(self):
  b=Scenarios([Scenario((Stage(1,100,0,0),),10000)],1)
  credit,_,d=root_cue(b,Cost());self.assertEqual(credit,0.)
 def test_QRF_prediction_capped_by_declared_sampling_limit(self):
  c=(1,100,0,1024,16,0,0.,(2.,.1,20.));credit,t=foreground_cue(c,Cost());self.assertEqual(credit,228.);self.assertAlmostEqual(t,.32)
 def test_expired_foreground_not_ranked_as_a_full_credit_opportunity(self):
  c=(1,100,0,100,1024,0,21.,(2.,.1,20.));self.assertEqual(foreground_cue(c,Cost())[0],0)
 def test_compute_budget_derives_from_existing_TBT_not_fixed_55ms(self):
  self.assertAlmostEqual(compute_target(.1),.08);self.assertAlmostEqual(compute_target(.06),.04)
 def test_valid_60ms_steps_are_not_rejected_under_100ms_TBT(self):
  class Sixty:
   strata=[]
   def seconds(self,n,*a):return .06
  aa=[{'id':str(i),'members':[str(i)],'cheap_credit':100,'cheap_work':1,'arrival':i}for i in range(4)]
  ms=[{'request':str(i),'prompt':64,'generated':0,'max_tokens':1024}for i in range(4)]
  chosen,d=choose(aa,ms,{'0'},10000,32,Sixty(),target_step=compute_target(.1));self.assertEqual(len(chosen),4)
class UncertainAdmission(unittest.TestCase):
 def test_point_forecast_zero_is_not_a_certificate_for_hiding_live_work(self):
  class Fixed:
   strata=[]
   def seconds(self,n,*a):return .03
  aa=[{'id':'p','members':['p'],'cheap_credit':100,'cheap_work':1,'arrival':0}, {'id':'u','members':['u'],'cheap_credit':0,'cheap_work':30,'hard_credit_expired':False,'arrival':1}]
  ms=[{'request':r,'prompt':64,'generated':0,'max_tokens':1024}for r in ['p','u']]
  chosen,d=choose(aa,ms,{'p'},10000,32,Fixed());self.assertIn('u',chosen)
 def test_all_uncertain_but_unexpired_work_keeps_compute_control(self):
  class Fixed:
   strata=[]
   def seconds(self,n,*a):return .03
  aa=[{'id':'u','members':['u'],'cheap_credit':0,'cheap_work':30,'hard_credit_expired':False,'arrival':1}]
  ms=[{'request':'u','prompt':64,'generated':0,'max_tokens':1024}]
  chosen,d=choose(aa,ms,set(),10000,32,Fixed());self.assertEqual(d['mode'],'credit_compute_admission')
 def test_latency_prompt_credit_is_not_diluted_over_the_full_stream(self):
  value,seconds=foreground_cue((0,6000,0,1024,1024,6000,0.,(2.,.1,20.)),Cost())
  self.assertGreater(value,6000);self.assertLess(seconds,2.)

if __name__=='__main__':unittest.main(verbosity=2)
