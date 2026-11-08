import unittest
from step_cost import StepCost,CostSnapshot
class CostContract(unittest.TestCase):
 def test_preempted_transition_is_usable(self):
  m=StepCost();m.observe_execution({'members':0,'transition':True,'preempted':3,'swap_out_blocks':10},.07)
  s=m.snapshot(.02,1);self.assertEqual(s.transition_seconds(3,160),.07);self.assertEqual(len(s.transitions),1)
 def test_multiple_swap_rounds_charged(self):
  m=StepCost();m.observe_execution({'members':0,'transition':True,'preempted':3,'swap_out_blocks':10},.07)
  self.assertAlmostEqual(m.snapshot(.02,1).transition_seconds(7,480),.21)
 def test_small_recompute_has_at_least_one_executor_iteration(self):
  s=StepCost().snapshot(.02,1);self.assertGreaterEqual(s.prefill_seconds(1),s.seconds(1,0,1))
 def test_future_wide_prefill_has_multiple_waves(self):
  class Fixed(CostSnapshot):
   def seconds(self,*args,**kw):return .1
  s=Fixed([], .02,1);self.assertAlmostEqual(s.prefill_seconds(64,64,0,slots=8),.8)
 def test_decode_does_not_use_preempted_transition_stratum(self):
  m=StepCost();m.observe(1,.02);m.observe_execution({'members':0,'transition':True,'preempted':1},.1)
  s=m.snapshot(.02,1);self.assertEqual(s.seconds(1),.02);self.assertGreaterEqual(s.transition_seconds(1),.1)
if __name__=='__main__':unittest.main(verbosity=2)
