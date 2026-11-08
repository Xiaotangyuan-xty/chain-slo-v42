import unittest
from compute_admission import choose
from step_cost import CostSnapshot
class Measured:
 strata=[((n,0,0,False,False,False),.02+n*.003,10)for n in range(1,33)]
 def seconds(self,n,*args):return .02+n*.003
class Admission(unittest.TestCase):
 def data(self,n=20,credit=100):
  aa=[{'id':str(i),'members':[str(i)],'cheap_credit':credit,'cheap_work':100,'arrival':i}for i in range(n)]
  ms=[{'request':str(i),'prompt':64,'generated':0,'max_tokens':1024}for i in range(n)]
  return aa,ms
 def test_compute_guard_also_bounds_admission_fallback(self):
  aa,ms=self.data();chosen,d=choose(aa,ms,set(),10000,32,Measured(),target_step=.055);self.assertLessEqual(len(chosen),11);self.assertTrue(d['compute_limited'])
 def test_existing_service_is_not_preempted_by_a_compute_observation(self):
  aa,ms=self.data();chosen,d=choose(aa,ms,{str(i)for i in range(18)},10000,32,Measured());self.assertTrue({str(i)for i in range(18)}<=chosen)
 def test_required_expired_member_can_make_progress(self):
  aa,ms=self.data(3);aa[2]['cheap_credit']=0
  chosen,d=choose(aa,ms,{'0'},10000,32,Measured(),required={'2'});self.assertIn('2',chosen)
 def test_all_expired_is_complete_required_drain(self):
  aa,ms=self.data(3,credit=0)
  for a in aa:a['hard_credit_expired']=True
  chosen,d=choose(aa,ms,set(),10000,32,Measured());self.assertTrue(chosen);self.assertEqual(d['mode'],'required_drain')
 def test_no_member_disappears_and_physical_capacity_holds(self):
  aa,ms=self.data(20);chosen,d=choose(aa,ms,set(),8,3,Measured());self.assertLessEqual(len(chosen),1);self.assertEqual(len(ms),20)
 def test_unmeasured_width_gets_bounded_online_probe(self):
  aa,ms=self.data(20);c=CostSnapshot([((1,0,0,False,False,False),.03,64)],.03,1)
  chosen,d=choose(aa,ms,{'0'},10000,32,c,target_step=.055);self.assertEqual(len(chosen),2);self.assertEqual(d['uncertain_compute_probe_admissions'],1)
if __name__=='__main__':unittest.main(verbosity=2)
