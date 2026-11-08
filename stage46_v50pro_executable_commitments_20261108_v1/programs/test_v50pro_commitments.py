import unittest
from service_commitments import Commitments

def views(*rows):return {r:{'generated':n,'running':run,'root':root,'expired':False}for r,n,run,root in rows}
class Contract(unittest.TestCase):
 def test_service_quantum_blocks_new_arrival_eviction(self):
  c=Commitments();v=views(('r',0,True,'root'),('n',0,False,'new'));c.observe(v,0.)
  selected,d=c.resolve({'n'},v,.1,fits=lambda ids:len(ids)<=1);self.assertEqual(selected,{'r'});self.assertEqual(c.stats['eviction_blocked'],1)
 def test_token_quantum_releases_and_restoration_deadline_cannot_be_extended(self):
  c=Commitments();v=views(('r',0,True,'root'),('n',0,False,'new'));c.observe(v,0.)
  v['r']['generated']=50;s,d=c.resolve({'n'},v,.5);self.assertEqual(s,{'n'});due=c.deferred['r']
  v['r']['running']=False;v['n']['running']=True;c.resolve({'n'},v,1.);self.assertEqual(c.deferred['r'],due)
  s,d=c.resolve({'n'},v,2.);self.assertIn('r',s);self.assertFalse(c.deferred);self.assertEqual(d['reason'],'restoration_obligation')
 def test_root_cooldown_covers_a_different_sibling(self):
  c=Commitments();c.last_evicted['root']=0.;v=views(('sibling',100,True,'root'),('n',0,False,'new'))
  c.observe(v,0.);s,d=c.resolve({'n'},v,2.,fits=lambda ids:len(ids)<=1);self.assertEqual(s,{'sibling'})
 def test_fallback_cannot_bypass_quantum(self):
  c=Commitments();v=views(('r',0,True,'root'));s,d=c.resolve(None,v,0.);self.assertEqual(s,{'r'})
 def test_expired_credit_does_not_retain_a_root_cooldown_protection(self):
  c=Commitments();c.last_evicted['root']=0.;v=views(('r',0,True,'root'),('n',0,False,'new'));v['r']['expired']=True
  selected,d=c.resolve({'n'},v,.2);self.assertEqual(selected,{'n'});self.assertIn('r',c.deferred)
 def test_finished_request_removes_obligation(self):
  c=Commitments();c.deferred['finished']=.1;c.resolve(None,{},1.);self.assertFalse(c.deferred)
 def test_infeasible_restoration_is_explicit_failure(self):
  c=Commitments();c.deferred['r']=0.;v=views(('r',0,False,'root'))
  with self.assertRaises(RuntimeError):c.resolve(set(),v,1.,fits=lambda _:False)

class RestorationService(unittest.TestCase):
 def test_restored_waiter_cannot_be_hidden_before_first_actual_token(self):
  c=Commitments();v=views(('r',0,False,'root'),('n',10,True,'new'));c.deferred['r']=0.
  s,d=c.resolve({'n'},v,1.);self.assertIn('r',s);self.assertIn('r',c.restoring)
  s,d=c.resolve({'n'},v,1.1);self.assertIn('r',s)
  v['r']['running']=True;v['r']['generated']=1;c.observe(v,1.2)
  self.assertNotIn('r',c.restoring);self.assertEqual(c.stats['restoration_service_confirmed'],1)

class ProtectedAdmission(unittest.TestCase):
 def test_protection_does_not_hide_a_nonpreemptive_feasible_arrival(self):
  c=Commitments();v=views(('r',0,True,'root'),('n',0,False,'new'));c.observe(v,0.)
  chosen,d=c.resolve({'n'},v,.1);self.assertEqual(chosen,{'r','n'});self.assertEqual(d['reason'],'protected_nonpreemptive_admission')
 def test_physical_free_space_does_not_override_compute_guard(self):
  c=Commitments();v=views(('r',0,True,'root'),('n',0,False,'new'));c.observe(v,0.)
  chosen,d=c.resolve({'n'},v,.1,coexistence_allowed=lambda ids:len(ids)<=1);self.assertEqual(chosen,{'r'})

class RestorationCollateral(unittest.TestCase):
 def test_restore_keeps_unrelated_running_members_if_resources_fit(self):
  c=Commitments();v=views(('r',0,False,'root'),('b',100,True,'peer'));c.serving['b']=(0,0.);c.deferred['r']=1.
  chosen,d=c.resolve({'b'},v,2.);self.assertEqual(chosen,{'r','b'});self.assertEqual(c.stats['restoration_capacity_eviction'],0)
 def test_capacity_forced_restore_displacement_has_its_own_obligation(self):
  c=Commitments();v=views(('r',0,False,'root'),('b',100,True,'peer'));c.serving['b']=(0,0.);c.deferred['r']=1.
  chosen,d=c.resolve({'b'},v,2.,fits=lambda x:len(x)<=1);self.assertEqual(chosen,{'r'});self.assertIn('b',c.deferred);self.assertEqual(c.stats['restoration_capacity_eviction'],1)

if __name__=='__main__':unittest.main(verbosity=2)
