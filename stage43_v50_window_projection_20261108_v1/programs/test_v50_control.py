import unittest
from unittest.mock import patch
from stage_scenarios import Stage,Scenario,Scenarios
from window_control import plan,candidate_sets
from window_projection import WindowProjector,foreground_credit
from step_cost import StepCost


class Cost:
    def __init__(self,points=None):self.points=points or{1:.02,2:.03,32:.072}
    def seconds(self,n,context_tokens=0.,prefill_tokens=0.,transition=False):
        if n<=0:return 0.
        if n in self.points:return self.points[n]
        if n<max(self.points):
            lo=max(k for k in self.points if k<n);hi=min(k for k in self.points if k>n)
            return self.points[lo]+(self.points[hi]-self.points[lo])*(n-lo)/(hi-lo)
        return self.points[max(self.points)]*n/max(self.points)
    def transition_seconds(self,n,context=0.):return n*.001
    def describe(self):return {'fixture':True}


def action(aid,width=1,decode=10,credit=1000,slack=2.):
    return {'id':aid,'kind':'compound','members':[aid+str(i)for i in range(width)],'arrival':0.,
        'native_priority':(0.,),'cheap_credit':credit,'cheap_work':decode*width,
        'scenarios':Scenarios([Scenario((Stage(width,decode,0,0),),credit)],slack)}


def members(actions):return {r:{'request':r,'prompt':0,'generated':0,'max_tokens':100,'state':'waiting'}for a in actions for r in a['members']}


class Projection(unittest.TestCase):
    def test_width1_to32_is_4p6_seconds_and_not_two(self):
        a=action('root');a['scenarios']=Scenarios([Scenario((Stage(1,50,0,0),Stage(32,50,0,0)),1000)],3)
        p=WindowProjector([a],{'root'},Cost(),10000,32,members([a]),1.,0.)
        v=p.evaluate({'root'});self.assertAlmostEqual(v['completion_seconds_by_action']['root'][0],4.6);self.assertEqual(v['credit'],0)

    def test_future_narrow_stage_recovers_from_current_congestion(self):
        a=action('root',32);a['scenarios']=Scenarios([Scenario((Stage(32,50,0,0),Stage(1,50,0,0)),1000)],5)
        p=WindowProjector([a],{'root'},Cost(),10000,32,members([a]),1.,0.)
        v=p.evaluate({'root'});self.assertAlmostEqual(v['completion_seconds_by_action']['root'][0],4.6);self.assertEqual(v['credit'],1000)

    def test_weighted_CDF_interpolation_preserves_conditional_credit(self):
        a=action('root');a['scenarios']=Scenarios([Scenario((Stage(1,10,0,0),),1000),Scenario((Stage(1,100,0,0),),10000)],1.)
        p=WindowProjector([a],{'root'},Cost(),10000,32,members([a]),1.,0.)
        v=p.evaluate({'root'});self.assertAlmostEqual(v['credit'],500+5000*(.8/1.8));self.assertAlmostEqual(v['probability_by_action']['root'],.5+.5*(.8/1.8))

    def test_planning_delay_reduces_slack(self):
        a=action('root',decode=50,slack=1.005)
        p=WindowProjector([a],{'root'},Cost(),10000,32,members([a]),1.,.01)
        self.assertEqual(p.evaluate({'root'})['credit'],0)

    def test_foreground_analytic_credit_matches_explicit_token_schedule(self):
        for generated in [0,5,50]:
            for elapsed in [0.,1.9,3.]:
                for selected in [False,True]:
                    c=(0,100,generated,80,128,0,elapsed,(2.,.1,20.));step=.04;ref=.03;window=.5;delay=.01
                    value=foreground_credit(c,selected,step,0,ref,0,window,delay)
                    remain=80-generated;first=min(remain,int(window/step))if selected else 0
                    times=[delay+k*step for k in range(1,first+1)]
                    times += [delay+window+(k-first)*ref for k in range(first+1,remain+1)]
                    prompt=100 if generated==0 and elapsed+times[0]<=2 else 0
                    expected=prompt+8*sum(generated+k>=2 and elapsed+t<=2+.1*(generated+k-1)for k,t in enumerate(times,1))
                    self.assertEqual(value,expected,(c,selected))

    def test_prefill_minimum_iteration_and_partial_service(self):
        c=(1,100,0,10,100,64,0.,(2.,.1,1.))
        self.assertEqual(foreground_credit(c,True,.02,.1,.02,.1,.05,0),180.)
        c=(1,100,0,10,100,4096,0.,(2.,.1,.15))
        self.assertEqual(foreground_credit(c,True,.02,.1,.02,.1,.05,0),0.)


class Control(unittest.TestCase):
    def test_integrated_one_for_two_KV_replacement(self):
        acts=[action('a',decode=50,credit=500,slack=1.1),action('b',decode=10,credit=450,slack=.5),action('c',decode=10,credit=450,slack=.5)]
        for a in acts:a['members']=[a['id']]
        ms=[{'request':'a','prompt':46,'generated':0,'max_tokens':100,'state':'running'},
            {'request':'b','prompt':14,'generated':0,'max_tokens':100,'state':'waiting'},
            {'request':'c','prompt':14,'generated':0,'max_tokens':100,'state':'waiting'}]
        selected,d=plan(acts,ms,{'a'},8,3,.02,costs=Cost({1:.02,2:.02,3:.02}),max_seconds=None)
        self.assertEqual(selected,{'b','c'});self.assertEqual(d['selected_charge'],8);self.assertGreater(d['paired']['delta_credit'],0)

    def test_candidate_generation_includes_joint_add_and_one_for_two(self):
        sets=list(candidate_sets({'a'},{'a'},{'a'},['a'],['b','c']))
        self.assertIn(frozenset({'a','b','c'}),sets);self.assertIn(frozenset({'b','c'}),sets)

    def test_more_than12_foreground_residents_do_not_disable_comparison(self):
        root=action('root',decode=10,credit=10000,slack=.395)
        root['members']=['r'];acts=[root];ms=[{'request':'r','prompt':0,'generated':0,'max_tokens':100,'state':'running'}]
        for i in range(31):
            rid=str(i);acts.append({'id':rid,'kind':'foreground','members':[rid],'arrival':1.,'native_priority':(1.,),
                'cheap_credit':100,'cheap_work':50,'foreground':(1,10,0,50,100,0,.4,(2.,.1,.5))})
            ms.append({'request':rid,'prompt':0,'generated':0,'max_tokens':100,'state':'running'})
        selected,d=plan(acts,ms,{m['request']for m in ms},10000,32,.02,costs=Cost({1:.020625,32:.04}),max_seconds=None)
        self.assertIsNotNone(selected,d);self.assertGreater(d['complete_sets'],1)
        self.assertGreaterEqual(len(selected),24);self.assertEqual(d['paired']['credit_by_action']['root'],10000)

    def test_anytime_keeps_complete_gain_when_next_search_times_out(self):
        acts=[action('a',credit=10),action('b',credit=100)]
        ms=list(members(acts).values());clock=[0.]
        class Projector:
            def __init__(self,*args,**kw):self.memo={}
            def evaluate(self,ids):
                ids=frozenset(ids);v=130. if ids==frozenset({'b'})else 100.
                self.memo[ids]={'credit':v,'credit_by_action':{'a':0.,'b':v},'probability_by_action':{'a':0.,'b':1.},'completion_seconds_by_action':{},'switch_seconds':0.,'complete_envelope':True}
                if ids==frozenset({'b'}):clock[0]=.009
                return self.memo[ids]
        with patch('window_control.time.perf_counter',side_effect=lambda:clock[0]),patch('window_control.WindowProjector',Projector):
            selected,d=plan(acts,ms,set(),10000,2,.02,costs=Cost(),max_seconds=.008)
        self.assertEqual(selected,{'b0'});self.assertTrue(d['budget_exhaustion_kept_completed_improvement'])

    def test_unaffected_credit_constant_does_not_block_a_real_marginal_gain(self):
        acts=[action('a',credit=10),action('b',credit=100),action('u',credit=1e6)]
        acts[2]['cheap_work']=1e10
        class Projector:
            def __init__(self,*args,**kw):self.memo={}
            def evaluate(self,ids):
                ids=frozenset(ids);gain=30. if 'a'not in ids else 0.
                v={'credit':1000100.+gain,'credit_by_action':{'a':0.,'b':100.+gain,'u':1e6},
                   'probability_by_action':{'a':0.,'b':1.,'u':1.},'completion_seconds_by_action':{},'switch_seconds':0.,'complete_envelope':True}
                self.memo[ids]=v;return v
        with patch('window_control.WindowProjector',Projector):
            selected,d=plan(acts,list(members(acts).values()),set(),10000,3,.02,costs=Cost(),max_seconds=None)
        self.assertIsNotNone(selected);self.assertIn('b0',selected);self.assertNotIn('a0',selected)
        self.assertEqual(d['required_credit_gain'],8.)

    def test_joint_swap_reservation_prices_only_overflow_as_recompute(self):
        acts=[action('a'),action('b'),action('new')]
        for a in acts:a['members']=[a['id']];a['resident']=a['id']!='new'
        ms={name:{'request':name,'prompt':160,'generated':16,'max_tokens':100,'prefill_tokens':0,
                  'state':'running'if name!='new'else'waiting','can_swap':True,'native_priority':(0.,)}for name in ['a','b','new']}
        p=WindowProjector(acts,{'a','b'},Cost(),10000,3,ms,1.,0.,swap_capacity_blocks=11)
        v=p.evaluate({'new'})
        self.assertEqual(v['CPU_swap_blocks_reserved'],11)
        self.assertEqual(len(v['recompute_members']),1)

    def test_unsupported_live_root_has_no_silent_zero_loss(self):
        a=action('root');a['scenarios']=Scenarios([],10,supported=False)
        selected,d=plan([a],list(members([a]).values()),set(),10000,32,.02,max_seconds=None)
        self.assertIsNone(selected);self.assertEqual(d['fallback_reason'],'unsupported_reference_scenario')

    def test_all_expired_work_returns_native_without_dropping_members(self):
        a=action('root');a['cheap_credit']=0
        selected,d=plan([a],list(members([a]).values()),set(),10000,32,.02,max_seconds=None)
        self.assertIsNone(selected);self.assertEqual(a['members'],['root0'])

    def test_cost_snapshot_index_retains_context_and_prefill_distinctions(self):
        c=StepCost()
        for f,t in [({'members':1,'context_tokens':512},.02),({'members':1,'context_tokens':8192},.08),({'members':1,'context_tokens':512,'prefill_tokens':2048},.1)]:c.observe_execution(f,t)
        s=c.snapshot(.02,1)
        self.assertEqual(s.seconds(1,512),.02);self.assertEqual(s.seconds(1,8192),.08);self.assertEqual(s.seconds(1,512,2048),.1)
        self.assertIn('index',s.describe())


if __name__=='__main__':unittest.main(verbosity=2)
