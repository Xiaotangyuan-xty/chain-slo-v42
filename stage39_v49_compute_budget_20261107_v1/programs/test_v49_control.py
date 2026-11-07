import math
import unittest
from credit_curve import CreditCurve, forecast_curve
from chain_forecast import StageForecaster
from mixed_control import plan, foreground_value, eligible_decode_count
from step_cost import StepCost


def member(rid, generated=0):
    return {'request':rid,'prompt':128,'generated':generated,'max_tokens':1024,
            'prefill_tokens':0,'native_priority':(0.,0.),'arrival':0.}


def action(rid, credit=1000., work=100., critical=40., slack=1.):
    return {'id':rid,'kind':'compound','members':[rid],'arrival':0.,
            'curve':CreditCurve([(critical,work,credit)],slack)}


def costs(points):
    m=StepCost()
    for n,t in points:
        for _ in range(8):m.observe(n,t)
    return m.snapshot(.02,1)


class ComputeControl(unittest.TestCase):
    def test_memory_room_is_not_free_compute(self):
        selected,detail=plan([action('valuable'),action('other')],
                             [member('valuable'),member('other')],set(),10000,32,.02,
                             costs=costs([(1,.02),(2,.04)]))
        self.assertEqual(len(selected),1)
        self.assertLess(detail['selected_charge'],10000)
        self.assertGreater(detail['objective_rate_proxy'],0)

    def test_real_batching_benefit_can_admit_multiple_actions(self):
        selected,_=plan([action('a',slack=10),action('b',slack=10)],
                        [member('a'),member('b')],set(),10000,32,.02,
                        costs=costs([(1,.02),(2,.025)]))
        self.assertEqual(selected,{'a','b'})

    def test_expired_resident_is_not_permanently_protected(self):
        selected,detail=plan([action('live'),action('expired',slack=-1)],
                             [member('live'),member('expired',900)],{'expired'},10000,32,.02,
                             previous={'expired'},lease_remaining={'expired':49},
                             costs=costs([(1,.02),(2,.04)]))
        self.assertEqual(selected,{'live'});self.assertEqual(detail['evicted'],['expired'])

    def test_cold_model_does_not_backfill_zero_value(self):
        selected,detail=plan([action('live'),action('zero',credit=0)],
                             [member('live'),member('zero')],{'live','zero'},10000,32,.02)
        self.assertEqual(selected,{'live'});self.assertFalse(detail['backfill'])

    def test_measured_flat_step_can_backfill_prefilled_resident(self):
        cs=costs([(1,.02),(2,.02)])
        selected,detail=plan([action('live'),action('zero',credit=0)],
                             [member('live'),member('zero')],{'live','zero'},10000,32,.02,costs=cs)
        self.assertEqual(selected,{'live','zero'});self.assertEqual(detail['backfill'],['zero'])
        ms=[member('live'),member('zero')];ms[1]['prefill_tokens']=128
        selected,_=plan([action('live'),action('zero',credit=0)],ms,{'live','zero'},10000,32,.02,costs=cs)
        self.assertEqual(selected,{'live'})

    def test_full_stage_packet_is_atomic(self):
        a=action('root',slack=10);a['members']=['x','y']
        selected,_=plan([a],[member('x'),member('y')],set(),1000,32,.02,
                        costs=costs([(1,.02),(2,.03)]))
        self.assertEqual(selected,{'x','y'})
        selected,_=plan([a],[member('x'),member('y')],set(),1000,1,.02)
        self.assertIsNone(selected)  # required work remains eligible for native fallback

    def test_physical_limits_and_no_value_never_drop_work(self):
        selected,_=plan([action('x')],[member('x')],set(),1,1,.02)
        self.assertIsNone(selected)
        selected,detail=plan([action('x',credit=0)],[member('x')],set(),1000,32,.02)
        self.assertIsNone(selected);self.assertEqual(detail['reason'],'native_required_work_fallback')

    def test_continuity_has_no_unconditional_resident_rule(self):
        cs=costs([(1,.02),(2,.04)])
        selected,detail=plan([action('old',credit=100),action('new',credit=100000)],
                             [member('old',10),member('new')],{'old'},10000,32,.02,costs=cs,
                             previous={'old'},lease_remaining={'old':40})
        self.assertEqual(selected,{'new'})
        self.assertIn('old',detail['evicted'])

    def test_prefix_is_conditional_and_generated_latency_credit_is_not_reearned(self):
        curve=CreditCurve([(100,100,10000)],1.)
        self.assertEqual(curve.credit(.02),0)
        self.assertEqual(curve.credit(.005),10000)
        v=foreground_value(0,10000,50,100,1024,0,0.,(2.,.1,20.),.03)
        self.assertEqual(v['expected_credit'],8*50)

    def test_cost_sample_filtering_monotonicity_and_bounded_storage(self):
        m=StepCost(capacity=8)
        for x in [float('nan'),0.,2.]:self.assertFalse(m.observe(2,x))
        for _ in range(30):m.observe(2,.04);m.observe(4,.03)
        self.assertEqual(len(m.samples[2]),8)
        p=m.snapshot(.02,1)
        self.assertLessEqual(p.seconds(2),p.seconds(4))
        self.assertGreater(p.seconds(8),p.seconds(4))

    def test_observed_fixed_cost_allows_learning_beyond_seen_batch_width(self):
        cs=costs([(1,.02),(2,.025)])
        self.assertAlmostEqual(cs.seconds(4),.035)
        actions=[action(str(i),slack=10)for i in range(4)]
        selected,_=plan(actions,[member(str(i))for i in range(4)],set(),10000,32,.02,costs=cs)
        self.assertEqual(selected,{'0','1','2','3'})

    def test_curve_matches_original_forecaster_without_any_test_targets(self):
        model={'neighbors':2,'prefill_tokens_per_decode_unit':256,'interstage_gap_decode_units':32,
               'forecast_guard_decode_units':50,'service_safety_factor':1.15,
               'records':[{'id':1,'stages':[{'inputs':[128],'outputs':[60]}, {'inputs':[80],'outputs':[20]}]},
                          {'id':2,'stages':[{'inputs':[128],'outputs':[90]}, {'inputs':[80],'outputs':[40]}]}]}
        f=StageForecaster(model);meta={'stages':2,'widths':[1,1],'observed':{},'neighbor_cache':{}}
        views=[{'member':0,'declared_input':128,'prompt':128,'reward_input':127,
                'generated':20,'prefill_tokens':0,'max_tokens':1024}]
        for slack in [0.,1.,5.,100.]:
            curve=forecast_curve(f,meta,0,views,slack)
            for unit in [.01,.03,.1]:
                original=f.estimate(meta,0,views,slack,unit)
                self.assertAlmostEqual(curve.credit(unit),original['expected_credited_tokens'])
                self.assertAlmostEqual(curve.view(unit)['probability'],original['probability'])

    def test_cumulative_deadlines_keep_late_ttft_recovery(self):
        self.assertGreater(eligible_decode_count(0,200,3.,0.,.03,2.,.1),0)


if __name__=='__main__':unittest.main(verbosity=2)
