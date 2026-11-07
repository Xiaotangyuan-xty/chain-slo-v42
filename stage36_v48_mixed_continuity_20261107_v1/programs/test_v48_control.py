import json,math,unittest
from pathlib import Path
from mixed_control import eligible_decode_count,foreground_value,interruption_loss,plan
from mixed_forecast import tot_model


def member(rid):
    return {'request':rid,'prompt':128,'generated':0,'max_tokens':1024,
            'arrival':0.,'native_priority':(0.,0.)}


def action(rid,density=1.,credit=100.,prefix=0.,slack=100.):
    return {'id':rid,'members':[rid],'density':density,'expected_credit':credit,
            'stage_work':50.,'probability':float(credit>0),'critical_seconds':10.,
            'slack':slack,'prefix_at_risk':prefix,'arrival':0.}


class Control(unittest.TestCase):
    def test_cumulative_token_deadlines_and_late_TTFT_recovery(self):
        for generated in [0,1,50,500]:
            for remaining in [0,1,50,200]:
                for elapsed in [0.,2.,10.,100.]:
                    for unit in [.03,.1,.2]:
                        expected=sum(generated+k>=2 and elapsed+.04+k*unit<=2.+.1*(generated+k-1)+1e-10 for k in range(1,remaining+1))
                        self.assertEqual(eligible_decode_count(generated,remaining,elapsed,.04,unit,2.,.1),expected)
        self.assertGreater(eligible_decode_count(0,200,3.,0.,.03,2.,.1),0)

    def test_latency_never_reearns_settled_prefix_and_deadline_is_conditional(self):
        a=foreground_value(0,100,50,100,1024,0,0.,(2.,.1,20.),.03)
        self.assertEqual(a['expected_credit'],8*50)
        self.assertEqual(foreground_value(1,100,10,100,1024,0,21.,(2.,.1,20.),.03)['expected_credit'],0)
        self.assertGreater(foreground_value(1,100,10,100,1024,0,0.,(2.,.1,20.),.03)['expected_credit'],8*90)

    def test_zero_prediction_is_not_completion_and_cap_bounds_survival(self):
        a=foreground_value(0,10,100,20,110,0,0.,(2.,.1,20.),.03)
        self.assertEqual(a['work'],10)

    def test_new_positive_request_does_not_evict_resident_with_spare_capacity(self):
        selected,detail=plan([action('old',1),action('new',1000)],
                             [member('old'),member('new')],{'old'},100,32,.04)
        self.assertEqual(selected,{'old','new'});self.assertFalse(detail['evicted'])

    def test_zero_forecast_work_gets_spare_capacity(self):
        selected,detail=plan([action('live'),action('zero',0,0)],
                             [member('live'),member('zero')],{'live'},100,32,.04)
        self.assertEqual(selected,{'live','zero'});self.assertIn('zero',detail['backfill'])

    def test_switch_cost_and_conditional_prefix_can_prevent_bad_eviction(self):
        selected,detail=plan([action('old',1,prefix=10000,slack=10),action('new',2)],
                             [member('old'),member('new')],{'old'},12,1,.04)
        self.assertEqual(selected,{'old'})
        selected,detail=plan([action('old',0,0),action('new',2)],
                             [member('old'),member('new')],{'old'},12,1,.04)
        self.assertEqual(selected,{'new'});self.assertEqual(detail['evicted'],['old'])

    def test_declared_budget_is_not_exceeded_and_oversized_delegates(self):
        ms=[member(str(i))for i in range(10)];actions=[action(m['request'])for m in ms]
        selected,detail=plan(actions,ms,set(),36,3,.04)
        self.assertEqual(len(selected),3);self.assertLessEqual(detail['selected_charge'],36)
        selected,detail=plan([action('x')],[member('x')],set(),1,1,.04)
        self.assertIsNone(selected);self.assertEqual(detail['reason'],'native_oversized_fallback')

    def test_tot_donors_are_calibration_only_and_width_cap_qualified(self):
        bank=json.loads((Path(__file__).resolve().parents[1]/'assets/TOT_CALIBRATION.json').read_text())
        for key,value in bank['models'].items():
            self.assertTrue(value['evaluation_graphs_disjoint']);self.assertFalse(value['performance_or_coverage_validated'])
            model=tot_model(key);self.assertEqual(len(model.records),14)
            self.assertTrue(all([len(s['outputs'])for s in r['stages']]==[3,1,3,1]for r in model.records))
            self.assertTrue(all(0<n<=256 for r in model.records for s in r['stages']for n in s['outputs']))
        self.assertIsNone(tot_model('unknown-model'))


if __name__=='__main__':unittest.main(verbosity=2)
