import unittest
from chain_forecast import StageForecaster
from service_model import ServiceModel,stage_seconds,waves
from replan_events import ReplanEvents
from chain_admission import select


class ServiceTests(unittest.TestCase):
    def test_half_second_is_not_clamped_to_point_two(self):
        m=ServiceModel();m.observe_round(.5)
        self.assertGreaterEqual(m.unit(1),.5)
        m.observe_execution({'members':1,'context_tokens':512},.5)
        self.assertGreaterEqual(m.unit(1,512),.5)
    def test_busy_round_extends_matching_executor_stratum(self):
        m=ServiceModel();m.observe_execution({'members':1,'context_tokens':512},.04)
        self.assertAlmostEqual(m.unit(1,512),.04)
        m.observe_round(.5)
        self.assertGreaterEqual(m.unit(1,512),.5)

    def test_batch_context_and_prefill_have_separate_evidence(self):
        m=ServiceModel()
        m.observe_execution({'members':1,'context_tokens':256},.04)
        m.observe_execution({'members':4,'context_tokens':8192},.16)
        m.observe_execution({'members':4,'context_tokens':8192,'prefill_tokens':1024},.6)
        self.assertGreater(m.unit(4,8192),m.unit(1,256))
        self.assertGreater(m.unit(4,8192,1024),m.unit(4,8192))
    def test_new_service_evidence_invalidates_cached_units(self):
        m=ServiceModel();m.observe_execution({'members':1},.1)
        self.assertAlmostEqual(m.unit(1),.1)
        m.observe_execution({'members':1},.5)
        self.assertAlmostEqual(m.unit(1),.5)

    def test_scheduler_time_is_charged(self):
        m=ServiceModel();m.observe_execution({'members':1,'scheduler_seconds':.3},.2)
        self.assertAlmostEqual(m.unit(1),.5)
    def test_foreground_recovery_uses_current_token_budget(self):
        m=ServiceModel();m.observe_round(1.)
        member={'prompt':3000,'generated':0,'prefill_tokens':3000}
        self.assertEqual(m.recovery(member),2.)
        m.set_token_budget(1000)
        self.assertEqual(m.recovery(member),3.)

    def test_transition_samples_are_retained(self):
        m=ServiceModel();m.observe_execution({'members':1,'transition':True},.8)
        self.assertEqual(m.unit(1,transition=True),.8)
        self.assertEqual(m.stats['transition_samples'],1)
    def test_recompute_and_prefill_use_integer_token_budget_waves(self):
        m=ServiceModel();m.observe_round(1.)
        t={'prompt':3000,'generated':0,'remaining':0,'max_tokens':100,'prefill_tokens':3000}
        seconds,_=stage_seconds([t],m,1000,4,16,2048)
        self.assertEqual(seconds,2.)
    def test_resource_limited_waves_do_not_hide_remaining_members(self):
        ts=[{'request':str(i),'prompt':1,'generated':0,'remaining':100,'max_tokens':100} for i in range(2)]
        m=ServiceModel();m.observe_round(1.)
        together,n=stage_seconds(ts,m,100,2,16,2048)
        serial,k=stage_seconds(ts,m,100,1,16,2048)
        partial,p=stage_seconds(ts,m,100,2,16,2048,{'0'})
        self.assertEqual((together,serial,partial),(100.,200.,200.));self.assertEqual((n,k,p),(1,2,2))
    def test_infeasible_stage_is_not_a_timely_root(self):
        ts=[{'prompt':10000,'generated':0,'remaining':100,'max_tokens':100}]
        self.assertIsNone(waves(ts,10,4,16))
    def test_foreground_recovery_is_positive_for_waiting_and_swapped(self):
        m=ServiceModel();v={'prompt':3000,'generated':100,'prefill_tokens':3100}
        self.assertGreater(m.recovery(v),0)
        self.assertGreater(m.recovery({**v,'swapped':True}),0)
        self.assertEqual(m.recovery({**v,'resident':True}),0)


class ForecastTests(unittest.TestCase):
    def fixture(self):
        data={'neighbors':1,'prefill_tokens_per_decode_unit':256,'interstage_gap_decode_units':0,
              'forecast_guard_decode_units':0,'service_safety_factor':1.,
              'records':[{'id':1,'stages':[{'inputs':[1,1],'outputs':[100,100]}]}]}
        f=StageForecaster(data);meta={'stages':1,'widths':[2],'observed':{},'neighbor_cache':{}}
        views=[{'request':str(i),'member':i,'declared_input':1,'prompt':1,'generated':0,'max_tokens':100,'prefill_tokens':0,'arrival':0} for i in range(2)]
        m=ServiceModel();m.observe_round(1.)
        g=f.estimate(meta,0,views,150.,service=m,capacity=100,slots=2)
        g.update(chain=1,stage=0,deadline=150.,arrival=0.,members=views)
        return f,g
    def test_partial_selection_recomputes_probability_and_credit(self):
        f,g=self.fixture();self.assertEqual(g['probability'],1.)
        partial=f.revalue(g,{'0'})
        self.assertEqual(partial['critical_mean_seconds'],200.)
        self.assertEqual(partial['probability'],0.);self.assertEqual(partial['expected_credited_tokens'],0.)
    def test_two_members_selected_retains_parallel_probability(self):
        f,g=self.fixture();full=f.revalue(g,{'0','1'})
        self.assertEqual(full['critical_mean_seconds'],100.);self.assertEqual(full['probability'],1.)
    def test_capacity_selector_cannot_reuse_full_parallel_credit(self):
        f,g=self.fixture();selected,rows=select([g],100,1,forecaster=f)
        self.assertIsNone(selected);self.assertEqual(rows[0]['probability'],0.)
        self.assertEqual(rows[0]['expected_credited_tokens'],0.)
    def test_foreground_reservation_changes_chain_waves(self):
        f,g=self.fixture();bg=[{'prompt':1,'generated':0,'max_tokens':100,'remaining':100}]
        partial=f.revalue(g,{'0'},bg)
        self.assertEqual(partial['probability'],0.)
    def test_future_stages_are_also_wave_limited(self):
        f,g=self.fixture();state=g['_forecast'];scenario=state['scenarios'][0]
        scenario['stages'].append([dict(t) for t in scenario['stages'][0]])
        state['slack']=350;state['slots']=1
        result=f._value(state)
        self.assertEqual(result['critical_mean_seconds'],400.)
        self.assertEqual(result['probability'],0.)
    def test_unsubmitted_siblings_are_costed_but_never_physically_selected(self):
        f,g=self.fixture();task=g['_forecast']['scenarios'][0]['stages'][0][1];task['request']=None
        result=f.revalue(g,{'0'})
        self.assertEqual(result['critical_mean_seconds'],200.)


class EventTests(unittest.TestCase):
    def test_arrival_completion_revision_configuration_and_state_all_invalidate(self):
        for key in ['membership_stage','prediction_revision','physical_state','configuration','mode_epoch','service_quantum']:
            e=ReplanEvents();e.commit({key:1},1,0,[10],[])
            self.assertIn(key,e.reasons({key:2},2,.01))
    def test_deadline_before_fifty_ticks(self):
        e=ReplanEvents();e.commit({},1,0,[.5],[])
        self.assertIn('deadline_wait_or_quantum',e.reasons({},2,.6))
    def test_capacity_release_not_routine_decode_consumption(self):
        e=ReplanEvents();e.reasons({'free_capacity':(100,50)},1,0)
        e.commit({'free_capacity':(100,50)},1,0,[],[])
        self.assertFalse(e.reasons({'free_capacity':(90,50)},2,.01))
        self.assertIn('capacity_release',e.reasons({'free_capacity':(95,50)},3,.02))
        self.assertFalse(e.reasons({'free_capacity':(94,50)},4,.03))

    def test_fifty_ticks_is_fallback(self):
        e=ReplanEvents();e.commit({},1,0,[],[])
        self.assertFalse(e.reasons({},50,.1))
        self.assertEqual(e.reasons({},51,.1),['50_tick_fallback'])
    def test_preemption_and_mode_explicit_invalidations(self):
        e=ReplanEvents();e.commit({},1,0,[],[]);e.invalidate('physical_preemption')
        self.assertIn('physical_preemption',e.reasons({},2,.01))
    def test_past_deadlines_do_not_cause_perpetual_replanning(self):
        e=ReplanEvents();e.commit({},1,5,[1,2,6],[])
        self.assertEqual(e.next_due,6)
        self.assertFalse(e.reasons({},2,5.1))


if __name__=='__main__':unittest.main(verbosity=2)
