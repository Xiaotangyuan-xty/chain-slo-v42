"""CPU tests for only the three requested forecast boundaries and mixed packing."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

from chain_forecast import StageForecaster
from joint_admission import select_joint


def donor(depth=1, width=1, output=100, identifier=1):
    return {'id': identifier, 'stages': [{'inputs': [256]*width, 'outputs': [output]*width}
                                       for _ in range(depth)]}


def forecaster(records):
    return StageForecaster({'records': records, 'neighbors': 32,
                            'prefill_tokens_per_decode_unit': 256,
                            'interstage_gap_decode_units': 32,
                            'forecast_guard_decode_units': 50,
                            'service_safety_factor': 1.15})


def meta(depth=1, width=1):
    return {'stages': depth, 'widths': [width]*depth, 'observed': {}, 'neighbor_cache': {}}


def member(rid='0', generated=0, cap=1024):
    return {'request': rid, 'member': int(rid), 'declared_input': 256,
            'prompt': 128, 'generated': generated, 'max_tokens': cap,
            'prefill_tokens': 0, 'arrival': 100.}


class ForecastBoundaries(unittest.TestCase):
    def test_unsubmitted_siblings_count_as_work(self):
        r=forecaster([donor(width=3)]).estimate(meta(width=3),0,[member()],1000,.03)
        self.assertEqual(r['estimated_unsubmitted_members'],2)
        self.assertEqual(r['remaining_work'],302)

    def test_completed_sibling_is_not_charged_again(self):
        m=meta(width=3);m['observed']={0:{1:{'declared_input':256,'output':100}}}
        r=forecaster([donor(width=3)]).estimate(m,0,[member()],1000,.03)
        self.assertEqual(r['estimated_unsubmitted_members'],1)
        self.assertEqual(r['remaining_work'],201)

    def test_actual_arrival_replaces_virtual_work_without_double_count(self):
        f=forecaster([donor(width=2)]);m=meta(width=2)
        first=f.estimate(m,0,[member()],1000,.03)
        v=member('1');v['prefill_tokens']=256
        second=f.estimate(m,0,[member(),v],1000,.03)
        self.assertEqual(first['remaining_work'],second['remaining_work'])
        self.assertEqual(second['estimated_unsubmitted_members'],0)

    def test_backoff_near_public_cap_only_has_four_tokens_left(self):
        r=forecaster([donor(output=100)]).estimate(meta(),0,[member(generated=1020)],1000,.03)
        self.assertEqual(r['remaining_work'],4)
        self.assertEqual(r['current_remaining_tokens'],4)

    def test_backoff_at_public_cap_has_no_fictitious_output_work(self):
        r=forecaster([donor()]).estimate(meta(),0,[member(generated=1024)],1000,.03)
        self.assertEqual(r['current_remaining_tokens'],0)

    def test_current_members_use_their_own_caps(self):
        r=forecaster([donor(width=2,output=200)]).estimate(meta(width=2),0,
                      [member('0',cap=10),member('1',cap=1000)],1000,.03)
        self.assertEqual(r['remaining_work'],210)

    def test_future_cap_is_not_the_smallest_current_member_cap(self):
        r=forecaster([donor(depth=2,output=200)]).estimate(meta(depth=2),0,[member(cap=10)],1000,.03)
        self.assertEqual(r['remaining_work'],211)

    def test_short_donor_cannot_silently_omit_tail(self):
        r=forecaster([donor(depth=4)]).estimate(meta(depth=6),0,[member()],1000,.03)
        self.assertFalse(r['supported']);self.assertEqual(r['probability'],0)
        self.assertIsNone(r['remaining_work'])

    def test_later_stage_without_support_does_not_raise(self):
        r=forecaster([donor(depth=4)]).estimate(meta(depth=8),7,[member()],1000,.03)
        self.assertFalse(r['supported']);self.assertEqual(r['score'],0)

    def test_longer_donor_covers_every_declared_stage(self):
        r=forecaster([donor(depth=6)]).estimate(meta(depth=5),0,[member()],1000,.03)
        self.assertTrue(r['supported']);self.assertEqual(r['remaining_work'],504)

    def test_nonboundary_predictions_keep_v45_work_formula(self):
        reference=Path(__file__).resolve().parents[2]/'stage15_stable_chain_v45_parallel_20261101_v2/round01/programs/chain_forecast.py'
        spec=importlib.util.spec_from_file_location('v45_forecast_reference',reference)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for depth in [4,6]:
            f=forecaster([donor(depth,2,output=200),donor(depth,2,output=400,identifier=2)])
            old=module.StageForecaster(f.model)
            for stage in range(depth):
                for generated in [0,50,150]:
                    views=[member('0',generated),member('1',generated)]
                    a=old.estimate(deepcopy(meta(depth,2)),stage,views,150,.03)
                    b=f.estimate(deepcopy(meta(depth,2)),stage,views,150,.03)
                    self.assertEqual({k:v for k,v in a.items() if k!='score'},
                                     {k:b[k] for k in a if k!='score'})


class MixedPacking(unittest.TestCase):
    def goal(self,rid='0',probability=1):
        return {'chain':int(rid),'stage':0,'probability':probability,'score':probability/100,
                'remaining_work':100,'critical_mean_seconds':1.,'deadline':1000.,
                'arrival':100.,'members':[member(rid)]}

    def test_foreground_does_not_disable_chain_selection(self):
        selected,_,fg=select_joint([self.goal()],[member('1')],24,4)
        self.assertEqual(selected,{'0','1'});self.assertEqual(fg,{'1'})

    def test_virtual_siblings_do_not_consume_physical_slots(self):
        g=self.goal();g['estimated_unsubmitted_members']=100
        selected,_,_=select_joint([g],[member('1')],24,2)
        self.assertEqual(selected,{'0','1'})

    def test_v45_chain_lease_is_not_spent_on_foreground(self):
        selected,_,_=select_joint([self.goal()],[member('1')],12,2,protected={'0'})
        self.assertIn('0',selected)

    def test_zero_probability_keeps_v45_native_drain_signal(self):
        selected,_,_=select_joint([self.goal(probability=0)],[member('1')],24,4)
        self.assertIsNone(selected)

    def test_foreground_borrows_unused_chain_capacity(self):
        selected,_,fg=select_joint([self.goal()],[member('1'),member('2'),member('3')],48,4)
        self.assertEqual(len(selected),4);self.assertEqual(len(fg),3)


if __name__ == '__main__':
    unittest.main()
