"""Semantic counterexamples for weighted credit and cache invalidation; CPU only."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from chain_forecast import StageForecaster
from token_reward import TokenReward
from test_boundaries import donor, forecaster, meta, member


class TokenForecast(unittest.TestCase):
    def test_condition_reward_on_the_same_timely_donor(self):
        f=forecaster([donor(output=10),donor(output=1000,identifier=2)])
        r=f.estimate(meta(),0,[member()],2.,.03)
        self.assertEqual(r['probability'],.5)
        self.assertEqual(r['expected_credited_tokens'],(128+8*10)/2)
        self.assertAlmostEqual(r['score'],104/505)
        self.assertLess(r['expected_credited_tokens'],r['probability']*r['mean_chain_tokens'])

    def test_completed_prefix_is_reward_not_remaining_work(self):
        f=forecaster([donor(depth=2,output=10)])
        small=meta(depth=2);small['observed']={0:{0:{'declared_input':256,'output':10,'reward_input':100,'reward_output':10}}}
        big=deepcopy(small);big['observed'][0][0]['reward_output']=1000
        a=f.estimate(small,1,[member()],1000,.03)
        b=f.estimate(big,1,[member()],1000,.03)
        self.assertEqual(a['remaining_work'],b['remaining_work'])
        self.assertEqual(b['expected_credited_tokens']-a['expected_credited_tokens'],8*990)
        self.assertGreater(b['score'],a['score'])

    def test_completed_current_sibling_reward_is_counted_once(self):
        f=forecaster([donor(width=3,output=100)]);m=meta(width=3)
        m['observed']={0:{1:{'declared_input':256,'output':100,'reward_input':256,'reward_output':100}}}
        r=f.estimate(m,0,[member()],1000,.03)
        self.assertEqual(r['remaining_work'],201)
        self.assertEqual(r['expected_credited_tokens'],(128+800)+(256+800)+(256+800))

    def test_expiry_skips_neighbor_search_and_never_reuses_probability(self):
        f=forecaster([donor()]);m=meta();v=[member()]
        self.assertGreater(f.estimate(m,0,v,1000,.03)['score'],0)
        with patch.object(f,'neighbors',side_effect=AssertionError('expired search')):
            r=f.estimate(m,0,v,0,.03)
        self.assertEqual(r['probability'],0);self.assertEqual(r['score'],0)
        self.assertEqual(r['reason'],'whole_chain_deadline_expired')
        self.assertGreater(f.estimate(m,0,v,1000,.03)['score'],0)

    def test_future_cache_invalidates_width_and_public_cap(self):
        f=forecaster([donor(depth=2,output=200)]);m=meta(depth=2);v=[member()]
        a=f.estimate(m,0,v,1000,.03)
        b=f.estimate(m,0,v,1000,.03)
        self.assertEqual(a,b);self.assertGreater(f.stats['future_cache_hits'],0)
        m['widths'][1]=2;c=f.estimate(m,0,v,1000,.03)
        self.assertGreater(c['remaining_work'],a['remaining_work'])
        m['future_max_tokens']=10;d=f.estimate(m,0,v,1000,.03)
        self.assertLess(d['remaining_work'],c['remaining_work'])

    def test_bounded_cache_matches_disabled_cache_across_progress_and_prefix(self):
        records=[donor(depth=3,output=100),donor(depth=3,output=200,identifier=2)]
        a=forecaster(records);b=StageForecaster(a.model,future_cache_capacity=0)
        a.future_cache_capacity=2
        for stage in range(3):
            m=meta(depth=3)
            for prior in range(stage):m['observed'][prior]={0:{'declared_input':256,'output':100}}
            for generated in [0,50,150,1000]:
                for slack in [1000.,3.,0.]:
                    av=a.estimate(deepcopy(m),stage,[member(generated=generated)],slack,.03)
                    bv=b.estimate(deepcopy(m),stage,[member(generated=generated)],slack,.03)
                    self.assertEqual(av,bv)
                    self.assertLessEqual(len(a.future_cache),2)

    def test_new_current_sibling_invalidates_neighbor_signature(self):
        f=forecaster([donor(width=2)]);m=meta(width=2)
        a=f.estimate(m,0,[member()],1000,.03)
        misses=f.stats['neighbor_cache_misses']
        b=f.estimate(m,0,[member(),member('1')],1000,.03)
        self.assertGreater(f.stats['neighbor_cache_misses'],misses)
        self.assertEqual(a['estimated_unsubmitted_members'],1)
        self.assertEqual(b['estimated_unsubmitted_members'],0)

    def test_unknown_support_is_not_expiry(self):
        r=forecaster([donor(depth=1)]).estimate(meta(depth=3),0,[member()],1000,.03)
        self.assertFalse(r['supported']);self.assertNotEqual(r['reason'],'whole_chain_deadline_expired')


class TextCredit(unittest.TestCase):
    def test_model_tokenization_without_special_tokens_and_bounded_cache(self):
        class Tokenizer:
            def encode(self,text,add_special_tokens):
                assert not add_special_tokens
                return text.split()
        c=TokenReward(capacity=2);t=Tokenizer();c.bind(t)
        self.assertEqual(c.count('one two',999),2)
        self.assertEqual(c.count('one two',999),2)
        self.assertEqual(c.stats['token_cache_hits'],1)
        for text in ['three','four']:c.count(text,0)
        self.assertEqual(len(c.cache),2)
        c.bind(Tokenizer());self.assertFalse(c.cache)


if __name__=='__main__':unittest.main(verbosity=2)
