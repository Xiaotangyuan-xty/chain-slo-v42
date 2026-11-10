import copy
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from gpu_idle import idle
from jitserve_goodput import compute

ROOT=Path(sys.argv[1]);sys.argv=sys.argv[:1]
class Tokenizer:
    def __call__(self,text,add_special_tokens):
        assert add_special_tokens is False
        return NS(input_ids=text.split())

class Goodput(unittest.TestCase):
    def fixture(self):
        import hashlib
        jobs=[];calls=[]
        for cid,latency,text in [(0,20.,'one two'),(1,40.0001,'one two three four'),(2,10.,'one')]:
            p='input words'
            jobs.append({'collection_id':cid,'request_type':2,'received_slo':[2*(cid+1),.1*(cid+1),20*(cid+1)],
                         'stages':[[{'request_id':0,'prompt':p}]]})
            r={'request_id':0,'success':True,'generated_text':text,'input_prompt_sha256':hashlib.sha256(p.encode()).hexdigest(),
               'client_request_start_monotonic':0.,'client_request_end_monotonic':latency,'server_ttft':1.}
            calls.append({'phase':'main','collection_id':cid,'request_type':2,'success':True,
                         'client_chain_start_monotonic':0.,'client_chain_end_monotonic':latency,'received_slo':jobs[-1]['received_slo'],
                         'stages':[{'stage_id':0,'success':True,'requests':[r]}]})
        return {'main':jobs},calls
    def test_boundary_and_whole_chain_zero_credit_and_weighting(self):
        w,c=self.fixture();r=compute(ROOT,w,c,100.,Tokenizer())
        self.assertEqual(r['compound_attained'],2)
        self.assertEqual(r['credited_weighted_tokens'],28)
        self.assertEqual(r['jitserve_token_goodput'],.28)
    def test_full_drain_denominator_includes_wait(self):
        w,c=self.fixture();a=compute(ROOT,w,c,100.,Tokenizer());b=compute(ROOT,w,c,200.,Tokenizer())
        self.assertEqual(a['credited_weighted_tokens'],b['credited_weighted_tokens'])
        self.assertEqual(a['jitserve_token_goodput'],2*b['jitserve_token_goodput'])
    def test_failed_chain_never_gets_credit(self):
        w,c=self.fixture();c[0]['success']=False
        self.assertEqual(compute(ROOT,w,c,100.,Tokenizer())['credited_weighted_tokens'],10)
    def test_partial_stage_is_rejected(self):
        w,c=self.fixture();c[0]['stages']=[]
        with self.assertRaises(AssertionError):compute(ROOT,w,c,100.,Tokenizer())
    def test_driver_bookkeeping_and_unrelated_job(self):
        self.assertTrue(idle('0, UUID, 4 MiB','',0))
        self.assertTrue(idle('0, UUID, 0 MiB','',0))
        self.assertFalse(idle('0, UUID, 4 MiB','123, 4 MiB',0))
        self.assertFalse(idle('0, UUID, 17 MiB','',0))
        self.assertFalse(idle('0, UUID, 4 MiB','',1))
if __name__=='__main__':unittest.main()
