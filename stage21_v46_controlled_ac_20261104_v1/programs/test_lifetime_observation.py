"""Reproduce native reset loss and verify independent timing on native objects."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

class LifetimeCPU(unittest.TestCase):
    def setUp(self):
        import lifetime_observation as module
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
        (self.path/'PROTOCOL.json').write_text(json.dumps({'lifetime_observation_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}))
        self.observer=module.install(None,self.path,self.path)
    def tearDown(self):
        self.observer.restore_for_cpu_test();self.tmp.cleanup()

    def group(self):
        from jitserve.request_info import RequestInfo,RequestType
        from vllm.sequence import Sequence,SequenceGroup
        from vllm import SamplingParams
        info=RequestInfo(RequestType.COLLECTIVE,(2.,.1,20.),0,77,120,32,64)
        info.chain_deadline=2000.;info.chain_stage_count=1;info.chain_stage_widths=[1]
        seq=Sequence(77,inputs={'prompt_token_ids':list(range(32)),'prompt':'CPU'},block_size=16)
        with patch('time.time',return_value=100.):
            group=SequenceGroup('cpu77',[seq],100.,info,SamplingParams(temperature=0,max_tokens=64))
        return group
    def append(self,seq,stamp):
        from vllm.sequence import Logprob
        with patch('lifetime_observation._wall',return_value=stamp),patch('lifetime_observation._clock',return_value=stamp):
            seq.append_token_id(1,{1:Logprob(0.)})

    def test_native_reset_drops_history_lifetime_capture_retains_gap(self):
        from jitserve.slo_tracker.slo_tracker import SLOTracker
        from vllm.core.scheduler import Scheduler
        from vllm.sequence import SequenceStatus
        group=self.group();seq=group.first_seq;tracker=SLOTracker(100)
        self.append(seq,100.1);tracker._update_metrics(group,100.1)
        self.append(seq,100.2);tracker._update_metrics(group,100.2)
        self.assertEqual(len(group.concord_metrics.TBT),1)
        native_reset=Scheduler._preempt_by_recompute
        stub=S(free_seq=lambda _:None)
        native_reset(stub,group)
        self.assertEqual(group.concord_metrics.TBT,[])
        self.assertEqual(seq.get_output_len(),2)
        self.append(seq,110.3);tracker._update_metrics(group,110.3)
        self.append(seq,110.4);tracker._update_metrics(group,110.4)
        self.assertLess(len(group.concord_metrics.TBT),3)
        seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED
        original_native=list(group.concord_metrics.TBT);original_ttft=group.concord_metrics.TTFT
        result=S(finished=True,outputs=[S(token_ids=list(seq.get_output_token_ids()))],ttft=original_ttft,tbt=original_native)
        self.observer.exported(group,result)
        self.assertEqual(len(result.tbt),3)
        for actual,expected in zip(result.tbt,[.1,10.1,.1]):self.assertAlmostEqual(actual,expected)
        self.assertAlmostEqual(result.ttft,.1)
        self.assertEqual(group.concord_metrics.TBT,original_native)
        self.assertEqual(group.concord_metrics.TTFT,original_ttft)
        self.observer.exported(group,result)
        rows=[json.loads(x)for x in(self.path/'LIFETIME_TIMINGS.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['recompute_prefix_counts'],[2])

    def test_real_RequestOutput_classmethod_final_export_and_single_token(self):
        from vllm.outputs import RequestOutput
        from vllm.sequence import SequenceStatus
        group=self.group();seq=group.first_seq;self.append(seq,100.25)
        seq.output_text='x';seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED
        group.concord_metrics.TBT=[]
        with patch('time.time',return_value=100.3):
            result=RequestOutput.from_seq_group(group,use_cache=False,seq_id_to_seq_group={})
        self.assertTrue(result.finished);self.assertEqual(result.tbt,[]);self.assertAlmostEqual(result.ttft,.25)

    def test_unobserved_prefix_is_rejected_not_filled(self):
        group=self.group();seq=group.first_seq
        result=S(finished=True,outputs=[S(token_ids=[1,2])],ttft=None,tbt=[])
        with self.assertRaisesRegex(RuntimeError,'counts differ'):self.observer.exported(group,result)

    def test_actual_native_block_manager_recompute_path_preserves_prefix(self):
        from controlled_native import NativeRuntime
        fixture=NativeRuntime();fixture.exercise_preemption(1)
        self.assertGreater(self.observer.counts['recompute_resets'],0)

    def test_scheduler_decisions_and_KV_are_identical_with_observer(self):
        from tests import MixedCPU
        fixture=MixedCPU()
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            observed,_=fixture.simulate('native',[2,0],cpu=1)
        self.observer.restore_for_cpu_test()
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            baseline,_=fixture.simulate('native',[2,0],cpu=1)
        self.assertEqual(observed,baseline)
        # Reinstall for tearDown to restore the same single installation.
        import lifetime_observation as module
        other=self.path/'again';other.mkdir();(other/'PROTOCOL.json').write_text((self.path/'PROTOCOL.json').read_text())
        self.observer=module.install(None,other,other)

if __name__=='__main__':unittest.main(verbosity=2)
