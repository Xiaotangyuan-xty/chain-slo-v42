import math
from types import SimpleNamespace as S
import unittest

class SafeStatisticsCPU(unittest.TestCase):
    def test_installer_uses_same_native_alias_and_restores_source_bytes(self):
        import hashlib,inspect,json,tempfile
        from pathlib import Path
        from jitserve.slo_tracker.slo_tracker import SLOTracker
        from vllm.slo_tracker.slo_tracker import SLOTracker as Alias
        from vllm.sequence import SequenceGroup
        from safe_statistics import install
        self.assertIs(SLOTracker,Alias)
        source=Path(inspect.getfile(SLOTracker));digest=hashlib.sha256(source.read_bytes()).hexdigest()
        old_tracker=SLOTracker._throughput_service_gain;old_service=SequenceGroup.service_compute
        try:
            with tempfile.TemporaryDirectory()as directory:
                p=Path(directory);(p/'PROTOCOL.json').write_text(json.dumps({'safe_statistics_source_sha256':digest}))
                receipt=install(None,p,p)
                self.assertFalse(receipt['original_source_bytes_changed']);self.assertTrue((p/'SAFE_STATISTICS_RUNTIME.json').exists())
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
        finally:
            SLOTracker._throughput_service_gain=old_tracker;SequenceGroup.service_compute=old_service
            if hasattr(SLOTracker,'_stage19_safe_installed'):delattr(SLOTracker,'_stage19_safe_installed')
    def test_numerical_equivalence_and_overflow_reproduction(self):
        from jitserve.slo_tracker.slo_tracker import SLOTracker
        from jitserve.request_info import RequestType
        from safe_statistics import safe_throughput_method
        original=SLOTracker._throughput_service_gain;fixed,receipt=safe_throughput_method(original)
        tracker=SLOTracker(100)
        for ratio in [0.,1e-6,.1,.5,.999,1.,1.001,2.,100.,500.]:
            g=S(concord_metrics=S(TTLT=1.),request_type=RequestType.THROUGHPUT,deadline=ratio)
            self.assertEqual(original(tracker,g,0,1,128,1),fixed(tracker,g,0,1,128,1))
        g=S(concord_metrics=S(TTLT=.001),request_type=RequestType.THROUGHPUT,deadline=80.)
        with self.assertRaises(OverflowError):original(tracker,g,0,1,128,1)
        self.assertEqual(fixed(tracker,g,0,1,128,1),136.)
        self.assertEqual(receipt['changed_expressions'],1)
    def test_last_token_interval_reproduction_and_cpu_fix(self):
        from jitserve.slo_tracker.slo_tracker import SLOTracker
        from jitserve.request_info import RequestSLOMetrics,RequestType
        from safe_statistics import finalize_metrics
        count={'n':0};metrics=RequestSLOMetrics(100.)
        seq=S(get_prompt_len=lambda:128,get_output_len=lambda:count['n'])
        g=S(first_seq=seq,concord_metrics=metrics,request_type=RequestType.COLLECTIVE,deadline=20.,TTFT_constraint=2.,TBT_constraint=.1,is_finished=lambda:count['n']==3)
        tracker=SLOTracker(100)
        tracker._update_metrics(g,100.)
        count['n']=1;tracker._update_metrics(g,100.1)
        count['n']=2;tracker._update_metrics(g,100.2)
        count['n']=3;metrics.TTLT=.3
        tracker._compute_service_gain(g,100.3,128,3,0,3)
        self.assertEqual(len(metrics.TBT),1)
        finalize_metrics(g,100.3,100,SLOTracker)
        self.assertEqual(len(metrics.TBT),2)
        self.assertAlmostEqual(sum(metrics.TBT),.2)
        finalize_metrics(g,100.3,100,SLOTracker)
        self.assertEqual(len(metrics.TBT),2)

if __name__=='__main__':unittest.main(verbosity=2)
