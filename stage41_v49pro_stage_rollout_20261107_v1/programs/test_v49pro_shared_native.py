import tempfile,unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from vllm.sequence import SequenceStatus
from jitserve.scheduler.policy import SLOPolicy
from shared_prediction import install
from prediction_transport import PROTOCOL
import controlled_native as reference


class SharedNative(unittest.TestCase):
    def test_actual_collective_allocation_and_cached_priority_follow_new_prediction(self):
        with tempfile.TemporaryDirectory()as directory,patch('time.time',return_value=100.):
            server=NS();receipt=install(server,SLOPolicy,directory,method='v49pro',port=65433,predictor_protocol=PROTOCOL)
            fixture=reference.NativeRuntime();a=fixture.group('1',2000.);b=fixture.group('2',2000.)
            b.collection_id=a.collection_id;b.request_info.collection_id=a.collection_id
            a.predict_output_length=1024;b.predict_output_length=1024
            a.request_info.output_len=32;b.request_info.output_len=48
            policy=SLOPolicy(schedule_interval=50,penalty_factor=100)
            policy.add_seq_group(a);policy.add_seq_group(b)
            self.assertAlmostEqual(policy.allocation_control(a),policy._get_interval_time()*80)
            first=policy.get_priority(a)
            b.request_info.output_len=128;b.request_info.prediction_revision=1
            self.assertAlmostEqual(policy.allocation_control(a),policy._get_interval_time()*160)
            second=policy.get_priority(a);self.assertNotEqual(first,second)
            self.assertEqual(policy.get_priority(b),second)
            policy.delete_seq_group(a);policy.delete_seq_group(b)
            self.assertFalse(policy.seq_group_dict)
            self.assertFalse(policy._prediction_revisions)


if __name__=='__main__':unittest.main(verbosity=2)
