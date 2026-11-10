import json
from pathlib import Path
from types import SimpleNamespace as S
import subprocess
import sys
import tempfile
import unittest
from prediction_transport import PROTOCOL


class InstallTests(unittest.TestCase):
    def test_actual_native_install_annotations_observer_and_fresh_guard(self):
        from install_v47pro import install
        import vllm.engine.llm_engine as engine
        from mixed_joint import MixedScheduler
        with tempfile.TemporaryDirectory() as directory:
            server=S(_handle_collective_request=lambda info,data,prompt:(False,info.collection_id))
            receipt=install(server,Path(directory),shared_prediction_config={'port':65040,'predictor_protocol':PROTOCOL})
            self.assertIs(engine.SLOScheduler,MixedScheduler)
            self.assertFalse(receipt['installer_runs_model']);self.assertFalse(receipt['installer_allocates_GPU'])
            self.assertTrue(receipt['prediction_comparison_ready']);self.assertFalse(receipt['stage_budget_adapter'])
            info=S(collection_id=1,slo_constraint=(2.,.1,20.))
            server._handle_collective_request(info,{'num_stages':4},'x')
            self.assertEqual(info.chain_stage_widths,[3,1,3,1]);self.assertEqual(info.chain_future_max_tokens,256)
            self.assertTrue((Path(directory)/'V47PRO_RUNTIME.json').is_file())
            with self.assertRaises((RuntimeError,ValueError)):install(server,Path(directory))

    def test_decision_writer_real_file_drain_and_idempotent_close(self):
        from install_v47pro import _DecisionWriter
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'decisions.jsonl';stream=path.open('x',buffering=1024)
            writer=_DecisionWriter(stream,queue_size=128,batch_size=8,flush_interval=.01)
            for i in range(100):writer.enqueue(2,{'kind':'plan','decision_id':i,'payload':{'tick':i}})
            writer.close();writer.close()
            rows=[json.loads(line)for line in path.read_text().splitlines()]
            self.assertEqual([row['decision_id']for row in rows],list(range(100)))
            self.assertEqual(rows[-1],{'scheduler':2,'kind':'plan','decision_id':99,'payload':{'tick':99}})
            self.assertEqual(writer.snapshot(),{'queued':100,'written':100,'dropped':0,'writer_errors':0})

    def test_atexit_drains_actual_sink_and_records_runtime_counts(self):
        script='''from collections import Counter
from types import ModuleType,SimpleNamespace as S
import sys
def module(name):
    value=ModuleType(name);sys.modules[name]=value;return value
native_chain=module('native_chain');native_chain.MODEL=S(stats={});native_chain.install=lambda server:None
mixed=module('mixed_joint');mixed.VERSION='test';mixed.VARIANT='test'
class MixedScheduler:
    def __init__(self):
        self.mode='pure';self.mode_epoch=0;self.v47_counts=Counter();self.mixed_counts=Counter()
        self.events=S(counts=Counter());self.cost_model=S(describe=lambda:{});self.last_joint_summary=None
mixed.MixedScheduler=MixedScheduler
pure_core=module('pure_core_v47');pure_core.MODEL=S(stats={})
annotations=module('native_tot_adapter');annotations.install_server_annotations=lambda server:{}
token_reward=module('token_reward');token_reward.REWARD=S(stats={})
vllm=module('vllm');vllm.__path__=[];engine_package=module('vllm.engine');engine_package.__path__=[]
engine=module('vllm.engine.llm_engine');vllm.engine=engine_package;engine_package.llm_engine=engine
observation=module('execution_observation');observation.install=lambda output:{'test':True}
from install_v47pro import install
server=S(_handle_collective_request=lambda *args:None)
install(server,sys.argv[1])
scheduler=MixedScheduler()
for i in range(2000):scheduler.decision_sink({'kind':'plan','decision_id':i,'tick':i})
'''
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,'-B','-c',script,directory],capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            rows=[json.loads(line)for line in(Path(directory)/'V47PRO_DECISIONS.jsonl').read_text().splitlines()]
            self.assertEqual([row['decision_id']for row in rows],list(range(2000)))
            self.assertTrue(all(row['scheduler']==0 and row['kind']=='plan' and row['tick']==i for i,row in enumerate(rows)))
            runtime=json.loads((Path(directory)/'V47PRO_RUNTIME.json').read_text())
            self.assertEqual(runtime['decision_log'],{'queued':2000,'written':2000,'dropped':0,'writer_errors':0})

    def test_full_queue_drops_without_blocking(self):
        from install_v47pro import _DecisionWriter
        import threading
        class BlockingStream:
            def __init__(self):self.entered=threading.Event();self.release=threading.Event();self.parts=[]
            def write(self,text):
                self.entered.set()
                if not self.release.wait(5.):raise OSError('fixture timeout')
                self.parts.append(text)
            def flush(self):pass
            def close(self):pass
        stream=BlockingStream();writer=_DecisionWriter(stream,queue_size=1,batch_size=1,flush_interval=60.)
        try:
            writer.enqueue(0,{'decision_id':0});self.assertTrue(stream.entered.wait(5.))
            writer.enqueue(0,{'decision_id':1});writer.enqueue(0,{'decision_id':2})
            self.assertEqual(writer.snapshot()['dropped'],1)
        finally:
            stream.release.set();writer.close()
        self.assertEqual(writer.snapshot(),{'queued':2,'written':2,'dropped':1,'writer_errors':0})
        rows=[json.loads(line)for line in ''.join(stream.parts).splitlines()]
        self.assertEqual([row['decision_id']for row in rows],[0,1])

    def test_writer_errors_are_isolated_and_counted(self):
        from install_v47pro import _DecisionWriter
        class BrokenStream:
            def write(self,text):raise OSError('fixture')
            def flush(self):pass
            def close(self):pass
        writer=_DecisionWriter(BrokenStream(),queue_size=8,batch_size=16,flush_interval=60.)
        for i in range(4):writer.enqueue(0,{'decision_id':i})
        writer.close()
        self.assertEqual(writer.snapshot(),{'queued':4,'written':0,'dropped':0,'writer_errors':1})


if __name__=='__main__':unittest.main(verbosity=2)
