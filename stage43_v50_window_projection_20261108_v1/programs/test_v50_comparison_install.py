"""Fresh-process A/Astar integration on the actual native ToT state machine."""
import json,subprocess,sys,unittest

SCRIPT=r'''
from pathlib import Path
from types import SimpleNamespace as NS
import ast,json,tempfile
import vllm.sequence
from jitserve.request_analyzer.graph_context import GraphMatchContext
from jitserve.request_info import RequestInfo,RequestType
from install_comparison_runtime import install
from prediction_transport import PROTOCOL
import inspect
method=METHOD
ctx=GraphMatchContext(generation_tokenizer=NS(encode=lambda *a,**k:[1]*10),use_all_node=False,stage_ratio_method='execution_time',graph_structure_type='tot',graph_matching_mode='online',use_total_deadline=False)
path=Path(inspect.getfile(GraphMatchContext)).parents[1]/'server.py'
tree=ast.parse(path.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['_handle_collective_request','_track_collection_completion']]
ns={'graph_ctx':ctx,'use_graph_matching':True,'Default_DeepResearch_Stage':6}
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+functions,type_ignores=[])
exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
server=NS(graph_ctx=ctx,**{n.name:ns[n.name]for n in functions})
with tempfile.TemporaryDirectory()as directory:
 receipt=install(server,directory,method=method,model_key='llama8',port=65433,predictor_protocol=PROTOCOL)
 assert not receipt['candidate_scheduler_installed']
 for sid,width in enumerate([3,1,3,1]):
  for member in range(width):
   info=RequestInfo(RequestType.COLLECTIVE,(2.,.1,20.),0,17,80,10,17,stage_id=sid,request_id=member)
   assert server._handle_collective_request(info,{'num_stages':4,'served_time':0.},'p')==(False,17)
   assert info.chain_stage_widths==[3,1,3,1]and info.chain_future_max_tokens==256
   if sid==0:assert info.deadline==(20. if method in ['Astar','v50']else 0.)
   server._track_collection_completion(17,False,10,17)
 assert ctx.collection_graph_unfinished_dict[17].is_finished
 assert ctx.collection_graph_unfinished_dict[17].stage_finished==4
 import torch
 assert not torch.cuda.is_initialized()
 print(json.dumps({'gate':'PASS','method':method,'native_implicit_ToT_complete':True,'shared_protocol':receipt['shared_prediction']['protocol']}))
'''


class Comparison(unittest.TestCase):
    def check(self,method):
        r=subprocess.run([sys.executable,'-c',SCRIPT.replace('METHOD',repr(method))],capture_output=True,text=True,timeout=60)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        d=json.loads(r.stdout.strip().splitlines()[-1]);self.assertEqual(d['gate'],'PASS')
    def test_native_A_shared_protocol_keeps_original_budget_and_implicit_ToT(self):self.check('A')
    def test_Astar_shared_protocol_keeps_declared_repairs_and_implicit_ToT(self):self.check('Astar')

    def test_v49pro_shared_entry_registers_native_ToT_with_real_keyword_API(self):
        script=SCRIPT.replace('method=METHOD',"method='v50'")
        script=script.replace(" assert not receipt['candidate_scheduler_installed']", " assert receipt['prediction_comparison_ready']and receipt['shared_prediction']['method']=='v50'\n from mixed_joint import MixedPolicy\n from vllm.core.scheduler import Scheduler\n import controlled_native as reference\n from collections import deque\n policy=MixedPolicy()")
        script=script.replace("   server._track_collection_completion(17,False,10,17)", "   group=reference.NativeRuntime().group(str(sid*3+member+1),info.chain_deadline)\n   group.request_info=info;group.collection_id=17\n   Scheduler.add_seq_group(NS(policy=policy,waiting=deque()),group)\n   assert policy.chains[17]['widths']==[3,1,3,1]and policy.chains[17]['future_max_tokens']==256\n   policy.delete_seq_group(group)\n   server._track_collection_completion(17,False,10,17)")
        r=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=90)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
