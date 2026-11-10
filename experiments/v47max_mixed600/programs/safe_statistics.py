"""Candidate common observation repair. No source file or scheduling policy edits."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

def safe_throughput_method(original):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)));changes=[]
    class Fix(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name)and node.func.id=='min'and len(node.args)==2:
                power=node.args[1]
                if isinstance(power,ast.BinOp)and isinstance(power.op,ast.Pow)and isinstance(power.left,ast.BinOp)and isinstance(power.left.op,ast.Div):
                    changes.append(ast.dump(node))
                    test=ast.BoolOp(op=ast.And(),values=[ast.Compare(left=power.left,ops=[ast.GtE()],comparators=[ast.Constant(1.)]),
                         ast.Compare(left=power.right,ops=[ast.GtE()],comparators=[ast.Constant(0)])])
                    return ast.copy_location(ast.IfExp(test=test,body=ast.Constant(1.),orelse=node),node)
            return node
    tree=Fix().visit(tree);assert len(changes)==1,changes
    ns=dict(original.__globals__);exec(compile(ast.fix_missing_locations(tree),original.__code__.co_filename,'exec'),ns)
    return ns[original.__name__],{'changed_expressions':len(changes),'rule':'For ratio>=1 and nonnegative penalty, return saturation1 before exponentiation; otherwise evaluate original expression.','mathematically_equivalent_on_original_defined_domain':True}

def finalize_metrics(group,cur_time,penalty_factor,tracker_cls):
    if group.is_finished():tracker_cls(penalty_factor)._update_metrics(group,cur_time)

def install(root,stage,output):
    from jitserve.slo_tracker.slo_tracker import SLOTracker
    from vllm.slo_tracker.slo_tracker import SLOTracker as Alias
    from vllm.sequence import SequenceGroup
    assert SLOTracker is Alias
    assert not getattr(SLOTracker,'_stage19_safe_installed',False)
    original=SLOTracker._throughput_service_gain
    fn,receipt=safe_throughput_method(original);source=Path(inspect.getfile(SLOTracker))
    expected=json.loads((Path(stage)/'PROTOCOL.json').read_text())['safe_statistics_source_sha256']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    SLOTracker._throughput_service_gain=fn;SLOTracker._stage19_safe_installed=True
    old=SequenceGroup.service_compute
    def service_compute(self,cur_time,penalty_factor):
        finalize_metrics(self,cur_time,penalty_factor,SLOTracker)
        return old(self,cur_time,penalty_factor)
    SequenceGroup.service_compute=service_compute
    receipt.update(schema='stage19-common-safe-observation/v2',source=str(source),source_sha256=expected,
                   original_source_bytes_changed=False,scheduling_policy_changed=False,
                   final_token_timing='Update native metrics once at finished output before serializing TBT, including final token delta; no update when request is not finished.')
    (Path(output)/'SAFE_STATISTICS_RUNTIME.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
