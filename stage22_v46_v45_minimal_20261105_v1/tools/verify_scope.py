"""Check the requested two-change scope, without importing Torch or CUDA."""
import ast
import hashlib
import json
from pathlib import Path

STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE.parent
V45=ROOT/'stage15_stable_chain_v45_parallel_20261101_v2/round01'


def verify():
    checks={}
    for name in ['programs/native_chain.py','programs/chain_admission.py',
                 'programs/controlled_native.py','assets/CHAIN_MODEL.json']:
        checks['v45_bytes_unchanged:'+name]=(STAGE/name).read_bytes()==(V45/name).read_bytes()
    a=ast.parse((V45/'programs/chain_forecast.py').read_text()).body
    b=ast.parse((STAGE/'programs/chain_forecast.py').read_text()).body
    old=next(n for n in a if isinstance(n,ast.ClassDef))
    new=next(n for n in b if isinstance(n,ast.ClassDef))
    funcs=lambda cls:{n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    old,new=funcs(old),funcs(new)
    checks['forecast_only_neighbors_and_estimate_changed']=set(old)==set(new) and all(
        ast.dump(old[name])==ast.dump(new[name])for name in old if name not in {'neighbors','estimate'})
    imports=set()
    attributes=set()
    for name in ['native_chain.py','chain_forecast.py','chain_admission.py',
                 'mixed_joint.py','joint_admission.py','install_v46.py']:
        tree=ast.parse((STAGE/'programs'/name).read_text())
        imports.update(n.module for n in ast.walk(tree)if isinstance(n,ast.ImportFrom))
        attributes.update(n.attr for n in ast.walk(tree)if isinstance(n,ast.Attribute))
    checks['no_superseded_cadence_or_budget_or_scheduler_imports']=not imports & {
        'scheduler_v46','forecast_v46','admission_v46','observation_v46','baseline_budget'}
    checks['no_future_evaluation_output_attributes']=not attributes & {'target_output_length','real_output_len'}
    checks['pure_chain_delegates_to_frozen_v45']='return NativeChainScheduler._schedule(self)' in (STAGE/'programs/mixed_joint.py').read_text()
    checks['no_event_key_or_wall_lease_added']=not any(token in (STAGE/'programs/mixed_joint.py').read_text()
        for token in ['last_plan_key','wall_budget','prefill_paid','ExecutionCadence'])
    return {'gate':'PASS'if all(checks.values())else'FAIL','checks':checks,'GPU_used':False}


if __name__=='__main__':
    result=verify();print(json.dumps(result,indent=2));raise SystemExit(result['gate']!='PASS')
