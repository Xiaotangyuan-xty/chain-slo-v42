"""Explicit eight-method dispatch; frozen candidate bytes remain separate."""
import atexit,hashlib,importlib,importlib.util,json,sys
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_common(stage):
    common=Path(stage)/'common_runtime';out={}
    for name in ['prediction_transport','native_tot_adapter','baseline_budget','execution_observation','shared_prediction']:
        if name in sys.modules:raise RuntimeError('Fresh process required for common runtime')
        path=common/(name+'.py');spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);out[name]=module
    return out


def install(server,output,*,method,model_key,port,predictor_protocol,timeout_seconds=10.):
    output=Path(output);stage=output.parents[1];binding=json.loads((stage/'SUITE_SPEC.json').read_text())
    assert binding['method']==method and method in ['A','Astar','v47','v47pro','v48','v49','v49pro','v50']
    from budget_contract import install as bind_budget
    runtime_budget=bind_budget(output)
    common=load_common(stage)
    from jitserve.scheduler.policy import SLOPolicy
    from jitserve.request_analyzer.graph_context import GraphMatchContext
    import vllm.engine.llm_engine as engine
    config={'port':port,'predictor_protocol':predictor_protocol,'timeout_seconds':timeout_seconds}
    shared=None;candidate=None;budget=None;metadata_bridge=None;active={};expected={}
    if method in ['A','Astar']:
        if method=='Astar':
            common['baseline_budget'].install(output)
            tot_budget=common['native_tot_adapter'].install_astar_tot_budget(GraphMatchContext)
            (output/'ASTAR_TOT_BUDGET_ADAPTER.json').write_text(json.dumps(tot_budget,indent=2)+'\n')
            budget={'DR':json.loads((output/'BASELINE_BUDGET_ADAPTER.json').read_text()),'implicit_ToT':tot_budget}
        shared=common['shared_prediction'].install(server,SLOPolicy,output,method=method,**config)
        common['native_tot_adapter'].install_server_annotations(server)
    else:
        assert not any(n in sys.modules for n in ['native_chain','mixed_joint','chain_forecast'])
        source=stage/'candidate_source';sys.path.insert(0,str(source/'programs'))
        installer_name='install_v47max' if binding.get('release')=='v47max' else 'install_'+method
        installer=importlib.import_module(installer_name)
        if method=='v47':
            candidate=installer.install(server,output)
            shared=common['shared_prediction'].install(server,SLOPolicy,output,method=method,**config)
            common['native_tot_adapter'].install_server_annotations(server)
        else:
            candidate=installer.install(server,output,model_key=model_key,shared_prediction_config=config)
            shared=candidate['shared_prediction']
            from frozen_v47.native_chain import ChainCompletionPolicy as FrozenPolicy
            metadata_bridge=common['native_tot_adapter'].install_v47_metadata_bridge(FrozenPolicy)
        names=['native_chain','mixed_joint','chain_forecast','chain_admission','token_reward','install_'+method]
        if binding.get('release') in ['v47pro3','v47max']:
            names+=['pure_core_v47','frozen_v47.chain_admission','frozen_v47.chain_forecast','frozen_v47.native_chain','frozen_v47.token_reward']
        if binding.get('release')=='v47max':names+=['mixed_max','install_v47max']
        for name in names:
            module=importlib.import_module(name);path=Path(module.__file__).resolve();rel=str(path.relative_to(source))
            assert sha(path)==binding['candidate_hashes'][rel]
            active[name]={'path':str(path),'sha256':sha(path)};expected[name]=rel
    if not getattr(engine.LLMEngine,'_v49pro_execution_observation',False):
        observation=common['execution_observation'].install(output)
    else:observation=candidate['execution_observer']
    for name,module in common.items():active[name]={'path':module.__file__,'sha256':sha(module.__file__)}
    receipt={'runtime_budget_contract':runtime_budget,'method':method,'model_key':model_key,'shared_prediction':shared,'candidate_receipt':candidate,'stage_budget_adapter':budget,'published_pure_ToT_metadata_bridge':metadata_bridge,
             'candidate_scheduler_installed':method not in ['A','Astar'],'execution_observer':observation,
             'common_hashes':{str(Path(row['path']).relative_to(stage)):row['sha256'] for name,row in active.items() if name in common},
             'active_modules':active,'frozen_candidate_modules_exact':all(active[n]['sha256']==binding['candidate_hashes'][rel] for n,rel in expected.items()),
             'shared_adapter_scope':'Identical shared framed prediction, public native ToT metadata and observation; A native budget retained; Astar installs both DR and implicit-ToT budget corrections; frozen candidate scoring unchanged',
             'installer_runs_model':False,'installer_allocates_GPU':False,'Astar_implicit_ToT_budget_installed':method=='Astar','release':binding['release'],'release_commit':binding['candidate_commit']}
    def write():
        outer=getattr(server,'engine',None);inner=getattr(outer,'engine',outer)
        schedulers=getattr(inner,'scheduler',[]) if inner is not None else []
        rows=[]
        for s in schedulers:
            model=getattr(s,'cost_model',None);supported=hasattr(model,'observe_execution');stats=getattr(model,'stats',{})
            rows.append({'type':type(s).__module__+'.'+type(s).__name__,'mixed_counts':dict(getattr(s,'mixed_counts',{})),
                         'v47_counts':dict(getattr(s,'v47_counts',{})),'cost_callback_supported':supported,
                         'cost_samples':stats.get('samples',stats.get('accepted_samples',0)) if supported else None})
        if output.exists():(output/'COMPARISON_SUITE_RUNTIME.json').write_text(json.dumps({**receipt,'schedulers':rows},indent=2)+'\n')
    write();atexit.register(write)
    return receipt
