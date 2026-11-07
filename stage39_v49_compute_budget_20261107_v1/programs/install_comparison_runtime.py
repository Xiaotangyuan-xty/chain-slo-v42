"""One explicit shared-runtime entry for future fresh A/Astar/v49 processes."""
from pathlib import Path
import json


def install(server, output, *, method, model_key, port, predictor_protocol,
            timeout_seconds=10.):
    config={'port':port,'predictor_protocol':predictor_protocol,'timeout_seconds':timeout_seconds}
    if method=='v49':
        from install_v49 import install as candidate
        return candidate(server,output,model_key=model_key,shared_prediction_config=config)
    if method not in ['A','Astar']:
        raise ValueError('Explicit comparison method required')
    from vllm.sequence import SequenceGroup
    from jitserve.request_analyzer.graph_context import GraphMatchContext
    from jitserve.scheduler.policy import SLOPolicy
    from native_tot_adapter import install_server_annotations,install_astar_tot_budget
    budget=None
    if method=='Astar':
        from baseline_budget import install as dr_budget
        dr_budget(output);budget=install_astar_tot_budget(GraphMatchContext)
    annotations=install_server_annotations(server)
    from shared_prediction import install as shared
    receipt={'method':method,'shared_prediction':shared(server,SLOPolicy,output,method=method,**config),
             'native_ToT_annotations':annotations,'Astar_ToT_budget':budget,
             'original_A_budget_unchanged':method=='A','candidate_scheduler_installed':False}
    (Path(output)/'COMPARISON_RUNTIME.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
