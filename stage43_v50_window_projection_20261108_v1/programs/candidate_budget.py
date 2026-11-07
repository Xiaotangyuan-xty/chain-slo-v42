"""Candidate's explicit corrected stage budgets, shared semantics with Astar.

No source baseline is modified and no future evaluation lengths are read. This
is a declared foundation of v50, not claimed as a new contribution.
"""
import json
from pathlib import Path
from baseline_budget import cumulative_ratio
from native_tot_adapter import tot_cumulative_ratio


def install(output):
    from jitserve.request_analyzer.graph_context import GraphMatchContext
    if getattr(GraphMatchContext,'_v50_budget_installed',False):raise RuntimeError('Fresh budget context required')
    GraphMatchContext._v50_budget_installed=True
    def corrected(self,num_stages,requests_per_stage,prompt,collection_id,is_deepresearch,stage_id):
        if is_deepresearch:
            return cumulative_ratio(self,num_stages,requests_per_stage,prompt,collection_id,stage_id)
        if requests_per_stage is not None:raise ValueError('Native ToT must use implicit graph path')
        return tot_cumulative_ratio(self,num_stages,prompt,collection_id,stage_id)
    GraphMatchContext._online_ratio=corrected
    receipt={'variant':'v50_corrected_stage_budget_reference_v1','semantics_equal_Astar':True,
             'current_stage_included':True,'future_evaluation_lengths_used':False,
             'native_baseline_A_changed':False,'not_claimed_as_novel_mechanism':True}
    (Path(output)/'V50_BUDGET_ADAPTER.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
