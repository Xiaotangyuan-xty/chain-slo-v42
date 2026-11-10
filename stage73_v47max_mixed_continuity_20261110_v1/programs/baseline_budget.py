"""Explicit A* semantic adapter; original A remains byte-for-byte native."""
import atexit,json,math
from pathlib import Path

def cumulative_ratio(ctx,num_stages,requests_per_stage,prompt,collection_id,stage_id):
    if not 0<=stage_id<num_stages:raise ValueError('Invalid stage index')
    fallback=(stage_id+1)/num_stages
    structure=ctx.collection_deepresearch_unfinished_dict[collection_id]
    unfinished=structure.convert_to_unfinished_graph(ctx._encode_prompt_len(prompt),None)
    match=ctx.dynamic_clustering.find_best_match(unfinished)
    if match and match.times and len(match.times)==num_stages and all(math.isfinite(x)and x>=0 for x in match.times)and sum(match.times)>0:
        return max(fallback,sum(match.times[:stage_id+1])/sum(match.times))
    return fallback

def install(output):
    from jitserve.request_analyzer.graph_context import GraphMatchContext
    old=GraphMatchContext._online_ratio
    def corrected(self,num_stages,requests_per_stage,prompt,collection_id,is_deepresearch,stage_id):
        if not is_deepresearch:return old(self,num_stages,requests_per_stage,prompt,collection_id,is_deepresearch,stage_id)
        return cumulative_ratio(self,num_stages,requests_per_stage,prompt,collection_id,stage_id)
    GraphMatchContext._online_ratio=corrected
    (Path(output)/'BASELINE_BUDGET_ADAPTER.json').write_text(json.dumps({'variant':'cumulative_through_current_declared_depth_v1',
       'original_native_source_changed':False,'current_stage_included':True,'default_denominator':'declared number of stages',
       'historical_match':'Only equal-depth valid historical stage times; retain max(historical,uniform).','future_evaluation_lengths_accessed':False},indent=2)+'\n')
