"""Separately qualified native implicit-ToT adapter; frozen policy files untouched.

The wire never carries requests_per_stage for ToT. Native ToTStructure infers
progress from completed calls. Only public known blueprint/caps are annotated;
no recorded candidate text or true future output length reaches the policy.
"""
import math
from functools import wraps

WIDTHS = (3, 1, 3, 1)
TOT_MAX_TOKENS = 256
VERSION = 'native_implicit_tot_structural_bridge_v1'


def install_server_annotations(server):
    previous = server._handle_collective_request
    @wraps(previous)
    def handle(info, request_dict, prompt):
        implicit = 'requests_per_stage' not in request_dict
        stages = request_dict.get('num_stages', 4)
        if implicit and stages != len(WIDTHS):
            raise ValueError('Only separately qualified native ToT3-thought/2-round graph supported')
        result = previous(info, request_dict, prompt)
        if implicit:
            if result[0] is not False:
                raise ValueError('Native ToT routed into DR by mistake')
            info.application = 'tot_native'
            info.chain_stage_count = len(WIDTHS)
            info.chain_stage_widths = list(WIDTHS)
            info.chain_future_max_tokens = TOT_MAX_TOKENS
        return result
    server._handle_collective_request = handle
    return {'version': VERSION, 'native_is_deepresearch': False, 'wire_requests_per_stage': False,
            'known_stage_widths': list(WIDTHS), 'known_future_sampling_cap': TOT_MAX_TOKENS,
            'annotation_order': 'after prior collective handler, before SequenceGroup construction',
            'scoring_formula_changed': False, 'future_output_truth_used': False}


def install_v47_metadata_bridge(policy_class):
    previous = policy_class.add_seq_group
    @wraps(previous)
    def add(self, group):
        result = previous(self, group)
        info = group.request_info
        if getattr(info, 'application', None) == 'tot_native':
            meta = self.chains[group.collection_id]
            if tuple(meta['widths']) != WIDTHS or meta['stages'] != len(WIDTHS):
                raise ValueError('Incorrect native ToT width/depth binding')
            meta['future_max_tokens'] = info.chain_future_max_tokens
        return result
    policy_class.add_seq_group = add
    return {'version': VERSION, 'effect': 'propagate known native ToT widths/cap to existing forecaster metadata',
            'future_token_reward_formula_changed': False, 'DR_path_unchanged': True}


def tot_cumulative_ratio(ctx, num_stages, prompt, collection_id, stage_id):
    structure = ctx.collection_graph_unfinished_dict[collection_id]
    actual_stage = structure.stage_finished
    if num_stages != len(WIDTHS) or tuple(structure.request_per_stage) != WIDTHS:
        raise ValueError('Unqualified native ToT structure')
    if stage_id != actual_stage or not 0 <= actual_stage < num_stages:
        raise ValueError('Explicit identity disagrees with native implicit progress')
    fallback = (actual_stage + 1) / num_stages
    unfinished = structure.convert_to_unfinished_graph(ctx._encode_prompt_len(prompt), None)
    match = ctx.dynamic_clustering.find_best_match(unfinished)
    if match and match.times and len(match.times) == num_stages and all(math.isfinite(x) and x >= 0 for x in match.times) and sum(match.times) > 0:
        return max(fallback, sum(match.times[:actual_stage + 1]) / sum(match.times))
    return fallback


def install_astar_tot_budget(graph_context_class):
    """Install after the unchanged DR Astar adapter. Native A never calls this."""
    previous = graph_context_class._online_ratio
    @wraps(previous)
    def corrected(self, num_stages, requests_per_stage, prompt, collection_id, is_deepresearch, stage_id):
        if is_deepresearch:
            return previous(self, num_stages, requests_per_stage, prompt, collection_id, is_deepresearch, stage_id)
        if requests_per_stage is not None:
            raise ValueError('Native implicit ToT must omit explicit DR widths')
        return tot_cumulative_ratio(self, num_stages, prompt, collection_id, stage_id)
    graph_context_class._online_ratio = corrected
    return {'version': 'Astar_native_tot_current_stage_budget_v1',
            'semantics': 'generalize existing current-stage-inclusive/declared-depth Astar repair to native implicit ToT',
            'source_native_A_changed': False, 'DR_existing_Astar_adapter_unchanged': True,
            'future_eval_lengths_used': False, 'predictor_retrained': False}
