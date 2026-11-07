"""Build replay-length ToT donors from the reserved Stage34 calibration split.

No GPU labels, actual evaluation-run outcomes, prompt text or eval templates enter
the runtime asset. This calibrates fixed service-replay lengths, not app quality.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def build(pool, reference):
    selected=[t for t in pool['templates']if t['application']=='tot_style'and t['split']=='calibration']
    evaluation={t['input_graph_fingerprint']for t in pool['templates']if t['split']=='evaluation'}
    by_graph={}
    for t in selected:by_graph.setdefault(t['input_graph_fingerprint'],t)
    if not by_graph or set(by_graph)&evaluation:
        raise ValueError('Calibration/evaluation input-graph separation failed')
    records=[]
    for fp,t in sorted(by_graph.items()):
        if [len(s)for s in t['stages']]!=[3,1,3,1]:raise ValueError('Unqualified ToT blueprint')
        stages=[]
        for stage in t['stages']:
            if any(not 1<=r['target_output_tokens']<=256 for r in stage):raise ValueError('Invalid public replay cap')
            stages.append({'inputs':[r['declared_input_tokens']for r in stage],
                           'outputs':[r['target_output_tokens']for r in stage]})
        records.append({'id':fp,'stages':stages})
    model={k:v for k,v in reference.items()if k not in ['records','fit_ids','cal_ids','training_trace_sha256']}
    model.update(records=records,fit_ids=sorted(by_graph),cal_ids=[],fit_graph_count=len(records),
                 cal_graph_count=0,training_trace_sha256=digest(records),
                 scope='Reserved public ToT-style fixed replay length profiles only; no evaluation run labels or natural-generation quality claim.')
    return {'model':model,'source_pool_sha256':pool['pool_sha256'],
            'selected_template_ids':sorted(t['template_id']for t in selected),
            'calibration_graph_fingerprints':sorted(by_graph),
            'evaluation_graphs_disjoint':True,'public_sampling_cap':256,
            'performance_or_coverage_validated':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pool-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    reference=json.loads((Path(__file__).resolve().parents[1]/'assets/CHAIN_MODEL.json').read_text())
    models={k:build(json.loads((args.pool_dir/(k+'.json')).read_text()),reference)for k in ['llama8','qwen14']}
    assert models['llama8']['calibration_graph_fingerprints']==models['qwen14']['calibration_graph_fingerprints']
    args.output.write_text(json.dumps({'schema':'reserved_tot_replay_length_donors/v1','source':'Stage34 prefrozen pool, public Game24 log prompts/recorded/reconstructed response lengths','CPU_only':True,'GPU_labels_used':False,'models':models},indent=2)+'\n')
