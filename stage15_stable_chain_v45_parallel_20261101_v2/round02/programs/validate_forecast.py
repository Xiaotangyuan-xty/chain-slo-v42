"""Verify FIT/CAL provenance and summarize held-out training-graph forecast errors."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from chain_forecast import StageForecaster
root,stage=map(Path,sys.argv[1:3]);model=json.loads((stage/'assets/CHAIN_MODEL.json').read_text());source=root/'repo/traces/deepresearch_training.json';graphs=json.loads(source.read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==model['training_trace_sha256']
ids=sorted([g['request_id'] for g in graphs],key=lambda k:hashlib.sha256(f'repair-v1/{k}'.encode()).digest())
assert set(model['fit_ids'])==set(ids[:295]);assert set(model['cal_ids'])==set(ids[295:])
fit={g['request_id'] for g in graphs if g['request_id'] in model['fit_ids']};assert len(fit)==295
f=StageForecaster(model);errors={};count=0
for g in graphs:
 if g['request_id'] not in model['cal_ids']:continue
 widths=[len(s['requests']) for s in g['stages']];meta={'stages':len(widths),'widths':widths,'observed':{},'neighbor_cache':{}}
 for sid,s in enumerate(g['stages']):
  views=[{'member':i,'declared_input':r['input_tokens'],'generated':0,'prefill_tokens':r['input_tokens'],'max_tokens':1024} for i,r in enumerate(s['requests'])]
  e=f.estimate(meta,sid,views,1000.,.033);truth=max(min(1024,max(1,r['output_tokens'])) for r in s['requests'])
  assert all(math.isfinite(v) for v in e.values() if isinstance(v,(int,float)));assert 0<=e['probability']<=1
  errors.setdefault(f'{len(widths)}stages_stage{sid}',[]).append(abs(e['current_remaining_tokens']-truth));count+=1
  meta['observed'][sid]={i:{'declared_input':r['input_tokens'],'output':min(1024,max(1,r['output_tokens']))} for i,r in enumerate(s['requests'])}
# Signature-overlap check is audit-only; evaluation records never enter the model.
def signature(stages):return json.dumps([sorted((r['input_tokens'],min(1024,max(1,r['output_tokens']))) for r in s) for s in stages])
training={signature([s['requests'] for s in g['stages']]) for g in graphs}
test=[json.loads(l) for l in (root/'repo/traces/deepresearch_filter.jsonl').open() if l.strip()]
overlap=sum(signature(g['stages']) in training for g in test)
result={'gate':'PASS' if overlap==0 else 'FAIL','fit_graphs':295,'cal_graphs':99,'cal_stage_states':count,'exact_full_graph_signature_overlap_with_evaluation':overlap,'model_sha256':hashlib.sha256((stage/'assets/CHAIN_MODEL.json').read_bytes()).hexdigest(),'current_stage_max_output_MAE_tokens':{k:statistics.fmean(v) for k,v in errors.items()},'scope':'FIT-only empirical heuristic; CAL reports errors without tuning or coverage claims. Runtime has no evaluation trace reader.'}
with (stage/'receipts/FORECAST_VALIDATION.json').open('x') as h:json.dump(result,h,indent=2)
print(json.dumps(result,indent=2));sys.exit(0 if result['gate']=='PASS' else 1)
