"""Independently check final client timing against per-token append history."""
import json
import math
from pathlib import Path

def key(row):return(row['phase'],row['collection_id'],row['stage_id'],row['request_id'])

def validate(attempt,flat,engine):
    attempt=Path(attempt);records={};bad=[]
    with(attempt/'LIFETIME_TIMINGS.jsonl').open()as stream:
        for line in stream:
            row=json.loads(line);k=key(row)
            if k in records:bad.append(['duplicate',k])
            records[k]=row
    for k,row in records.items():
        e=engine.get(k);client=flat.get(k)
        if e is None or client is None:bad.append(['identity',k]);continue
        times=row['token_append_monotonic'];intervals=[b-a for a,b in zip(times,times[1:])]
        if len(times)!=e['output_tokens']or row['output_tokens']!=e['output_tokens']:bad.append(['token_count',k])
        if row['TBT']!=intervals or client['server_tbt']!=intervals:bad.append(['intervals',k])
        ttft=row['first_token_wall']-row['arrival_wall']
        if row['TTFT']!=ttft or client['server_ttft']!=ttft:bad.append(['ttft',k])
        if not all(math.isfinite(x)for x in times)or any(not math.isfinite(x)or x<0 for x in intervals)or not math.isfinite(ttft)or ttft<0:bad.append(['clock',k])
        if times and(times[0]<e['start_monotonic']or times[-1]>e['finish_monotonic']):bad.append(['engine_bounds',k])
        if row['request_uuid']!=e['request_uuid']:bad.append(['uuid',k])
    expected=set(flat)==set(engine)==set(records)
    runtime=json.loads((attempt/'LIFETIME_OBSERVER_RUNTIME.json').read_text())
    checks={'lifetime_timing_exact_request_set':expected,'lifetime_raw_timestamps_match_client':not bad,
            'lifetime_observer_declared':runtime['version']=='lifetime_token_append_v1'and runtime['native_metrics_reset_or_policy_logic_changed']is False}
    receipt={'checks':checks,'bad':bad,'requests':len(records),
             'requests_with_recompute':sum(bool(r['recompute_prefix_counts'])for r in records.values()),
             'intervals':sum(len(r['TBT'])for r in records.values()),'scope':'Actual per-token append timestamps; no filling, interpolation or native-metric reset suppression.'}
    (attempt/'LIFETIME_TIMING_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return checks
