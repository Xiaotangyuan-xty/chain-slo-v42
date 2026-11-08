"""Cheap admission cues in common terminal-credit / service-second units.

These cues prune/admit; active displacement still uses the paired dynamic
projector. Whole-root conditional prefix reward is only earned at terminal
completion. The same native token-credit formula is the shared objective, not
a call to the baseline's policy score. No real future lengths are accessed.
"""
import math,statistics
from paired_rollout import _timely_count

def prefill_seconds(costs,tokens,width,context,slots=32):
    if tokens<=0:return 0.
    if hasattr(costs,'prefill_seconds'):return costs.prefill_seconds(tokens,width,context,slots)
    waves=max(1,math.ceil(width/slots));per=tokens/waves
    return waves*max(1,math.ceil(per/2048))*costs.seconds(min(width,slots),context/waves,min(2048,per))

def root_cue(bundle,costs,slots=32):
    if not bundle.supported or not bundle.rows:return 0.,1.,{'supported':False}
    times=[]
    for row in bundle.rows:
        duration=0.
        for st in row.stages:
            width=max(1,min(slots,st.width));waves=math.ceil(st.width/width)
            duration+=st.decode*costs.seconds(width,st.context/waves,0)*waves
            duration+=prefill_seconds(costs,st.prefill,st.width,st.context,slots)
            duration+=st.gap_units*costs.seconds(width,st.context/waves,0)
        times.append(duration)
    # Optimistic isolated feasibility is a cheap cue, never a probability oracle.
    credit=statistics.fmean(row.credit if duration<=bundle.slack else 0. for row,duration in zip(bundle.rows,times))
    return credit,max(.001,statistics.fmean(times)),{'supported':True,'donors':len(times),'isolated_timely_fraction':sum(t<=bundle.slack for t in times)/len(times),'isolated_seconds':statistics.fmean(times)}

def foreground_cue(context,costs,context_tokens=None):
    kind,prompt,generated,prediction,cap,prefill,elapsed,slo=context
    remaining=max(0,min(cap,max(1,int(prediction)))-generated)
    if not remaining and generated<cap:remaining=min(50,cap-generated)
    if kind==0:remaining=min(remaining,50)  # Renewable near-term token/prompt credit.
    ctx=prompt+generated if context_tokens is None else context_tokens;step=costs.seconds(1,ctx,0);prep=prefill_seconds(costs,prefill,1,ctx)
    service=prep+remaining*step
    if kind==1:credit=(prompt+8*(generated+remaining))if elapsed+service<=slo[2]else 0.
    else:
        credit=prompt if generated==0 and remaining and elapsed+prep+step<=slo[0]else 0.
        credit+=8*_timely_count(1,remaining,prep,step,generated,elapsed,slo[0],slo[1])
    return float(credit),max(.001,service)

def compute_target(tbt_budget,planning_reserve=.010,metadata_reserve=.010):
    """Internal target derives from the existing SLO, which is never modified."""
    return max(.005,float(tbt_budget)-planning_reserve-metadata_reserve)
