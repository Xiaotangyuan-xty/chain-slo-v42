"""Shared native foreground-credit primitives used by the production dynamic projector.

Only the first service window differs between alternatives. The continuation
order, donor indices, cost snapshot, physical budgets and planning delay are
shared. Future stages change width/prefill/context at barriers. Omitted future
arrivals are unknown; empirical donor attainment is not a calibrated probability.
"""
import math
import statistics
from stage_scenarios import Scenario, Stage


def foreground_scenario(context):
    kind,prompt,generated,prediction,cap,prefill,elapsed,slo=context
    predicted=min(cap,max(1,int(prediction)));remaining=max(0,predicted-generated)
    if not remaining and generated<cap:remaining=min(cap-generated,50)
    return Scenario((Stage(1,float(remaining),float(prefill),prompt+generated+remaining/2),),
                    prompt+8*(generated+remaining),'QRF_point_prediction')


def _timely_count(lo,hi,t0,per_token,generated,elapsed,ttft,tbt):
    lo=max(lo,2-generated)  # Native output credit excludes the first token.
    if hi<lo:return 0
    slope=per_token-tbt
    intercept=elapsed+t0-ttft-(generated-1)*tbt
    if abs(slope)<1e-12:return hi-lo+1 if intercept<=1e-10 else 0
    boundary=-intercept/slope
    if slope>0:hi=min(hi,math.floor(boundary+1e-9))
    else:lo=max(lo,math.ceil(boundary-1e-9))
    return max(0,hi-lo+1)

