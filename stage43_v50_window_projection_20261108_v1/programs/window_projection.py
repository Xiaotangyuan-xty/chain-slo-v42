"""Cached scenario envelopes and same-window counterfactual token credit.

Future stages retain their own width/prefill/context. A shared continuation
projection replaces full event simulation for every alternative; this is an
approximation, not a calibrated latency/probability oracle. No test horizon or
future request/output truth is consumed.
"""
from dataclasses import dataclass
import math
import numpy as np
from paired_rollout import foreground_scenario,_timely_count


class ProjectionBudgetExceeded(RuntimeError):
    pass


@dataclass
class Envelope:
    action: dict
    decode: np.ndarray
    prefill_steps: np.ndarray
    prefill_tokens: np.ndarray
    credit: np.ndarray
    tail: np.ndarray
    first_reference: np.ndarray
    slack: float
    width_ratio: float


def _first(row, action, costs, count, context):
    st=row.stages[0]
    ratio=max(1.,st.width/max(1,len(action['members'])))
    return (st.decode*costs.seconds(count,context,0)*ratio
            +math.ceil(st.prefill/2048.)*costs.seconds(count,context,min(2048.,st.prefill)))


def prepare(actions,reference,costs,slots,by_id,expired=lambda:False):
    """Materialize each scenario once, and price future stages once per plan."""
    count=max(1,sum(len(a['members'])for a in actions if a['id']in reference))
    context=sum(by_id[r]['prompt']+by_id[r]['generated']for a in actions if a['id']in reference for r in a['members'])
    # Public current predictions bound the common continuation's background.
    release=[]
    for a in actions:
        if a['id']not in reference:continue
        if 'scenarios'in a and a['scenarios'].rows:
            rows=a['scenarios'].rows
        elif 'foreground'in a:rows=[foreground_scenario(a['foreground'])]
        else:continue
        timeline=[];cursor=0.
        for sid in range(len(rows[0].stages)):
            stages=[row.stages[sid]for row in rows]
            width=max(1,int(round(sum(st.width for st in stages)/len(stages))))
            ctx=sum(st.context for st in stages)/len(stages)
            if sid==0:duration=sum(_first(r,a,costs,count,context)for r in rows)/len(rows)
            else:
                n=max(1,min(slots,width))
                duration=sum(st.decode*costs.seconds(n,st.context,0)*st.width/n
                    +math.ceil(st.prefill/2048.)*costs.seconds(n,st.context,min(2048.,st.prefill))
                    +st.gap_units*costs.seconds(n,st.context,0)for st in stages)/len(stages)
            timeline.append((cursor,cursor+duration,width,ctx));cursor+=duration
        release.append((a['id'],timeline))
    result={}
    for a in actions:
        if expired():raise ProjectionBudgetExceeded('envelope_preparation')
        if 'foreground'in a:continue
        bundle=a['scenarios'];rows=bundle.rows
        if not rows:raise ValueError('Unsupported scenario bundle')
        first_ref=[];tails=[]
        for row in rows:
            current=_first(row,a,costs,count,context);cursor=current;tail=0.
            for stage in row.stages[1:]:
                bg=[(phase[2],phase[3])for aid,phases in release if aid!=a['id']for phase in phases if phase[0]<=cursor<phase[1]]
                own=min(slots,stage.width);background=min(max(0,slots-own),sum(x[0]for x in bg))
                bg_context=sum(x[1]for x in bg)*background/max(1,sum(x[0]for x in bg))
                n=max(1,own+background);ctx=stage.context*own/max(1,stage.width)+bg_context
                service=(stage.decode*costs.seconds(n,ctx,0)*stage.width/max(1,own)
                         +math.ceil(stage.prefill/2048.)*costs.seconds(n,ctx,min(2048.,stage.prefill))
                         +stage.gap_units*costs.seconds(max(1,own),stage.context,0))
                tail+=service;cursor+=service
            first_ref.append(current);tails.append(tail)
        result[a['id']]=Envelope(a,np.asarray([r.stages[0].decode for r in rows],dtype=float),
            np.asarray([math.ceil(r.stages[0].prefill/2048.)for r in rows],dtype=float),
            np.asarray([r.stages[0].prefill for r in rows],dtype=float),
            np.asarray([r.credit for r in rows],dtype=float),np.asarray(tails),np.asarray(first_ref),
            bundle.slack,max(1.,rows[0].stages[0].width/max(1,len(a['members']))))
    return result



def weighted_cdf(times,credits,deadline):
    """Monotone weighted CDF interpolation between historical scenario times.

    Below the shortest scenario: zero; above the longest: full expectation.
    Between support points: integrate a piecewise-uniform service-time bin,
    preserving each next donor's credit weight. This is a declared smoothing
    assumption, not an empirical-frequency/calibration guarantee.
    """
    if not len(times)or deadline<float(np.min(times)):return 0.,0.
    order=np.argsort(times,kind='stable');ts=times[order];cs=credits[order]
    cumulative=np.cumsum(cs)/len(cs);probability=np.arange(1,len(cs)+1,dtype=float)/len(cs)
    return float(np.interp(deadline,ts,cumulative)),float(np.interp(deadline,ts,probability))


def foreground_credit(context,selected,step,prefill_step,reference_step,
                      reference_prefill,window,delay):
    kind,prompt,generated,prediction,cap,prefill,elapsed,slo=context
    remaining=max(0,min(cap,max(1,int(prediction)))-generated)
    if not remaining and generated<cap:remaining=min(cap-generated,50)
    prefill_seconds=math.ceil(prefill/2048.)*prefill_step if prefill else 0.
    reference_prefill_seconds=math.ceil(prefill/2048.)*reference_prefill if prefill else 0.
    if not selected:
        offset=delay+window+reference_prefill_seconds
        if kind==1:return float(prompt+8*(generated+remaining))if elapsed+offset+remaining*reference_step<=slo[2]else 0.
        credit=prompt if generated==0 and remaining and elapsed+offset+reference_step<=slo[0]else 0
        return credit+8*_timely_count(1,remaining,offset,reference_step,generated,elapsed,slo[0],slo[1])
    service=max(0.,window-prefill_seconds)
    first=min(remaining,max(0,int(math.floor(service/max(step,1e-9)+1e-10))))
    if prefill_seconds>window:
        tail_prefill=reference_prefill_seconds*(1-window/prefill_seconds)
    else:tail_prefill=0.
    tail_offset=delay+window+tail_prefill
    if first==remaining:
        finish=delay+prefill_seconds+remaining*step
    else:finish=tail_offset+(remaining-first)*reference_step
    if kind==1:return float(prompt+8*(generated+remaining))if elapsed+finish<=slo[2]else 0.
    credit=0.
    if first:
        if generated==0 and elapsed+delay+prefill_seconds+step<=slo[0]:credit+=prompt
        credit+=8*_timely_count(1,first,delay+prefill_seconds,step,generated,elapsed,slo[0],slo[1])
    if remaining>first:
        if generated+first==0 and elapsed+tail_offset+reference_step<=slo[0]:credit+=prompt
        credit+=8*_timely_count(1,remaining-first,tail_offset,reference_step,generated+first,elapsed,slo[0],slo[1])
    return credit


class WindowProjector:
    def __init__(self,actions,reference,costs,capacity,slots,by_id,window,planning_delay,
                 expired=lambda:False,swap_capacity_blocks=512,block_size=16):
        self.swap_capacity_blocks=swap_capacity_blocks;self.block_size=block_size
        self.actions=actions;self.reference=frozenset(reference);self.costs=costs
        self.by_id=by_id;self.window=window;self.delay=planning_delay;self.expired=expired
        self.envelopes=prepare(actions,reference,costs,slots,by_id,expired)
        self.memo={}
        n=max(1,sum(len(a['members'])for a in actions if a['id']in reference))
        ctx=sum(by_id[r]['prompt']+by_id[r]['generated']for a in actions if a['id']in reference for r in a['members'])
        self.reference_count=n;self.reference_context=ctx
        self.reference_step=costs.seconds(n,ctx,0);self.reference_prefill=costs.seconds(n,ctx,2048)

    def evaluate(self,ids):
        ids=frozenset(ids)
        if ids in self.memo:return self.memo[ids]
        if self.expired():raise ProjectionBudgetExceeded('set_evaluation')
        selected=[r for a in self.actions if a['id']in ids for r in a['members']]
        n=max(1,len(selected));ctx=sum(self.by_id[r]['prompt']+self.by_id[r]['generated']for r in selected)
        step=self.costs.seconds(n,ctx,0);prefill=self.costs.seconds(n,ctx,2048)
        evicted=[r for a in self.actions if a.get('resident')and a['id']not in ids for r in a['members']if r in self.by_id and self.by_id[r].get('state')=='running']
        restored=[r for a in self.actions if a['id']in ids for r in a['members']if self.by_id[r].get('state')=='swapped']
        # Reserve the *joint* CPU swap budget in the same eviction order as
        # the executor. Individual can_swap checks are not additive guarantees.
        free=self.swap_capacity_blocks;recompute={}
        ordered=sorted(evicted,key=lambda r:tuple(self.by_id[r].get('native_priority',(0.,))),reverse=True)
        for rid in ordered:
            m=self.by_id[rid];need=math.ceil((m['prompt']+m['generated'])/self.block_size)
            if m.get('can_swap',True)and need<=free:free-=need
            else:recompute[rid]=max(0,m['prompt']+m['generated']-m.get('prefill_tokens',0))
        affected=evicted+restored
        switch=self.costs.transition_seconds(len(affected),sum(self.by_id[r]['prompt']+self.by_id[r]['generated']for r in affected))
        delay=self.delay+switch;credits={};probabilities={};completion={}
        for a in self.actions:
            if 'foreground'in a:
                p=min(2048.,a['foreground'][5])
                own_prefill=self.costs.seconds(n,ctx,p)if p else 0.
                ref_prefill=self.costs.seconds(self.reference_count,self.reference_context,p)if p else 0.
                context=list(a['foreground'])
                if a['id']not in ids:context[5]+=sum(recompute.get(r,0)for r in a['members'])
                if context[5]!=a['foreground'][5]:
                    p=min(2048.,context[5]);ref_prefill=self.costs.seconds(self.reference_count,self.reference_context,p)
                value=foreground_credit(tuple(context),a['id']in ids,step,own_prefill,self.reference_step,ref_prefill,self.window,delay)
                credits[a['id']]=value;probabilities[a['id']]=float(value>0);continue
            e=self.envelopes[a['id']]
            if a['id']in ids:
                prefill_values=np.asarray([self.costs.seconds(n,ctx,min(2048.,t))if t else 0. for t in e.prefill_tokens])
                first=e.decode*step*e.width_ratio+e.prefill_steps*prefill_values
                fraction=np.minimum(1.,self.window/np.maximum(first,1e-12))
                times=delay+np.where(first<=self.window,first,self.window+(1-fraction)*e.first_reference)+e.tail
            else:
                restore=sum(recompute.get(r,0)for r in a['members'])/2048.*self.reference_prefill
                times=delay+self.window+e.first_reference+e.tail+restore
            value,probability=weighted_cdf(times,e.credit,e.slack)
            credits[a['id']]=value;probabilities[a['id']]=probability
            completion[a['id']]=times.tolist()
        result={'credit':sum(credits.values()),'credit_by_action':credits,
                'probability_by_action':probabilities,'completion_seconds_by_action':completion,
                'switch_seconds':switch,'affected_members':len(affected),'recompute_members':sorted(recompute),'CPU_swap_blocks_reserved':self.swap_capacity_blocks-free,'complete_envelope':True}
        self.memo[ids]=result;return result

    def paired(self,ids):
        before=self.evaluate(self.reference);after=self.evaluate(ids)
        delta={a['id']:after['credit_by_action'][a['id']]-before['credit_by_action'][a['id']]for a in self.actions}
        return {**after,'reference_credit':before['credit'],'delta_credit':after['credit']-before['credit'],
                'resident_loss':sum(max(0.,-delta[a['id']])for a in self.actions if a.get('resident')),
                'candidate_gain':sum(max(0.,v)for v in delta.values()),'paired_action_delta':delta,
                'same_window_seconds':self.window,'planning_delay_seconds':self.delay}
