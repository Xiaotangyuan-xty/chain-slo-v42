"""Proposal-specific bounded event rollout from the exact physical running set.

Each member can finish before a barrier; future stages change width/prefill and
background overlap. First-window evictions must be restored at its fixed end.
Subsequent continuation uses this mechanism's public credit/work order, never
A/Astar's score. Unknown future arrivals are excluded, so estimates are not a
calibrated completion-probability guarantee.
"""
import math
import statistics
import numpy as np
from paired_rollout import foreground_scenario,_timely_count

class ProjectionBudgetExceeded(RuntimeError):pass

def weighted_cdf(times,credits,deadline):
    if not len(times):return 0.,0.
    timely=np.asarray(times)<=deadline
    return float(np.sum(np.asarray(credits)*timely)/len(times)),float(np.mean(timely))

class WindowProjector:
    def __init__(self,actions,reference,costs,capacity,slots,by_id,window,planning_delay,
                 expired=lambda:False,swap_capacity_blocks=512,block_size=16,event_limit=256,scenario_limit=4):
        self.actions=actions;self.reference=frozenset(reference);self.costs=costs;self.capacity=capacity;self.slots=slots
        self.by_id=by_id;self.window=window;self.delay=planning_delay;self.expired=expired;self.swap_capacity=swap_capacity_blocks
        self.block_size=block_size;self.event_limit=event_limit;self.scenario_limit=scenario_limit;self.memo={}
        self.owner={r:a['id']for a in actions for r in a['members']}
        if not self.reference<=set(by_id):raise ValueError('Reference must contain physical request IDs')
    def _switch(self,selected):
        evicted=[r for r in self.by_id if r in self.reference and r not in selected]
        evicted.sort(key=lambda r:tuple(self.by_id[r].get('native_priority',(0.,))),reverse=True)
        restored=[r for r in selected if self.by_id[r].get('state')=='swapped'];free=self.swap_capacity;recompute={}
        for r in evicted:
            m=self.by_id[r];n=math.ceil((m['prompt']+m['generated'])/self.block_size)
            if m.get('can_swap',True)and n<=free:free-=n
            else:recompute[r]=max(0,m['prompt']+m['generated']-m.get('prefill_tokens',0))
        affected=evicted+restored
        switch=self.costs.transition_seconds(len(affected),sum(self.by_id[r]['prompt']+self.by_id[r]['generated']for r in affected))
        restore=self.costs.transition_seconds(len(evicted),sum(self.by_id[r]['prompt']+self.by_id[r]['generated']for r in evicted if r not in recompute))
        return evicted,recompute,switch,restore,self.swap_capacity-free
    def _one(self,selected,index):
        evicted,recompute,switch,restore,reserved=self._switch(selected);states={};finish={};latency_credit={};now=self.delay+switch
        window_end=now+self.window;restore_end=window_end+self.window;restore_charged=False
        def stage_members(a,row,sid):
            st=row.stages[sid];result=[]
            for i in range(st.width):
                rid=(st.member_ids[i]if st.member_ids else(a['members'][i]if sid==0 and i<len(a['members'])else None))
                # Virtual missing siblings/future stages are modeled, never
                # returned as physical execution candidates.
                remaining=st.member_decode[i]if st.member_decode else st.decode
                mid=st.member_context[i]if st.member_context else st.context/max(1,st.width)
                growth=st.member_context_growth[i]if st.member_context_growth else remaining
                start=st.member_start_context[i]if st.member_start_context else max(0.,mid-growth/2)
                result.append({'rid':rid,'remaining':remaining,
                  'prefill':st.member_prefill[i]if st.member_prefill else st.prefill/max(1,st.width),
                  'context':start,'context_growth_per_unit':growth/max(1e-12,remaining),'done':False,'emitted':0.})
            return result
        for a in self.actions:
            if self.expired():raise ProjectionBudgetExceeded('scenario_materialization')
            row=(a['scenarios'].rows[index%len(a['scenarios'].rows)]if'scenarios'in a else foreground_scenario(a['foreground']))
            ms=stage_members(a,row,0)
            for m in ms:
                if m['rid']in recompute:m['prefill']+=recompute[m['rid']]
            slack=a['scenarios'].slack if'scenarios'in a else (a['foreground'][7][2]-a['foreground'][6]if a['foreground'][0]==1 else a['foreground'][7][0]+a['foreground'][7][1]*max(0,a['foreground'][2]+row.stages[0].decode-1)-a['foreground'][6])
            states[a['id']]={'a':a,'row':row,'sid':0,'members':ms,'gap_until':now,'slack':slack,'complete':False}
            latency_credit[a['id']]=0.
        max_time=max([s['slack']for s in states.values()]+[restore_end])+1e-6
        events=0;restored_order=[]
        while now<=max_time and any(not s['complete']for s in states.values()):
            events+=1
            if events>self.event_limit or self.expired():raise ProjectionBudgetExceeded('dynamic_event_budget')
            if now>=window_end-1e-9 and not restore_charged:
                now+=restore;restore_end=now+self.window;restore_charged=True
            phase=0 if now<window_end-1e-9 else 1 if now<restore_end-1e-9 else 2
            eligible=[];future=[]
            for aid,s in states.items():
                if s['complete']:continue
                if s['gap_until']>now+1e-9:future.append(s['gap_until']);continue
                a=s['a'];density=a.get('cheap_credit',0.)/max(1.,a.get('cheap_work',1.))
                for m in s['members']:
                    if m['done']:continue
                    if phase==0 and m['rid']not in selected and not(s['sid']>0 and set(a['members'])&set(selected)):continue
                    priority=(0 if phase==1 and m['rid']in evicted else 1,-density,a.get('arrival',0.),aid)
                    eligible.append((priority,s,m))
            eligible.sort(key=lambda item:item[0]);allocated=[];slots=self.slots;blocks=self.capacity
            for _,s,m in eligible:
                charge=max(1,math.ceil((m['context']+min(50.,max(0.,m['remaining'])))/self.block_size))
                if slots and charge<=blocks:allocated.append((s,m));slots-=1;blocks-=charge
            if not allocated:
                if phase==0:future.append(window_end)
                elif phase==1:future.append(restore_end)
                if not future:break
                now=min(t for t in future if t>now+1e-9);continue
            n=len(allocated);context=sum(m['context']for s,m in allocated);pref=[(s,m)for s,m in allocated if m['prefill']>1e-9]
            prefill_total=sum(m['prefill']for s,m in pref);step=self.costs.seconds(n,context,min(2048.,prefill_total))
            if step<=0 or not math.isfinite(step):raise ValueError('Nonpositive execution cost')
            # Each prefill quantum consumes an entire executor iteration. When
            # chunked, all selected decoders make one token of progress too.
            rates=[];dt=math.inf
            for s,m in allocated:
                if m['prefill']>1e-9:
                    share=2048.*m['prefill']/max(1.,prefill_total);iterations=max(1,math.ceil(m['prefill']/share-1e-10));duration=iterations*step
                    rate=m['prefill']/duration;kind='prefill'
                else:rate=1./step;duration=max(0.,m['remaining'])/rate;kind='decode'
                rates.append((s,m,rate,kind));dt=min(dt,duration)
            boundary=window_end if phase==0 else restore_end if phase==1 else math.inf
            # Do not split a prefill iteration into fictional fractional calls.
            if not pref:dt=min(dt,max(0.,boundary-now))
            if not math.isfinite(dt)or dt<=1e-10:
                now=max(now+1e-8,boundary)if math.isfinite(boundary)else now+1e-8;continue
            for s,m,rate,kind in rates:
                a=s['a']
                if kind=='prefill':m['prefill']=max(0.,m['prefill']-rate*dt);continue
                work=min(m['remaining'],rate*dt);old=m['emitted'];m['emitted']+=work;m['remaining']=max(0.,m['remaining']-work);m['context']+=work*m['context_growth_per_unit']
                if 'foreground'in a and a['foreground'][0]==0:
                    c=a['foreground'];lo=int(math.floor(old+1e-8))+1;hi=int(math.floor(m['emitted']+1e-8));per_token=1./rate;offset=now-old/rate
                    if c[2]==0 and old<1<=m['emitted']and c[6]+offset+per_token<=c[7][0]:latency_credit[a['id']]+=c[1]
                    latency_credit[a['id']]+=8*_timely_count(lo,hi,offset,per_token,c[2],c[6],c[7][0],c[7][1])
                if m['remaining']<=1e-8:m['done']=True
            now+=dt
            for aid,s in states.items():
                if s['complete']or not all(m['done']for m in s['members']):continue
                s['sid']+=1
                if s['sid']==len(s['row'].stages):s['complete']=True;finish[aid]=now
                else:
                    stage=s['row'].stages[s['sid']];s['members']=stage_members(s['a'],s['row'],s['sid']);s['gap_until']=now+stage.gap_units*self.costs.seconds(min(self.slots,stage.width),stage.context,0)
        credits={};probabilities={}
        for aid,s in states.items():
            a=s['a'];timely=aid in finish and finish[aid]<=s['slack'];probabilities[aid]=float(timely)
            credits[aid]=latency_credit[aid]if'foreground'in a and a['foreground'][0]==0 else float(s['row'].credit)if timely else 0.
        return {'credit_by_action':credits,'probability_by_action':probabilities,'finish':finish,'events':events,'switch_seconds':switch,'restoration_seconds':restore,'recompute_members':sorted(recompute),'CPU_swap_blocks_reserved':reserved,'eviction_order':evicted,'affected_members':len(evicted)+sum(self.by_id[r].get('state')=='swapped'for r in selected)}
    def evaluate(self,ids):
        ids=frozenset(ids)
        if ids in self.memo:return self.memo[ids]
        if not ids<=set(self.by_id):raise ValueError('Nonphysical selection')
        count=max([len(a['scenarios'].rows)for a in self.actions if'scenarios'in a]+[1]);count=min(count,self.scenario_limit)
        rows=[self._one(ids,i)for i in range(count)];credits={a['id']:statistics.fmean(r['credit_by_action'][a['id']]for r in rows)for a in self.actions};probs={a['id']:statistics.fmean(r['probability_by_action'][a['id']]for r in rows)for a in self.actions}
        result={'credit':sum(credits.values()),'credit_by_action':credits,'probability_by_action':probs,'completion_seconds_by_action':{a['id']:[r['finish'].get(a['id'],float('inf'))for r in rows]for a in self.actions},'switch_seconds':rows[0]['switch_seconds'],'restoration_seconds':rows[0]['restoration_seconds'],'eviction_order':rows[0]['eviction_order'],'affected_members':rows[0]['affected_members'],'recompute_members':rows[0]['recompute_members'],'CPU_swap_blocks_reserved':rows[0]['CPU_swap_blocks_reserved'],'complete_envelope':True,'dynamic_events':sum(r['events']for r in rows),'reference_semantics':'exact_physical_running'}
        # None denotes censored beyond all modeled credit deadlines; avoid
        # nonfinite JSON values while retaining probability zero.
        result['completion_seconds_by_action']={k:[v if math.isfinite(v)else None for v in vs]for k,vs in result['completion_seconds_by_action'].items()}
        self.memo[ids]=result;return result
