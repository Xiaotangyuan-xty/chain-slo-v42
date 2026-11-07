"""Event-based, paired rolling-window forecasts on identical donor scenarios.

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


def simulate(actions, first, costs, capacity, slots, window, *, scenario=0,
             planning_seconds=0., prefill_limit=2048, block_size=16, event_limit=256, expired=lambda:False):
    states={};deadline_max=planning_seconds
    for a in actions:
        if 'scenarios'in a:
            bundle=a['scenarios']
            if not bundle.rows:continue
            row=bundle.rows[scenario%len(bundle.rows)];slack=bundle.slack
        else:
            row=foreground_scenario(a['foreground']);c=a['foreground']
            slack=(c[7][0]+c[7][1]*max(0,c[2]+row.stages[0].decode-1)-c[6])if c[0]==0 else c[7][2]-c[6]
        if slack<=0 and a['id']not in first:continue
        states[a['id']]={'a':a,'row':row,'sid':0,'decode':row.stages[0].decode,
                         'prefill':row.stages[0].prefill,'gap':0.,'slack':slack,
                         'credit':0.,'complete':False,'finish':None,'generated':0.,'excluded':False}
        deadline_max=max(deadline_max,slack)
    # Charge only members actually displaced/restored, never the entire old set.
    evicted=[a for a in actions if a.get('resident')and a['id']not in first]
    switched=sum(len(a['members'])for a in evicted)
    restored=[a for a in actions if a['id']in first and a.get('swapped')]
    transition_members=switched+sum(len(a['members'])for a in restored)
    transition_context=sum(a.get('resident_context',0.)for a in evicted+restored)
    switch_seconds=costs.transition_seconds(transition_members,transition_context)
    now=planning_seconds+switch_seconds;window_end=now+window
    events=0;complete_envelope=True
    while states and now<=deadline_max+1e-9:
        events+=1
        if events>event_limit or expired():complete_envelope=False;break
        in_window=now<window_end-1e-9
        common_backfill=in_window and not any(aid in first and not s['complete']and s['gap']<=now for aid,s in states.items())
        eligible=[];gaps=[]
        for aid,s in states.items():
            if s['complete']:continue
            if s['slack']<now and not(in_window and aid in first):continue
            if in_window and not common_backfill and aid not in first:s['excluded']=True;continue
            if s['gap']>now:gaps.append(s['gap']);continue
            if (not in_window or common_backfill)and s['excluded']:
                s['prefill']+=s['a'].get('resume_prefill',0.);s['excluded']=False
            eligible.append(s)
        eligible.sort(key=lambda s:(s['a'].get('continuation_priority',(s['a'].get('arrival',0.),)),s['a']['id']))
        allocations=[];free_slots=slots;free_blocks=capacity
        for s in eligible:
            stage=s['row'].stages[s['sid']]
            need=max(1,math.ceil((stage.context/max(1,stage.width))/block_size))
            count=min(stage.width,free_slots,free_blocks//need)
            if count<=0:continue
            allocations.append((s,count,stage));free_slots-=count;free_blocks-=count*need
        if not allocations:
            future=[x for x in gaps if x>now+1e-9]
            if in_window:future.append(window_end)
            if not future:
                if eligible:complete_envelope=False
                break
            now=min(future);continue
        n=sum(k for _,k,_ in allocations)
        context=sum(st.context*k/max(1,st.width)for _,k,st in allocations)
        pref=sum(k for s,k,_ in allocations if s['prefill']>1e-8)
        batch_prefill=min(prefill_limit,sum(s['prefill'] for s,_,_ in allocations if s['prefill']>1e-8))if pref else 0
        step=costs.seconds(n,context,batch_prefill)
        rates=[];dt=math.inf
        for s,k,st in allocations:
            if s['prefill']>1e-8:
                tokens_per_step=prefill_limit*k/max(1,pref)
                delta=max(1,math.ceil(s['prefill']/tokens_per_step-1e-10))*step
                rate=s['prefill']/delta;mode='prefill'
            else:
                rate=(k/max(1,st.width))/step;delta=s['decode']/rate;mode='decode'
            rates.append((s,rate,mode));dt=min(dt,max(0.,delta))
        if in_window:dt=min(dt,window_end-now)
        if gaps:dt=min(dt,min(x-now for x in gaps if x>now))
        if not math.isfinite(dt):complete_envelope=False;break
        # Zero-length stages still advance a barrier on this event.
        for s,rate,mode in rates:
            if mode=='prefill':s['prefill']=max(0.,s['prefill']-rate*dt)
            else:
                before=s['generated'];progress=min(s['decode'],rate*dt)
                s['decode']=max(0.,s['decode']-progress);s['generated']+=progress
                a=s['a']
                if 'foreground'in a and a['foreground'][0]==0:
                    c=a['foreground'];lo=math.floor(before+1e-8)+1;hi=math.floor(s['generated']+1e-8)
                    # k is the additional token index relative to the plan snapshot.
                    t0=now-before/rate
                    s['credit']+=8*_timely_count(lo,hi,t0,1/rate,c[2],c[6],c[7][0],c[7][1])
                    if c[2]==0 and lo<=1<=hi and c[6]+t0+1/rate<=c[7][0]+1e-9:s['credit']+=c[1]
        now+=dt
        for s,_,st in allocations:
            if s['prefill']>1e-8 or s['decode']>1e-8:continue
            s['sid']+=1
            if s['sid']==len(s['row'].stages):
                s['complete']=True;s['finish']=now
                if not('foreground'in s['a']and s['a']['foreground'][0]==0):
                    s['credit']=s['row'].credit if now<=s['slack']+1e-9 else 0.
            else:
                st=s['row'].stages[s['sid']];s['decode']=st.decode;s['prefill']=st.prefill;s['generated']=0.
                s['gap']=now+st.gap_units*costs.seconds(st.width,st.context,0)
        if all(s['complete']or s['slack']<now for s in states.values()):break
    values={a['id']:{'credit':0.,'timely':False,'finish':None}for a in actions}
    for aid,s in states.items():values[aid]={'credit':s['credit'],'timely':s['finish']is not None and s['finish']<=s['slack']+1e-9,'finish':s['finish']}
    return {'credit':sum(v['credit']for v in values.values()),'actions':values,
            'switch_seconds':switch_seconds,'affected_members':transition_members,
            'events':events,'complete_envelope':complete_envelope}


class PairedRollout:
    def __init__(self, actions, reference, costs, capacity, slots, window, *, planning_seconds=0., scenario_limit=8, expired=lambda:False):
        self.actions=actions;self.reference=frozenset(reference);self.costs=costs
        self.capacity,self.slots,self.window=capacity,slots,window
        self.planning_seconds=planning_seconds
        self.expired=expired
        self.scenarios=min(scenario_limit,max((len(a['scenarios'].rows)for a in actions if 'scenarios'in a),default=1))
        self.memo={}

    def evaluate(self, ids):
        key=frozenset(ids)
        if key not in self.memo:
            runs=[]
            for r in range(self.scenarios):
                runs.append(simulate(self.actions,key,self.costs,self.capacity,self.slots,self.window,scenario=r,planning_seconds=self.planning_seconds,expired=self.expired))
                if self.expired():break
            credits={a['id']:statistics.fmean(run['actions'][a['id']]['credit']for run in runs)for a in self.actions}
            self.memo[key]={'credit':statistics.fmean(r['credit']for r in runs),'credit_by_action':credits,
                'probability_by_action':{a['id']:statistics.fmean(r['actions'][a['id']]['timely']for r in runs)for a in self.actions},
                'completion_seconds_by_action':{a['id']:[r['actions'][a['id']]['finish']for r in runs]for a in self.actions},
                'switch_seconds':statistics.fmean(r['switch_seconds']for r in runs),
                'complete_envelope':len(runs)==self.scenarios and all(r['complete_envelope']for r in runs)}
        return self.memo[key]

    def paired(self, ids):
        before=self.evaluate(self.reference);after=self.evaluate(ids)
        differences={aid:after['credit_by_action'][aid]-credit for aid,credit in before['credit_by_action'].items()}
        return {**after,'reference_credit':before['credit'],'delta_credit':after['credit']-before['credit'],
                'resident_loss':sum(-v for aid,v in differences.items()if v<0 and next(a for a in self.actions if a['id']==aid).get('resident')),
                'candidate_gain':sum(max(0.,v)for v in differences.values()),'paired_action_delta':differences,
                'same_window_seconds':self.window,'planning_delay_seconds':self.planning_seconds,
                'complete_envelope':before['complete_envelope']and after['complete_envelope']}
