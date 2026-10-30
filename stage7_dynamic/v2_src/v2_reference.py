"""One first action then EDF; explicit independent tool timers, no GPU billing."""
import math,time
from slo_selector import score as legacy_score

def forecasts(view,predictor,profile,conformal=True,include_away=True,errors=None):
    jobs=[]
    for goal in view.goals:
        if goal.state=='AWAY' and not include_away:continue
        try:est=predictor.estimate(goal,profile,conformal)
        except (ValueError,RuntimeError) as exc:
            if errors is None:raise
            errors.append({'program':goal.program,'phase':goal.state,'reason':str(exc)});continue
        c=goal.ready
        prefix=(c.prompt_tokens+max(0,c.emitted-1) if c.emitted else c.produced_prefix) if c else goal.produced_prefix
        recovery=profile.recovery(view.cache.get(goal.program),prefix)
        phases=list(est['phases'])
        if recovery:
            idx=next(i for i,x in enumerate(phases) if x[0]=='compute')
            phases[idx]=('compute',phases[idx][1]+recovery)
        key=c.key if c else 'away:'+goal.program
        jobs.append({'key':key,'program':goal.program,'state':goal.state,'deadline':goal.deadline,
             **est,'work':est['work']+recovery,'point':est['point']+recovery,
             'current':est['current']+recovery if c else 0.,'recovery':recovery,'phases':tuple(phases)})
    return jobs

def legacy(jobs,key,h,now):
    ready=[j for j in jobs if j['state']!='AWAY']
    if not ready:raise ValueError('legacy has no legal first target')
    # The exact V1 kernel is shared for degenerate zero-AWAY equivalence.
    score=legacy_score(ready,key,h,now)
    return {'score':score,'finished_goals':len(ready),'goal_denominator':len(ready),
            'gpu_busy':math.fsum(j['work'] for j in ready),'idle':0.,'elapsed':math.fsum(j['work'] for j in ready),
            'timer_intervals':0,'initial_compute':math.fsum(j['work'] for j in ready),'kernel':'shared_v1'}

def project(jobs,key,h,now,deadline=math.inf,allow_shared_zero=True):
    if not jobs:raise ValueError('empty goal denominator')
    if allow_shared_zero and all(j['state']!='AWAY' and not any(k=='away' and v>0 for k,v in j['phases']) for j in jobs):
        return legacy(jobs,key,h,now)
    states={};t=0.;busy=0.;idle=0.;finished={};timer_intervals=0;events=0
    initial_compute=math.fsum(v for j in jobs for k,v in j['phases'] if k=='compute')
    for j in jobs:
        states[j['key']]={'job':j,'phases':list(j['phases']),'idx':0,'remaining':0.,'ready_at':0.}
    def transition(k,at=None):
        nonlocal timer_intervals
        stamp=t if at is None else at
        s=states[k];ph=s['phases']
        if s['idx']>=len(ph):finished[k]=now+stamp;return
        kind,duration=ph[s['idx']];s['idx']+=1
        if duration<0 or not math.isfinite(duration):raise ValueError('invalid phase')
        if kind=='away':
            timer_intervals+=int(duration>0);s['ready_at']=stamp+duration;s['remaining']=None
        elif kind=='compute':s['ready_at']=stamp;s['remaining']=duration
        else:raise ValueError('phase kind')
    for k in states:transition(k)
    def activate_timers():
        changed=True
        while changed:
            changed=False
            for k,s in states.items():
                if k in finished:continue
                if s['remaining'] is None and s['ready_at']<=t+1e-12:
                    transition(k,s['ready_at']);changed=True
                elif s['remaining']==0. and s['ready_at']<=t+1e-12:
                    transition(k,s['ready_at']);changed=True
    def run_compute(k,length):
        nonlocal t,busy
        s=states[k]
        if s['remaining'] is None or s['ready_at']>t+1e-12:raise ValueError('AWAY not dispatchable')
        duration=min(length,s['remaining']);s['remaining']-=duration;t+=duration;busy+=duration
        if s['remaining']<1e-10:s['remaining']=0.;transition(k)
    activate_timers()
    if key is not None:
        chosen=states.get(key)
        if chosen is None or chosen['job']['state']=='AWAY':raise ValueError('first action must target READY')
        run_compute(key,min(h,chosen['job']['current']));activate_timers()
    while len(finished)<len(states):
        events+=1
        if events>1024 or time.perf_counter()>deadline:raise TimeoutError('reference budget')
        activate_timers()
        ready=[(s['job']['deadline'],k) for k,s in states.items() if k not in finished and s['remaining'] is not None and s['ready_at']<=t+1e-12]
        if ready:
            k=min(ready)[1];run_compute(k,states[k]['remaining'])
        else:
            times=[s['ready_at'] for k,s in states.items() if k not in finished]
            if not times:break
            nxt=min(times)
            if nxt<t-1e-10:raise RuntimeError('calendar time regression')
            idle+=max(0.,nxt-t);t=nxt
    if len(finished)!=len(jobs):raise RuntimeError('goal denominator incomplete')
    if not math.isclose(busy,initial_compute,abs_tol=1e-8):raise RuntimeError('compute conservation')
    if not math.isclose(t,busy+idle,abs_tol=1e-8):raise RuntimeError('wall conservation')
    good=sum(finished[j['key']]<=j['deadline'] for j in jobs)
    mean=sum(finished.values())/len(jobs)-now
    return {'score':(good,-mean),'finishes':finished,'finished_goals':len(finished),'goal_denominator':len(jobs),
       'gpu_busy':busy,'initial_compute':initial_compute,'idle':idle,'elapsed':t,
       'timer_intervals':timer_intervals,'kernel':'event_calendar'}
