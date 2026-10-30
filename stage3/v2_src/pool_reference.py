"""One finite action then identical modeled batching with independent timers."""
import math,time

def project(jobs,action,now,perf,first_instance,deadline=math.inf):
    states={j['key']:{'job':j,'phases':list(j['phases']),'idx':0,'remaining':0.,'ready_at':0.} for j in jobs}
    t=0.;busy=0.;idle=0.;finished={}
    def transition(k,at):
        s=states[k]
        if s['idx']==len(s['phases']):finished[k]=now+at;return
        kind,v=s['phases'][s['idx']];s['idx']+=1
        if kind=='away':s['remaining']=None;s['ready_at']=at+v
        else:s['remaining']=v;s['ready_at']=at
    for k in states:transition(k,0.)
    def activate():
        for z in range(50):
            changed=False
            for k,s in states.items():
                if k not in finished and s['remaining'] is None and s['ready_at']<=t+1e-10:
                    transition(k,s['ready_at']);changed=True
            if not changed:return
        raise RuntimeError('timer transition cycle')
    def batch(instance,anchor=None):
        ready=sorted((s['job']['deadline'],k) for k,s in states.items() if k not in finished and s['remaining'] is not None and s['ready_at']<=t+1e-10 and s['job']['instance']==instance)
        keys=[k for _,k in ready]
        if anchor in keys:keys.remove(anchor);keys.insert(0,anchor)
        return keys[:4]
    def run(keys,cap=math.inf):
        nonlocal t,busy
        B=len(keys);S=sum(states[k]['job']['context'] for k in keys)
        speed=perf.decode_seconds/perf.iteration(B,S)
        work=min(min(states[k]['remaining'] for k in keys),cap)
        duration=work/speed;t+=duration;busy+=duration
        for k in keys:
            states[k]['remaining']-=work
            if states[k]['remaining']<1e-9:transition(k,t)
    activate();k,h=action
    if states[k]['job']['state']=='AWAY':raise ValueError('AWAY cannot be an immediate dispatch target')
    first=batch(first_instance,k)
    if k not in first:raise ValueError('AWAY/nonlocal immediate dispatch')
    run(first,h);instance=1-first_instance
    for event in range(2048):
        if len(finished)==len(jobs):break
        if time.perf_counter()>deadline:raise TimeoutError('batched reference timeout')
        activate();keys=batch(instance)
        if not keys:keys=batch(1-instance)
        if keys:run(keys);instance=1-instance
        elif len(finished)<len(jobs):
            nxt=min(s['ready_at'] for key,s in states.items() if key not in finished)
            idle+=max(0,nxt-t);t=max(t,nxt)
    if len(finished)!=len(jobs):raise RuntimeError('incomplete modeled denominator')
    if not math.isclose(t,busy+idle,abs_tol=1e-6):raise RuntimeError('modeled clock conservation')
    return {'good':sum(finished[j['key']]<=j['deadline'] for j in jobs),
        'mean':statistics_mean(finished.values())-now,'denominator':len(jobs),'busy':busy,'idle':idle}
def statistics_mean(xs):return math.fsum(xs)/len(xs)
