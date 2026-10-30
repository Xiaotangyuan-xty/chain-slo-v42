"""Ports of the two paper algorithms with explicitly registered common features.

Only primitive public views are accepted. No replay specification enters here.
"""
import math,statistics,time
from contracts import CacheEntry,quantile

class Performance:
    def __init__(self,p):
        self.p=p;self.decode_seconds=p['decode_anchor'];self.copy_seconds_per_token=p['copy_seconds_per_token']
        self.reset_online()
    def reset_online(self):self.factors={B:1. for B in [1,2,3,4]}
    def prefill(self,N,C=0):
        if N<=0:return 0.
        if N+C>2050:raise ValueError('prefill outside native profile')
        a=self.p['prefill_coefficients'];x=(1,N,N*N,N*C,C,C*C)
        return max(1e-5,sum(v*w for v,w in zip(a,x)))
    def iteration(self,B,S):
        a=self.p['decode_coefficients'];return max(.001,a[0]+a[1]*B+a[2]*S)*self.factors.get(B,1.)
    def observe(self,B,S,seconds):
        a=self.p['decode_coefficients'];base=max(.001,a[0]+a[1]*B+a[2]*S)
        ratio=min(2.,max(.5,seconds/base));old=self.factors[B]
        self.factors[B]=min(2.,max(1.,.98*old+.02*ratio))
        return {'B':B,'S':S,'seconds':seconds,'base':base,'ratio':ratio,'old_factor':old,'new_factor':self.factors[B]}
    def recovery(self,entry,needed):
        if entry.gpu>=needed:return 0.
        c=max(entry.gpu,min(entry.host,needed))
        return max(0,c-entry.gpu)*self.copy_seconds_per_token+self.prefill(needed-c,c)

def features(g):
    c=g.ready
    return [c.prompt_tokens,c.ordinal,c.emitted,len(g.history),
        sum(h.outputs for h in g.history),sum(h.prompt_tokens for h in g.history)]

def gmax(rows,B=4,cutoff=.95):
    if not rows:return []
    B=min(B,len(rows));v=sorted(rows,key=lambda r:(-r['priority'],r['key']))
    minimum=v[B-1]['priority']*cutoff
    pool=sorted([x for x in rows if x['priority']>=minimum],key=lambda r:(r['length'],r['key']))
    groups=[pool[i:i+B] for i in range(max(1,len(pool)-B+1))]
    return max(groups,key=lambda x:(sum(r['priority'] for r in x),tuple(r['key'] for r in x)))

def pack_route(candidates,eta=.5,tau=.05):
    """Paper Algorithm 1, equations9/10 and §6.2; no fake queue ordering."""
    A=[x for x in candidates if x['capacity']]
    if not A:return None,{'reason':'capacity_reject'}
    w=min(x['W'] for x in A);Z=A[0]['Z']
    C=[x for x in A if Z<=0 or (x['W']-w)/Z<=eta+1e-12]
    T=[x for x in C if x['tpot'] is not None and x['tpot']<tau]
    chosen=min(T,key=lambda x:(-x['decoding'],-x['cached'],x['queued'],x['instance'])) if T else min(C,key=lambda x:(x['active']+x['prefill_backlog'],-x['cached'],x['instance']))
    return chosen['instance'],{'capacity_set':[x['instance'] for x in A],'cache_set':[x['instance'] for x in C],
        'SLO_set':[x['instance'] for x in T],'fallback':not bool(T),'eta':eta,'tau':tau,'candidates':candidates}

def route(prompt,pid,visible,pool_cache,perf,arm,median_output,remaining_P95=None):
    rows=[]
    for i in range(2):
        own=[x for x in visible if x['instance']==i]
        cached=pool_cache[i].get(pid).gpu
        cached=min(prompt,cached);N=prompt-cached
        base=perf.prefill(prompt,0)-perf.prefill(0,prompt)
        W=max(0,perf.prefill(N,cached)-perf.prefill(0,prompt))
        B=len(own)+1;S=sum(x['prompt']+x['emitted'] for x in own)+prompt
        supported=B<=4 and S<=8192
        iter_cost=perf.iteration(min(4,B),min(S,8192))
        # Single-GPU replicas share service time; cross-replica interference is
        # common and exposed. The paper's admission estimate remains per engine.
        tpot=iter_cost+(perf.prefill(N,cached)/max(1,median_output) if own else 0) if supported else None
        rows.append({'instance':i,'capacity':prompt<=8192,'W':W,'Z':base,'cached':cached,
            'tpot':tpot,'prediction_supported':supported,'projected_B':B,'projected_S':S,'active':len(own),'decoding':sum(x['emitted']>0 for x in own),
            'queued':max(0,len(own)-4),'prefill_backlog':sum(x['prompt'] for x in own if not x['emitted'])/1024})
    if arm=='PackServe_port':return pack_route(rows)
    if arm=='JITServe_GMAX_port':
        # K=2 replica-specific dummy priorities (§4.3): both available replicas.
        for x in rows:
            B=min(4,x['active']+1)
            S=sum(z['prompt']+z['emitted'] for z in visible if z['instance']==x['instance'])+prompt
            x['replica_generation_upper']=max(1,remaining_P95 or median_output)*perf.iteration(B,S)+x['W']
        chosen=min(rows,key=lambda x:(x['replica_generation_upper'],x['instance']))
        return chosen['instance'],{'power_of_K':2,'candidates':rows}
    # Common deployment router for the two per-instance scheduling algorithms;
    # it is an infrastructure component, never an additional comparison arm.
    z=min(rows,key=lambda x:(x['W']+x['active']*perf.decode_seconds,-x['cached'],x['instance']))
    return z['instance'],{'common_cache_load_router':True,'candidates':rows}

def jit_batch(goals,now,instance_keys,qrf,graphs,perf,age,previous,cache):
    rows=[]
    active=[g for g in goals if g.ready is not None and g.ready.key in instance_keys]
    predictions=qrf.predict([features(g) for g in active],quantiles=.95) if active else []
    for g,estimated in zip(active,predictions):
        c=g.ready
        # P95 QRF uses only public numeric features, trained afresh on FIT.
        remaining=max(1.,float(estimated))
        # Gaussian prefix-pattern match, accumulated execution share (§4.1).
        matches=[]
        for graph in graphs:
            if len(graph['calls'])<=c.ordinal:continue
            score=((graph['calls'][c.ordinal]['prompt']-c.prompt_tokens)/256)**2
            for j,h in enumerate(g.history):
                if j>=len(graph['calls']):score+=1000;break
                score+=((graph['calls'][j]['outputs']-h.outputs)/64)**2
            matches.append((score,graph))
        graph=min(matches,key=lambda x:x[0])[1] if matches else None
        share=graph['shares'][c.ordinal] if graph else 1.
        subdeadline=g.arrival+(g.deadline-g.arrival)*share
        B=min(4,len(instance_keys))
        context=sum(z.ready.prompt_tokens+z.ready.emitted for z in goals if z.ready and z.ready.key in instance_keys)/max(1,len(instance_keys))*B
        generation=remaining*perf.iteration(B,context)
        feasible=subdeadline-now>=generation
        reward=(1. if feasible else 0.)+.01*age.get(c.key,0)
        priority=reward/max(generation,1e-6)
        rows.append({'key':c.key,'length':c.prompt_tokens+c.emitted,'priority':priority,
            'remaining_P95':remaining,'subdeadline':subdeadline,'accumulated_share':share,'generation_upper':generation})
    chosen=gmax(rows)
    # Paper cost-aware preemption: require increased grouped payoff to repay
    # measured recovery; comparisons share the public profile and cache state.
    scores={r['key']:r['priority'] for r in rows};old=[k for k in previous if k in scores]
    stall=0.
    for g in goals:
        c=g.ready
        if c is not None and c.key in {r['key'] for r in chosen}-set(old):
            e=cache.get(g.program);needed=c.prompt_tokens+max(0,c.emitted-1)
            stall+=perf.recovery(e,needed)
    if len(old)==len(chosen) and old:
        new_score=sum(scores[x['key']] for x in chosen);old_score=sum(scores[k] for k in old)
        frame=50*perf.iteration(len(chosen),sum(x['length'] for x in chosen))
        if new_score<=1.1*old_score or (new_score-old_score)*frame<=stall*old_score:
            chosen=[next(x for x in rows if x['key']==k) for k in old]
    return [r['key'] for r in chosen],{'algorithm':'GMAX','rows':rows,'cutoff':.95,'preemption_margin':.1,'estimated_stall_seconds':stall}
