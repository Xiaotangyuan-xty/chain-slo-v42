"""Same group conformal recipe, extended only to observed AWAY phases.

History labels are generated on registered FIT/CAL only. Online calls accept
primitive observed features. Future mean phases are conditional historical
statistics, never an evaluator callback or a borrowed absolute next prefix.
"""
from dataclasses import dataclass,field
from types import MappingProxyType
import bisect,math,statistics
from contracts import digest
from v2_design import workload

def bucket(prompt,ordinal):return (prompt//128,min(ordinal,5))
def scale(x):return max(16.,.25*x)
def rows(inp,profile):
    d=profile.decode_seconds;out=[]
    for p in inp['programs']:
        cs=p['calls']
        for i,c in enumerate(cs):
            prior=0 if not i else cs[i-1]['prompt_tokens']+cs[i-1]['outputs']-1
            extra=profile.prefill(c['prompt_tokens']-prior,prior)/d-1
            tail=[]
            for j in range(i+1,len(cs)):
                last=cs[j-1];x=cs[j];prev=last['prompt_tokens']+last['outputs']-1
                comp=x['outputs']-1+profile.prefill(x['prompt_tokens']-prev,prev)/d
                tail.extend((last['away']/d,comp))
            out.append({'seed':inp['seed'],'program':p['id'],'ordinal':i,'prompt':c['prompt_tokens'],
                'outputs':c['outputs'],'extra':extra,'tail':tuple(tail),'future':math.fsum(tail),'tool':c['away']/d})
    return out

def mean_vector(rs):
    n=max((len(r[4]) for r in rs),default=0)
    return tuple(statistics.fmean(r[4][j] if j<len(r[4]) else 0. for r in rs) for j in range(n))

@dataclass(frozen=True)
class Predictor:
    groups:tuple
    q_cal:float
    provenance:tuple
    version:str='away-timer-group-conformal-v2'
    _ready:object=field(init=False,repr=False,compare=False)
    _away:object=field(init=False,repr=False,compare=False)

    def __post_init__(self):
        ready={};away={}
        for k,records in self.groups:
            for e in range(max(r[1] for r in records)):
                rs=[r for r in records if r[1]>e]
                c=statistics.fmean(r[1]-e+(r[2] if not e else 0.) for r in rs)
                t=statistics.fmean(r[1]-e+(r[2] if not e else 0.)+r[3] for r in rs)
                ready[k+(e,)]=(max(.01,c),max(.01,t),mean_vector(rs),len(rs))
            for outputs in sorted({r[1] for r in records}):
                rs=sorted((r for r in records if r[1]==outputs and r[4] and r[4][0]>0),key=lambda r:r[4][0])
                if not rs:continue
                times=tuple(r[4][0] for r in rs);vectors=[]
                for idx in range(len(rs)):vectors.append(mean_vector(rs[idx:]))
                away[k+(outputs,)]=(times,tuple(vectors))
        object.__setattr__(self,'_ready',MappingProxyType(ready));object.__setattr__(self,'_away',MappingProxyType(away))

    def ready_base(self,prompt,ordinal,emitted):
        try:return self._ready[bucket(prompt,ordinal)+(emitted,)]
        except KeyError:raise ValueError('READY prediction support exhausted/unknown bucket')

    def away_base(self,prompt,ordinal,outputs,elapsed_units):
        try:times,vecs=self._away[bucket(prompt,ordinal)+(outputs,)]
        except KeyError:raise ValueError('AWAY history support unknown')
        idx=bisect.bisect_right(times,elapsed_units)
        if idx==len(times):raise ValueError('AWAY survival support exhausted')
        v=list(vecs[idx]);v[0]-=elapsed_units
        if v[0]<=0:raise ValueError('nonpositive conditional tool residual')
        return tuple(v),len(times)-idx,idx,times[-1]

    def support_boundaries(self,prompt,ordinal,outputs):
        try:return self._away[bucket(prompt,ordinal)+(outputs,)][0]
        except KeyError:return ()

    def estimate(self,goal,profile,conformal=True):
        d=profile.decode_seconds;Q=self.q_cal if conformal else 0.
        if goal.ready is not None:
            c=goal.ready;mc,mt,tail,n=self.ready_base(c.prompt_tokens,c.ordinal,c.emitted)
            cur=mc+Q*scale(mc);total=mt+Q*scale(mt)
            # The old total intrinsic budget is unchanged. Only its AWAY part
            # moves to timers; all conformal padding stays on compute phases.
            phases=[['compute',cur]]
            extra=max(0.,total-cur-math.fsum(tail));future_compute=math.fsum(tail[1::2])
            for j in range(0,len(tail),2):
                phases.append(['away',tail[j]])
                phases.append(['compute',tail[j+1]+(extra*tail[j+1]/future_compute if future_compute else 0.)])
            support={'phase':'READY','bucket':bucket(c.prompt_tokens,c.ordinal),'survivors':n,'emitted':c.emitted}
            point_current=mc
        else:
            last=goal.history[-1];elapsed=goal.tool_elapsed/d
            tail,n,idx,hi=self.away_base(last.prompt_tokens,last.ordinal,last.outputs,elapsed)
            mt=max(.01,math.fsum(tail));total=mt+Q*scale(mt);cur=0.;point_current=0.
            extra=total-mt;future_compute=math.fsum(tail[1::2])
            if future_compute<=0:raise ValueError('AWAY future-compute support empty')
            phases=[]
            for j in range(0,len(tail),2):
                phases.append(['away',tail[j]])
                phases.append(['compute',tail[j+1]+extra*tail[j+1]/future_compute])
            support={'phase':'AWAY','bucket':bucket(last.prompt_tokens,last.ordinal),'survivors':n,
                     'elapsed_units':elapsed,'survival_interval':idx,'max_tool_units':hi,'observed_outputs':last.outputs}
        phases=[(kind,max(0.,v)*d) for kind,v in phases if v>0]
        # Floating budget correction belongs to compute, never to a timer.
        if phases:
            j=max(i for i,x in enumerate(phases) if x[0]=='compute')
            value=total*d-math.fsum(v for i,(_,v) in enumerate(phases) if i!=j)
            if value<-1e-9:raise ValueError('phase budget negative')
            phases[j]=('compute',max(0.,value))
        if abs(math.fsum(v for _,v in phases)-total*d)>1e-8:raise RuntimeError('intrinsic phase conservation')
        return {'point':mt*d,'work':total*d,'current':cur*d,'current_point':point_current*d,
          'mean_compute':(mt-math.fsum(tail[::2]))*d,'mean_away':math.fsum(tail[::2])*d,
          'phases':tuple(phases),'support':support,'conformal':conformal}

    def to_dict(self):return {'groups':self.groups,'q_cal':self.q_cal,'provenance':self.provenance,'version':self.version}
    @classmethod
    def from_dict(cls,d):
        groups=tuple((tuple(k),tuple((r[0],r[1],r[2],r[3],tuple(r[4])) for r in rs)) for k,rs in d['groups'])
        return cls(groups,d['q_cal'],tuple(d['provenance']),d['version'])

def fit(fit_rows,provenance):
    groups={}
    for r in fit_rows:groups.setdefault(bucket(r['prompt'],r['ordinal']),[]).append((r['seed'],r['outputs'],r['extra'],r['future'],r['tail']))
    return Predictor(tuple((k,tuple(v)) for k,v in sorted(groups.items())),0.,tuple(provenance))

def calibrate(base,cal_inputs,profile,save_rows=None):
    scores=[];counts={'READY_states':0,'AWAY_boundary_states':0,'READY_unsupported':0,'AWAY_unsupported_boundary_states':0,
                      'AWAY_unsupported_elapsed_units':0.,'AWAY_total_elapsed_units':0.};witnesses=[]
    for inp in cal_inputs:
        score=0.;witness=None
        for r in rows(inp,profile):
            for e in range(r['outputs']):
                try:mc,mt,_,n=base.ready_base(r['prompt'],r['ordinal'],e)
                except ValueError:counts['READY_unsupported']+=1;continue
                tc=r['outputs']-e+(r['extra'] if not e else 0.);tt=tc+r['future']
                z=max((tc-mc)/scale(mc),(tt-mt)/scale(mt));counts['READY_states']+=1
                if z>score:score=z;witness={'phase':'READY','program':r['program'],'ordinal':r['ordinal'],'emitted':e,'score':z}
            if r['tool']<=0:continue
            boundaries=base.support_boundaries(r['prompt'],r['ordinal'],r['outputs'])
            counts['AWAY_total_elapsed_units']+=r['tool']
            maxsupport=max(boundaries,default=0.)
            counts['AWAY_unsupported_elapsed_units']+=max(0.,r['tool']-maxsupport)
            points={0.,math.nextafter(r['tool'],-math.inf)}
            for x in set(boundaries):
                if x<r['tool']:
                    points.add(x);points.add(math.nextafter(x,-math.inf))
            for elapsed in sorted(x for x in points if 0<=x<r['tool']):
                try:v,n,idx,hi=base.away_base(r['prompt'],r['ordinal'],r['outputs'],elapsed)
                except ValueError:counts['AWAY_unsupported_boundary_states']+=1;continue
                mean=max(.01,math.fsum(v));truth=r['future']-elapsed;z=(truth-mean)/scale(mean)
                counts['AWAY_boundary_states']+=1
                if z>score:score=z;witness={'phase':'AWAY','program':r['program'],'ordinal':r['ordinal'],'elapsed_units':elapsed,'score':z}
                if save_rows is not None:save_rows({'seed':inp['seed'],'phase':'AWAY','program':r['program'],'ordinal':r['ordinal'],
                     'elapsed_units':elapsed,'mean_units':mean,'truth_units':truth,'score':z,'survivors':n})
        scores.append(score);witnesses.append({'seed':inp['seed'],'score':score,'witness':witness})
    rank=math.ceil((len(scores)+1)*.9)
    if rank>len(scores):raise ValueError('insufficient CAL workloads')
    Q=sorted(scores)[rank-1]
    final=Predictor(base.groups,Q,base.provenance)
    return final,{'alpha':.1,'unit':'whole workload','n':len(scores),'rank':rank,'q_cal':Q,'scores':scores,
        'enumeration':counts,'maxima':witnesses,
        'continuous_AWAY_certificate':'within each fixed strict-survival interval numerator truth-minus-mean is constant; positive standardized residual is nondecreasing until scale floor, so endpoints bound its supremum',
        'supported_phases_only':True,'no_JCT_coverage_claim':True}
