"""Bounded executor-call wall-time estimator with explicit workload strata.

This measures host executor boundaries (including transfers/asynchronous wait),
not CUDA kernel latency. Planning and scheduler iteration timing are recorded
separately. Long, prefill, swap, preempted and ignored samples are retained in
separate strata; only malformed/nonpositive measurements are rejected.
"""
from collections import Counter, OrderedDict, deque
import math
import statistics


def bucket(value, quantum):
    return int(math.ceil(max(0, value) / quantum))


class StepCost:
    version = 'executor_boundary_context_prefill_transition_v2'

    def __init__(self, capacity=64, strata=256):
        self.capacity, self.max_strata = capacity, strata
        self.samples = OrderedDict(); self.stats = Counter()

    def observe(self, members, seconds):
        return self.observe_execution({'members': members, 'context_tokens': 0,
                                      'prefill_tokens': 0, 'transition': False}, seconds)

    def observe_execution(self, features, seconds):
        n = features['members']
        if not isinstance(n, int) or not 0 <= n <= 4096 or not math.isfinite(seconds) or seconds <= 0:
            self.stats['invalid_samples'] += 1; return False
        key = (n, bucket(features.get('context_tokens', 0)/max(1,n), 1024),
               bucket(features.get('prefill_tokens', 0), 256),
               bool(features.get('transition')), bool(features.get('preempted')),
               bool(features.get('ignored')))
        if key not in self.samples:
            if len(self.samples) >= self.max_strata:
                _, old = self.samples.popitem(last=False)
                self.stats['strata_evicted'] += 1; self.stats['samples_evicted'] += len(old)
            self.samples[key] = deque(maxlen=self.capacity)
        self.samples.move_to_end(key)
        if len(self.samples[key]) == self.capacity:self.stats['samples_overwritten'] += 1
        self.samples[key].append(seconds); self.stats['accepted_samples'] += 1
        self.stats['long_over1s'] += seconds > 1.
        self.stats['prefill_samples'] += features.get('prefill_tokens', 0) > 0
        self.stats['transition_samples'] += bool(features.get('transition'))
        self.stats['planned_samples'] += bool(features.get('planning_seconds', 0))
        self.stats['preempted_samples'] += bool(features.get('preempted'))
        self.stats['ignored_samples'] += bool(features.get('ignored'))
        return True

    def snapshot(self, fallback_unit, reference_members):
        fallback_unit = max(.001, float(fallback_unit))
        return CostSnapshot([(k, statistics.median(v), len(v)) for k,v in self.samples.items()],
                            fallback_unit, max(1, reference_members))


class CostSnapshot:
    def __init__(self,strata,fallback,reference):
        import bisect
        self.strata,self.fallback,self.reference=tuple(strata),fallback,reference
        self._cache={};self._nearest={};self.queries=0;self.cells={}
        # Pre-index immutable medians once. A query no longer scans all strata
        # for every action/scenario/width combination.
        for k,t,count in self.strata:
            if k[4]or k[5]:continue
            self.cells.setdefault((k[1],k[2],k[3]),[]).append((k[0],t,count))
        for cell,values in self.cells.items():
            high=.001;points=[]
            for n,t,count in sorted(values):high=max(high,t);points.append((n,high))
            self.cells[cell]=(tuple(n for n,_ in points),tuple(t for _,t in points))
        self._bisect=bisect.bisect_left

    def seconds(self,members,context_tokens=0.,prefill_tokens=0.,transition=False):
        if members<1 and not transition:return 0.
        key=(members,bucket(context_tokens/max(1,members),1024),bucket(prefill_tokens,256),bool(transition))
        if key in self._cache:return self._cache[key]
        self.queries+=1;query=key[1:]
        if query not in self._nearest:
            choices=[cell for cell in self.cells if cell[2]==bool(transition)and(cell[1]>0)==(prefill_tokens>0)]
            self._nearest[query]=min(choices,key=lambda c:(abs(c[0]-query[0])+abs(c[1]-query[1]),c))if choices else None
        cell=self._nearest[query]
        if cell is None:
            value=self.fallback*max(1.,members/self.reference)
            if prefill_tokens:value*=max(1.,prefill_tokens/256.)
            if transition:value*=1.+max(0,context_tokens)/2048.
        else:
            ns,ts=self.cells[cell];i=self._bisect(ns,members)
            if i==0:value=ts[0]
            elif i==len(ns):value=ts[-1]*max(1.,members/max(1,ns[-1]))
            else:value=ts[i-1]+(ts[i]-ts[i-1])*(members-ns[i-1])/(ns[i]-ns[i-1])
            value*=max(1.,query[0]/max(1,cell[0]),query[1]/max(1,cell[1]))
        self._cache[key]=max(.000001,value);return self._cache[key]

    def transition_seconds(self,affected_members,context_tokens=0.):
        return self.seconds(affected_members,context_tokens,0.,True)if affected_members else 0.

    def describe(self,limit=16):
        return {'version':StepCost.version,'index':'context_prefill_cell_then_width_bisect_v1',
                'strata':[{'members':k[0],'context_1024':k[1],'prefill_256':k[2],
                'transition':k[3],'preempted':k[4],'ignored':k[5],'seconds':t,'samples':c}for k,t,c in self.strata[-limit:]],
                'total_strata':len(self.strata),'logged_strata_limit':limit,
                'fallback_unit':self.fallback,'reference_members':self.reference,'queries_materialized':self.queries,
                'limitation':'Executor host wall time, sparse context/prefill strata; not CUDA kernel latency.'}
