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
    def __init__(self, strata, fallback, reference):
        self.strata, self.fallback, self.reference = tuple(strata), fallback, reference
        self._cache = {}; self.queries = 0

    def seconds(self, members, context_tokens=0., prefill_tokens=0., transition=False):
        if members < 1 and not transition:return 0.
        # Query only widths encountered by the rollout, never all actions x widths.
        key = (members, bucket(context_tokens/max(1,members),1024),bucket(prefill_tokens,256),bool(transition))
        if key in self._cache:return self._cache[key]
        self.queries += 1
        matching = [(k,t,c)for k,t,c in self.strata if k[3] == bool(transition)
                    and (k[2] > 0) == (prefill_tokens > 0) and not k[4] and not k[5]]
        if matching:
            distance = lambda z: (abs(z[0][1]-key[1]) + abs(z[0][2]-key[2]) + abs(z[0][0]-members)/max(1,members), z[0])
            # Match context/prefill before building a width envelope; samples from
            # a long-context/drain batch cannot permanently poison every width.
            nearest = min(matching,key=distance)[0]
            points = sorted((k[0],t,c)for k,t,c in matching if k[1:3] == nearest[1:3])
            high = .001; monotone=[]
            for n,t,c in points:high=max(high,t);monotone.append((n,high))
            left = max((x for x in monotone if x[0] <= members),default=monotone[0])
            right = min((x for x in monotone if x[0] >= members),default=monotone[-1])
            if left[0] != right[0]:value=left[1]+(right[1]-left[1])*(members-left[0])/(right[0]-left[0])
            else:value=left[1]*max(1.,members/max(1,left[0]))
            # Unseen larger contexts/prefills get conservative explicit scaling.
            value *= max(1.,key[1]/max(1,nearest[1]),key[2]/max(1,nearest[2]))
        else:
            value = self.fallback * max(1.,members/self.reference)
            if prefill_tokens:value *= max(1.,prefill_tokens/256.)
            if transition:value *= 1. + max(0,context_tokens)/2048.
        self._cache[key] = max(.000001,value)
        return self._cache[key]

    def transition_seconds(self, affected_members, context_tokens=0.):
        return self.seconds(affected_members,context_tokens,0.,True) if affected_members else 0.

    def describe(self, limit=16):
        return {'version':StepCost.version,'strata':[{'members':k[0],'context_1024':k[1],'prefill_256':k[2],
                'transition':k[3],'preempted':k[4],'ignored':k[5],'seconds':t,'samples':c}for k,t,c in self.strata[-limit:]],
                'total_strata':len(self.strata),'logged_strata_limit':limit,
                'fallback_unit':self.fallback,'reference_members':self.reference,
                'queries_materialized':self.queries,
                'limitation':'Executor host wall time includes execution/transfer/wait; sparse context/prefill strata use conservative extrapolation. Not CUDA kernel latency.'}
