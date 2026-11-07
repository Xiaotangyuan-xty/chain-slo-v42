"""Bounded online decode-step cost proxy; no model execution or trace labels.

Only ordinary completed decode-only scheduler intervals are sampled. The proxy
is monotone in active members, not a claim of an identified GPU performance law.
"""
from collections import Counter, deque
import bisect
import math
import statistics


class StepCost:
    version = 'observed_monotone_decode_step_v1'

    def __init__(self, capacity=64):
        self.capacity = capacity
        self.samples = {}
        self.stats = Counter()

    def observe(self, members, seconds):
        if (not isinstance(members, int) or not 1 <= members <= 128
                or not math.isfinite(seconds) or not .001 <= seconds <= 1.):
            self.stats['rejected_samples'] += 1
            return False
        self.samples.setdefault(members, deque(maxlen=self.capacity)).append(seconds)
        self.stats['accepted_samples'] += 1
        return True

    def snapshot(self, fallback_unit, reference_members):
        fallback_unit = min(1., max(.001, fallback_unit))
        points = []
        high = .001
        for count, values in sorted(self.samples.items()):
            high = max(high, statistics.median(values))
            points.append((count, high))
        return CostSnapshot(points, fallback_unit, max(1, reference_members),
                            {n: len(v) for n, v in self.samples.items()})


class CostSnapshot:
    def __init__(self, points, fallback, reference, counts):
        self.points, self.fallback, self.reference = points, fallback, reference
        self.counts = counts

    def seconds(self, members):
        if members < 1:
            return 0.
        if not self.points:
            # Do not invent a speed-up for batches smaller than the current
            # observed reference. Above it, use conservative linear scaling.
            return self.fallback * max(1., members / self.reference)
        counts = [n for n, _ in self.points]
        i = bisect.bisect_left(counts, members)
        if i == 0:
            return self.points[0][1]
        if i == len(counts):
            n, seconds = self.points[-1]
            if len(self.points) >= 2:
                previous_n, previous_seconds = self.points[-2]
                slope = (seconds - previous_seconds) / (n - previous_n)
                intercept = seconds - slope * n
                if slope > 0 and intercept >= 0:
                    # Preserve an observed fixed-cost component. Assuming full
                    # proportional cost can trap selection at the largest
                    # previously observed batch instead of learning beyond it.
                    return intercept + slope * members
            return seconds * members / n
        lo, left = self.points[i - 1]; hi, right = self.points[i]
        return left + (right - left) * (members - lo) / (hi - lo)

    def evidenced_flat(self, before, after):
        return (after >= before > 0
                and self.counts.get(before, 0) >= 8
                and self.counts.get(after, 0) >= 8
                and self.seconds(after) <= self.seconds(before) + 1e-12)

    def describe(self):
        return {'version': StepCost.version, 'batch_seconds': self.points,
                'sample_counts': self.counts, 'fallback_unit': self.fallback,
                'reference_members': self.reference,
                'limitation': 'Online member-count proxy; not hardware profiling or a context-length latency oracle.'}
