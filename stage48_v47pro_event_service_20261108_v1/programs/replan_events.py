"""Event/deadline replanning; 50 ticks is only a final fallback."""
from collections import Counter


class ReplanEvents:
    def __init__(self):
        self.signature=None;self.last_tick=-50;self.next_due=None;self.invalidations=set()
        self.counts=Counter();self.last_reasons=[];self.last_free_capacity=None

    def invalidate(self, reason):
        self.invalidations.add(reason)

    def reasons(self, signature, tick, now):
        reasons=set(self.invalidations)
        free=signature.get('free_capacity')
        if free is not None:
            if self.last_free_capacity is not None and any(a>b for a,b in zip(free,self.last_free_capacity)):
                reasons.add('capacity_release')
            self.last_free_capacity=free
        if self.signature is None:
            reasons.add('initial_or_mode')
        else:
            for key,value in signature.items():
                if key=='free_capacity':continue
                if value!=self.signature.get(key):reasons.add(key)
        if self.next_due is not None and now>=self.next_due:reasons.add('deadline_wait_or_quantum')
        if tick-self.last_tick>=50:reasons.add('50_tick_fallback')
        return sorted(reasons)

    def commit(self,signature,tick,now,deadlines,reasons):
        self.signature=signature;self.last_tick=tick;self.invalidations.clear()
        self.next_due=min((t for t in deadlines if t>now+1e-6),default=None)
        self.last_reasons=reasons
        self.counts.update(reasons)
