"""Event/deadline replanning; 50 ticks is only a final fallback."""
from collections import Counter


class ReplanEvents:
    def __init__(self):
        self.signature=None;self.last_tick=-50;self.next_due=None;self.invalidations=set()
        self.counts=Counter();self.last_reasons=[];self.last_free_capacity=None
        self.last_plan_monotonic=None
        self.urgent={'initial_or_mode','mode_epoch','membership_stage','deadline_wait_or_quantum',
                     'physical_preemption','capacity_release','configuration','50_tick_fallback'}

    def invalidate(self, reason):
        self.invalidations.add(reason)

    def reasons(self, signature, tick, now, coalesce_seconds=0.):
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
        if (reasons and not self.urgent.intersection(reasons) and self.last_plan_monotonic is not None
                and now-self.last_plan_monotonic<coalesce_seconds):
            self.counts['coalesced_replans']+=1
            return []
        return sorted(reasons)

    def commit(self,signature,tick,now,deadlines,reasons,waiting=False,started=None):
        self.signature=signature;self.last_tick=tick;self.invalidations.clear()
        crossed=started is not None and any(started<t<=now for t in deadlines)
        if waiting:deadlines=list(deadlines)+[now+.5]
        self.next_due=now if crossed else min((t for t in deadlines if t>now+1e-6),default=None)
        self.last_reasons=reasons
        self.last_plan_monotonic=now
        self.counts.update(reasons)
