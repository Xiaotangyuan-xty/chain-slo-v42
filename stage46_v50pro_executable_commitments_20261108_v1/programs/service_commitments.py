"""Executable bounded service and restoration obligations, using public state.

A quantum ends after 50 newly generated tokens OR 1.5 seconds. This time bound
is explicit, not a claim that every request always obtains 50 tokens. Displaced
physical members receive a non-extendible 1.5-second restoration deadline.
Root cooldown applies across siblings. No native baseline priority is consumed.
"""
from collections import Counter

class Commitments:
    def __init__(self, quantum=50, max_seconds=1.5, cooldown=4.):
        self.quantum,self.max_seconds,self.cooldown=quantum,max_seconds,cooldown
        self.serving={};self.deferred={};self.restoring={};self.last_evicted={};self.stats=Counter();self.sink=lambda e:None
    def event(self,kind,now,**kw):
        self.stats[kind]+=1;self.sink({'kind':kind,'monotonic':now,**kw})
    def observe(self,views,now):
        live=set(views)
        for rid in list(self.serving):
            if rid not in live:self.serving.pop(rid)
        for rid in list(self.deferred):
            if rid not in live:self.deferred.pop(rid)
        for rid in list(self.restoring):
            if rid not in live:self.restoring.pop(rid);continue
            start,due,dispatch=self.restoring[rid]
            if views[rid]['generated']>start:
                self.restoring.pop(rid);self.event('restoration_service_confirmed',now,request=rid,seconds_after_due=now-due,seconds_after_dispatch=now-dispatch)
        for root in list(self.last_evicted):
            if now-self.last_evicted[root]>self.cooldown+self.max_seconds:self.last_evicted.pop(root)
        for rid,v in views.items():
            if v['running']and rid not in self.serving:
                self.serving[rid]=(v['generated'],now)
                self.event('service_begin',now,request=rid,root=v['root'],generated=v['generated'])
    def protected(self,views,now):
        return {rid for rid,(start,begin)in self.serving.items()if rid in views and views[rid]['running']and not views[rid]['expired']and views[rid]['generated']-start<self.quantum and now-begin<self.max_seconds}
    def due(self,views,now):return {r for r,d in self.deferred.items()if r in views and now>=d}
    def restoration_selection(self,mandatory,views,now,fits):
        running={r for r,v in views.items()if v['running']}
        selected=set(mandatory)|self.protected(views,now)
        if not fits(selected):selected=set(mandatory)
        if not fits(selected):
            self.event('restoration_capacity_failure',now,requests=sorted(mandatory));raise RuntimeError('Restoration obligation exceeds physical capacity')
        # Restoration must not evict unrelated residents merely because their
        # quantum has ended. Keep each physically feasible incumbent first.
        for r in sorted(running-selected,key=lambda r:(views[r]['expired'],r)):
            if fits(selected|{r}):selected.add(r)
        collateral=running-selected
        if collateral:
            for r in collateral:
                self.deferred.setdefault(r,now+self.max_seconds);self.last_evicted[views[r]['root']]=now;self.serving.pop(r,None)
            self.event('restoration_capacity_eviction',now,requests=sorted(collateral),restore_by=now+self.max_seconds)
        return selected
    def resolve(self,proposal,views,now,fits=lambda selected:True,coexistence_allowed=None):
        self.observe(views,now);running={r for r,v in views.items()if v['running']};due=self.due(views,now)
        if self.restoring:
            selected=self.restoration_selection(set(self.restoring),views,now,fits)
            return selected,{'reason':'awaiting_restoration_service','requests':sorted(self.restoring)}
        if due:
            # A newly proposed plan cannot extend a prior deadline.
            selected=self.restoration_selection(due,views,now,fits)
            for r in sorted(due):
                due_time=self.deferred.pop(r);delay=now-due_time;self.serving.pop(r,None);self.restoring[r]=(views[r]['generated'],due_time,now)
                self.event('restoration_enforced',now,request=r,overdue_seconds=delay)
            return selected,{'reason':'restoration_obligation','due':sorted(due)}
        if proposal is None:
            # Even the fallback cannot circumvent a service obligation.
            proposal=set(running)if self.protected(views,now)or self.deferred or self.restoring else None
            if proposal is None:return None,{'reason':'no_obligation'}
        selected=set(proposal)&set(views)
        protected=self.protected(views,now)
        cooling={r for r in running if not views[r]['expired']and now-self.last_evicted.get(views[r]['root'],-float('inf'))<self.cooldown}
        if (protected|cooling)-selected:
            # Physical no-change is always preferable to silently overflowing.
            merged=selected|protected|cooling
            if fits(merged)and(coexistence_allowed is None or coexistence_allowed(merged)):
                self.event('protected_nonpreemptive_admission',now,requests=sorted(merged-running))
                return merged,{'reason':'protected_nonpreemptive_admission'}
            self.event('eviction_blocked',now,requests=sorted((protected|cooling)-selected))
            return set(running),{'reason':'service_quantum_or_root_cooldown'}
        evicted=running-selected
        if (self.deferred or self.restoring)and evicted:
            self.event('nested_eviction_blocked',now,requests=sorted(evicted))
            return set(running),{'reason':'outstanding_restoration'}
        if evicted:
            for rid in sorted(evicted):
                self.deferred.setdefault(rid,now+self.max_seconds)
                self.last_evicted[views[rid]['root']]=now;self.serving.pop(rid,None)
            self.event('eviction_committed',now,requests=sorted(evicted),restore_by=now+self.max_seconds)
        return selected,{'reason':'accepted','evicted':sorted(evicted)}
