"""Small conservative online executor-host service model, not CUDA kernel timing."""
from collections import Counter, deque
import math
import statistics


def bucket(n):
    return 1 << max(0, int(math.ceil(math.log2(max(1, n)))))


def upper(values):
    values = sorted(values)
    return values[min(len(values)-1, math.ceil(.9*len(values))-1)]


class ServiceModel:
    def __init__(self, token_budget=2048):
        self.samples = {}
        self.token_budget = token_budget
        self.rounds = deque(maxlen=64)
        self.stats = Counter()
        self.last_round = .05
        self._unit_cache = {}

    def set_token_budget(self, value):
        value=max(1,int(value))
        if value!=self.token_budget:
            self.token_budget=value;self._unit_cache.clear()

    def observe_round(self, seconds):
        if math.isfinite(seconds) and seconds > 0:
            self.last_round = seconds
            self._unit_cache.clear()
            self.rounds.append(seconds)
            self.stats['busy_rounds']+=1

    def observe_execution(self, values, seconds):
        duration = seconds
        if not math.isfinite(duration) or duration <= 0 or values.get('error'):
            return
        scheduler=max(0.,values.get('scheduler_seconds',0.))
        planning=max(0.,values.get('planning_seconds',0.))
        if math.isfinite(scheduler):self.stats['scheduler_seconds']+=scheduler
        if math.isfinite(planning):self.stats['planning_seconds']+=planning
        n = max(1, values.get('members', 1))
        key = (bucket(n), bucket(values.get('context_tokens', 0)/n/256),
               bucket(values.get('prefill_tokens', 0)/256) if values.get('prefill_tokens', 0) else 0,
               bool(values.get('transition')))
        self.samples.setdefault(key, deque(maxlen=64)).append(duration)
        self._unit_cache.clear()
        self.stats['samples'] += 1
        self.stats['transition_samples'] += bool(key[3])
        # A bounded feature grid: batch<=128, context<=8192 and chunk<=2048.
        if len(self.samples) > 512:
            self.samples.pop(next(iter(self.samples)))

    def unit(self, members, context=0, prefill=0, transition=False):
        n=max(1,members)
        key=(bucket(n),bucket(context/n/256),bucket(prefill/256) if prefill else 0,bool(transition))
        if key not in self._unit_cache:
            self._unit_cache[key]=self._unit(members,context,prefill,transition)
        return self._unit_cache[key]

    def _unit(self, members, context=0, prefill=0, transition=False):
        n = max(1, members)
        key = (bucket(n), bucket(context/n/256), bucket(prefill/256) if prefill else 0, bool(transition))
        exact = self.samples.get(key)
        if exact:
            return max(exact[-1], upper(exact))
        compatible = [(k, v) for k, v in self.samples.items() if k[2] == key[2] and k[3] == key[3]]
        if compatible:
            k, values = min(compatible, key=lambda kv: sum(abs(math.log2(max(1,a)/max(1,b))) for a,b in zip(kv[0][:3],key[:3])))
            scale = max(1., key[0]/k[0]) * max(1., key[1]/k[1])
            self.stats['scaled_estimates'] += 1
            return max(values[-1], upper(values)) * scale
        # No upper clamp. A real 0.5s round remains at least 0.5s until new evidence.
        fallback = max(self.last_round, upper(self.rounds)) if self.rounds else .05
        self.stats['fallback_estimates'] += 1
        return fallback * max(1., math.ceil(prefill/self.token_budget))

    def recovery(self, member):
        if member.get('resident'):
            return 0.
        context = member['prompt'] + member['generated']
        if member.get('swapped'):
            return self.unit(1, context, 0, True)
        prefill = member.get('prefill_tokens', member['prompt'])
        return math.ceil(max(0,prefill)/self.token_budget) * self.unit(1, context, min(prefill,self.token_budget), False)

    def describe(self):
        return {'token_budget':self.token_budget,'stats':dict(self.stats), 'last_round_seconds':self.last_round,
                'bins':[{'key':list(k),'n':len(v),'upper_seconds':upper(v),'last_seconds':v[-1]} for k,v in sorted(self.samples.items())],
                'timing':'executor host boundaries only; scheduler/planning and busy rounds are separate diagnostics'}


def member_blocks(m, block_size=16, horizon=50):
    return math.ceil((m['prompt']+m.get('generated',0)+min(max(0,m.get('remaining',m['max_tokens']-m.get('generated',0))),horizon))/block_size)


def waves(tasks, capacity, slots, block_size, first=None):
    """Fixed non-overlapping waves; first may be the actual selected members.

    Remaining members wait for the whole first wave to finish. This is a
    conservative execution assumption, never an all-members-parallel shortcut.
    """
    pending=list(tasks)
    result=[]
    if first is not None:
        head=[t for t in pending if t.get('request') in first]
        pending=[t for t in pending if t.get('request') not in first]
        if head:
            if len(head)>slots or sum(member_blocks(t,block_size) for t in head)>capacity:
                return None
            result.append(head)
    while pending:
        wave=[];used=0
        for task in pending:
            charge=member_blocks(task,block_size)
            if len(wave)<slots and used+charge<=capacity:
                wave.append(task);used+=charge
        if not wave:
            return None
        result.append(wave)
        ids={id(t) for t in wave};pending=[t for t in pending if id(t) not in ids]
    return result


def stage_seconds(tasks, service, capacity, slots, block_size, token_budget, first=None, background=()):
    reserved=sum(member_blocks(t,block_size) for t in background)
    available=max(0,capacity-reserved); available_slots=max(0,slots-len(background))
    schedule=waves(tasks, available, available_slots, block_size, first)
    if schedule is None:
        return math.inf, 0
    elapsed=0.
    for wave in schedule:
        batch=list(wave)+list(background)
        context=sum(t['prompt']+t.get('generated',0)+t.get('remaining',0) for t in batch)
        # Charge actual integer prefill chunks, including recompute context.
        prefill=sum(t.get('prefill_tokens',0) for t in wave)
        prefill_rounds=math.ceil(prefill/max(1,token_budget))
        elapsed += prefill_rounds*service.unit(len(batch),context,min(prefill,token_budget),False)
        elapsed += max((t.get('remaining',0) for t in wave), default=0)*service.unit(len(batch),context,0,False)
        if any(t.get('swapped') for t in wave):
            elapsed += service.unit(len(batch),context,0,True)
    return elapsed,len(schedule)
