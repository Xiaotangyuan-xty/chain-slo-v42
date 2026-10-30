"""Primitive, immutable online state; evaluator futures never enter these types."""
from dataclasses import dataclass,asdict
from contracts import CallView,CacheView

@dataclass(frozen=True)
class ObservedCall:
    ordinal:int
    prompt_tokens:int
    outputs:int
    finished:float
    service:float|None=None
    wait:float|None=None

@dataclass(frozen=True)
class GoalView:
    program:str
    arrival:float
    deadline:float
    state:str
    history:tuple
    ready:CallView|None=None
    tool_entered:float|None=None
    tool_elapsed:float|None=None
    produced_prefix:int=0

    def __post_init__(self):
        if self.state not in ('READY','RUNNING','AWAY'):raise ValueError('unfinished arrived goals only')
        if self.state=='AWAY':
            if self.ready is not None or not self.history or self.tool_entered is None or self.tool_elapsed is None or self.tool_elapsed<0:
                raise ValueError('AWAY may contain only observed past and tool-entry/elapsed')
        elif self.ready is None:raise ValueError('READY goal requires a released call')

@dataclass(frozen=True)
class View:
    now:float
    goals:tuple
    cache:CacheView

    @property
    def calls(self):return tuple(g.ready for g in self.goals if g.ready is not None)
    def to_dict(self):return asdict(self)

def from_dict(d):
    from contracts import CacheEntry
    if set(d)!= {'now','goals','cache'}:raise TypeError('non-whitelisted online view field')
    cache=CacheView(d['cache']['gpu_capacity'],d['cache']['host_capacity'],tuple(CacheEntry(**x) for x in d['cache']['entries']))
    goals=[]
    for x in d['goals']:
        x=dict(x);x['history']=tuple(ObservedCall(**r) for r in x['history'])
        x['ready']=CallView(**x['ready']) if x.get('ready') else None
        goals.append(GoalView(**x))
    return View(d['now'],tuple(goals),cache)

def make_view(now,client_metadata,states,histories,ready_calls,cache,tool_entered,prefixes):
    """Whitelisting boundary: evaluator dictionaries may have hidden fields."""
    goals=[]
    for pid,state in states.items():
        if state not in ('READY','RUNNING','AWAY'):continue
        public=client_metadata[pid];entered=tool_entered.get(pid)
        goals.append(GoalView(pid,public['arrival'],public['deadline'],state,tuple(histories.get(pid,())),
            ready_calls.get(pid),entered if state=='AWAY' else None,
            max(0.,now-entered) if state=='AWAY' else None,prefixes.get(pid,0)))
    return View(now,tuple(goals),cache)
