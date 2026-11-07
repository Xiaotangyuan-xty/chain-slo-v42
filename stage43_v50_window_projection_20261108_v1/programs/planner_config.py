"""Single immutable source for the published candidate planning contract."""
from dataclasses import dataclass,asdict


@dataclass(frozen=True)
class PlannerConfig:
    max_seconds:float=.010
    final_margin_seconds:float=.002
    max_evaluations:int=20
    scenario_limit:int=16
    max_evicted_members:int=8
    challenger_count:int=6
    min_gain:float=8.
    relative_gain:float=.005
    horizon:int=50


CONFIG=PlannerConfig()
def contract():return asdict(CONFIG)
