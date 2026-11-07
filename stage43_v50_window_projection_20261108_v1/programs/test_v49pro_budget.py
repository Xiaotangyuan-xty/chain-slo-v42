"""CPU synthesis only: bounded materialization and observed planner wall time."""
from collections import Counter
import json
import statistics
import time
import unittest
from mixed_control import plan
from stage_scenarios import Scenarios,Scenario,Stage
from step_cost import StepCost


class Budget(unittest.TestCase):
    def test_64_and_256_candidates_have_bounded_expensive_pool(self):
        results=[]
        for count in [64,256]:
            actions=[];members=[];calls=[]
            for i in range(count):
                aid=str(i)
                bundle=Scenarios([Scenario((Stage(1,50+j,0,256),Stage(2,20+j,256,512)),2000+j,j)for j in range(8)],10)
                def factory(i=i,bundle=bundle):calls.append(i);return bundle
                actions.append({'id':aid,'kind':'compound','members':[aid],'arrival':float(i),
                                'native_priority':(float(i),),'scenario_factory':factory,
                                'cheap_credit':2000.,'cheap_work':100.})
                members.append({'request':aid,'prompt':128,'generated':0,'max_tokens':1024,'prefill_tokens':0})
            durations=[];reasons=Counter();max_loaded=0
            for _ in range(15):
                calls.clear();cost=StepCost().snapshot(.02,4);start=time.perf_counter()
                selected,detail=plan(actions,members,set(map(str,range(4))),10000,8,.02,costs=cost)
                durations.append(time.perf_counter()-start)
                max_loaded=max(max_loaded,len(calls));reasons[detail.get('fallback_reason','intervention')]+=1
                self.assertLessEqual(len(calls),12)
            results.append({'candidates':count,'runs':len(durations),'mean_ms':statistics.fmean(durations)*1000,
                            'p95_ms':sorted(durations)[int(.95*(len(durations)-1))]*1000,'max_ms':max(durations)*1000,
                            'max_materialized_actions':max_loaded,'reasons':reasons})
        print(json.dumps({'CPU_SYNTHETIC_PLANNER_BUDGET':results,'GPU_performance_claim':False}))


if __name__=='__main__':unittest.main(verbosity=2)
