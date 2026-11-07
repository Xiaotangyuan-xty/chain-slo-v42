"""Open-loop public-state trace replay; no GPU/performance oracle claims."""
from pathlib import Path
import json,statistics,time,unittest,math
from collections import Counter
from chain_forecast import StageForecaster
from stage_scenarios import forecast_scenarios
from step_cost import CostSnapshot
from window_control import plan

P=Path(__file__).resolve().parents[1]


class TraceReplay(unittest.TestCase):
    def test_real_observed_snapshots_complete_evaluations_and_can_intervene(self):
        packet=json.loads((P/'assets/TRACE_PUBLIC_SNAPSHOTS.json').read_text())
        model=StageForecaster.load(P/'assets/CHAIN_MODEL.json')
        tot=json.loads((P/'assets/TOT_CALIBRATION.json').read_text())
        tot_model=StageForecaster(tot['models']['llama8']['model'])
        samples=[];counts=Counter();per_gpu={}
        for fixture in packet['SNAPSHOTS']:
            actions=fixture['actions'];members=fixture['members'];cost=CostSnapshot(fixture['cost_strata'],fixture['unit'],fixture['cost_reference'])
            for a in actions:
                if a['kind']!='compound':continue
                meta=a.pop('meta');meta['observed']={int(s):{int(i):z for i,z in ms.items()}for s,ms in meta['observed'].items()};meta['neighbor_cache']={}
                mod=tot_model if a['domain']=='tot'else model
                def factory(a=a,meta=meta,mod=mod):return forecast_scenarios(mod,meta,a['stage'],a['views'],a['slack'],max_scenarios=16)
                a['scenario_factory']=factory
            start=time.perf_counter()
            selected,detail=plan(actions,members,set(fixture['running']),fixture['capacity'],fixture['slots'],fixture['unit'],costs=cost,swap_capacity_blocks=max(0,512-sum(math.ceil((m['prompt']+m['generated'])/16)for m in members if m.get('state')=='swapped')))
            duration=time.perf_counter()-start;samples.append(duration)
            counts['snapshots']+=1;counts['with_complete_comparison']+=detail['complete_sets']>=2;counts['interventions']+=selected is not None;counts['complete_sets']+=detail['complete_sets']
            counts[detail.get('fallback_reason','intervention')]+=1
            per_gpu.setdefault(str(fixture['GPU']),Counter()).update({'snapshots':1,'complete_comparison':detail['complete_sets']>=2,'interventions':selected is not None})
            if selected is not None:
                self.assertTrue(selected<={m['request']for m in members})
                self.assertGreater(detail['paired']['delta_credit'],detail['required_credit_gain'])
        report={'snapshots':dict(counts),'per_gpu':per_gpu,'mean_ms':statistics.fmean(samples)*1000,
                'p95_ms':sorted(samples)[int(.95*(len(samples)-1))]*1000,'max_ms':max(samples)*1000,
                'source_membership_mismatches':sum(s['membership_mismatch']for s in packet['SNAPSHOTS']),
                'source_priority_is_public_corrected_budget_proxy':True,'GPU_goodput_prediction_claim':False}
        print(json.dumps({'V50_PUBLIC_TRACE_REPLAY':report}))
        self.assertEqual(counts['snapshots'],64)
        self.assertGreaterEqual(counts['with_complete_comparison'],48)
        self.assertGreater(counts['interventions'],0)
        self.assertEqual(report['source_membership_mismatches'],0)


if __name__=='__main__':unittest.main(verbosity=2)
