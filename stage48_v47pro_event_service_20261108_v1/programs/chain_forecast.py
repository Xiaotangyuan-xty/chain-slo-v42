"""v47 empirical root reward, repriced for selected capacity/prefill waves."""
from collections import Counter, OrderedDict
import json
import math
from pathlib import Path
import statistics
import time


class StageForecaster:
    def __init__(self, model, future_cache_capacity=8192):
        self.model = model
        self.records = model['records']
        self.k = model['neighbors']
        self.prefill_unit = model['prefill_tokens_per_decode_unit']
        self.gap_units = model['interstage_gap_decode_units']
        self.guard_units = model['forecast_guard_decode_units']
        self.factor = model['service_safety_factor']
        self.features = {r['id']: [(math.log1p(sum(z['inputs'])), math.log1p(max(z['outputs'])),
                                    len(z['outputs'])) for z in r['stages']] for r in self.records}
        self.future_cache = OrderedDict()
        self.future_cache_capacity = future_cache_capacity
        self.stats = Counter()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def neighbors(self, meta, stage, views):
        observed, widths = meta['observed'], meta['widths']
        current = {i: z['declared_input'] for i, z in observed.get(stage, {}).items()}
        current.update({v['member']: v['declared_input'] for v in views})
        signature = (tuple(widths), meta['stages'], tuple(sorted(current.items())),
                     tuple((sid, tuple((i, z['declared_input'], z['output'])
                           for i, z in sorted(members.items())))
                           for sid, members in sorted(observed.items()) if sid < stage))
        cached = meta['neighbor_cache'].get(stage)
        if cached is not None and cached[0] == signature:
            self.stats['neighbor_cache_hits'] += 1
            return cached[1]
        self.stats['neighbor_cache_misses'] += 1
        current_log = math.log1p(sum(current.values()) * widths[stage] / max(1, len(current)))
        pool = [r for r in self.records if len(r['stages']) == meta['stages']]
        if not pool:
            pool = [r for r in self.records if len(r['stages']) >= meta['stages']]
        if not pool:
            return []

        def distance(row):
            fs = self.features[row['id']]
            score = abs(fs[stage][0] - current_log) + .35 * sum(abs(a - b[2]) for a, b in zip(widths, fs))
            for sid, members in observed.items():
                if sid < stage and sid < len(fs):
                    score += .5 * abs(math.log1p(sum(x['declared_input'] for x in members.values())) - fs[sid][0])
                    score += .5 * abs(math.log1p(max(x['output'] for x in members.values())) - fs[sid][1])
            return score, row['id']

        result = sorted(pool, key=distance)[:self.k]
        meta['neighbor_cache'][stage] = (signature, result)
        return result

    @staticmethod
    def member_target(row, stage, member, width, cap):
        outputs = row['stages'][stage]['outputs']
        index = min(len(outputs) - 1, member * len(outputs) // max(1, width))
        return min(cap, outputs[index])

    def future_components(self, row, stage, widths, cap):
        key = (row['id'], stage, tuple(widths), cap)
        if key in self.future_cache:
            self.stats['future_cache_hits'] += 1
            self.future_cache.move_to_end(key)
            return self.future_cache[key]
        self.stats['future_cache_misses'] += 1
        work = critical = reward = 0.
        for sid in range(stage + 1, len(widths)):
            z = row['stages'][sid]
            outputs = [min(cap, x) for x in z['outputs']]
            ratio = widths[sid] / max(1, len(outputs))
            work += (sum(outputs) + sum(z['inputs']) / self.prefill_unit) * ratio
            critical += max(outputs) + max(z['inputs']) / self.prefill_unit + self.gap_units
            reward += (sum(z['inputs']) + 8 * sum(outputs)) * ratio
        result = (work, critical, reward)
        self.future_cache[key] = result
        if len(self.future_cache) > self.future_cache_capacity:
            self.future_cache.popitem(last=False)
        return result

    @staticmethod
    def zero(reason, missing=0, supported=False):
        return {'supported': supported, 'probability': 0., 'remaining_work': None,
                'critical_mean_seconds': None, 'current_remaining_tokens': None,
                'score': 0., 'expected_credited_tokens': 0., 'mean_chain_tokens': None,
                'donors': 0, 'survival_backoff': False,
                'estimated_unsubmitted_members': missing, 'reason': reason}

    def estimate(self, meta, stage, views, slack, token_seconds=None, *, service=None,
                 capacity=1000000, slots=32, block_size=16, token_budget=2048):
        from service_model import ServiceModel
        start=time.perf_counter();self.stats['estimate_calls']+=1
        if service is None:
            service=ServiceModel()
            if token_seconds is not None:service.observe_round(token_seconds)
        try:
            if slack<=0:return self.zero('whole_chain_deadline_expired',supported=True)
            width=meta['widths'][stage];done=meta['observed'].get(stage,{})
            live={v['member'] for v in views}
            missing=[i for i in range(width) if i not in live and i not in done]
            donors=self.neighbors(meta,stage,views)
            if not donors:return self.zero('no_full_depth_training_support',len(missing))
            cap=meta.get('future_max_tokens',1024)
            survivors=[r for r in donors if all(self.member_target(r,stage,v['member'],width,v['max_tokens'])>v['generated'] for v in views)]
            selected=survivors or donors
            prefix=sum(z.get('reward_input',z['declared_input'])+8*z.get('reward_output',z['output'])
                       for members in meta['observed'].values() for z in members.values())
            scenarios=[]
            for row in selected:
                current=[];reward=prefix
                for v in views:
                    n=max(0,self.member_target(row,stage,v['member'],width,v['max_tokens'])-v['generated'])
                    if not survivors:n=min(max(0,v['max_tokens']-v['generated']),max(50,n))
                    current.append({**v,'remaining':n})
                    reward+=v.get('reward_input',v['prompt'])+8*(v['generated']+n)
                for member in missing:
                    z=row['stages'][stage];idx=min(len(z['outputs'])-1,member*len(z['outputs'])//max(1,width))
                    n=min(cap,z['outputs'][idx]);inp=z['inputs'][min(idx,len(z['inputs'])-1)]
                    current.append({'request':None,'prompt':inp,'generated':0,'prefill_tokens':inp,'remaining':n,'max_tokens':cap})
                    reward+=inp+8*n
                stages=[current]
                for sid in range(stage+1,meta['stages']):
                    z=row['stages'][sid];future=[]
                    for member in range(meta['widths'][sid]):
                        idx=min(len(z['outputs'])-1,member*len(z['outputs'])//max(1,meta['widths'][sid]))
                        n=min(cap,z['outputs'][idx]);inp=z['inputs'][min(idx,len(z['inputs'])-1)]
                        future.append({'request':None,'prompt':inp,'generated':0,'prefill_tokens':inp,'remaining':n,'max_tokens':cap})
                        reward+=inp+8*n
                    stages.append(future)
                scenarios.append({'stages':stages,'reward':reward})
            state={'scenarios':scenarios,'slack':slack,'service':service,'capacity':capacity,
                   'slots':slots,'block_size':block_size,'token_budget':token_budget,
                   'supported':True,'survival_backoff':not bool(survivors),'estimated_unsubmitted_members':len(missing)}
            return {**self._value(state), '_forecast':state}
        finally:self.stats['estimate_seconds']+=time.perf_counter()-start

    def _value(self,state,first=None,background=()):
        from service_model import stage_seconds
        service=state['service'];critical=[];works=[];rewards=[];wave_counts=[]
        for scenario in state['scenarios']:
            elapsed=0.;work=0.;count=0
            for sid,tasks in enumerate(scenario['stages']):
                seconds,nwaves=stage_seconds(tasks,service,state['capacity'],state['slots'],state['block_size'],state['token_budget'],first if sid==0 else None,background)
                elapsed+=seconds;count+=nwaves
                if sid:elapsed+=self.gap_units*service.unit(1)
                for task in tasks:
                    context=task['prompt']+task['generated']+task['remaining']
                    work+=task['remaining']*service.unit(1,context)
                    prefill=task.get('prefill_tokens',0)
                    work+=math.ceil(prefill/state['token_budget'])*service.unit(1,context,min(prefill,state['token_budget']))
                    if task.get('swapped'):work+=service.unit(1,context,0,True)
            critical.append(self.factor*elapsed+self.guard_units*service.unit(1))
            works.append(work);rewards.append(scenario['reward']);wave_counts.append(count)
        timely=[c<=state['slack'] for c in critical]
        expected=statistics.fmean(r if ok else 0. for r,ok in zip(rewards,timely))
        work=max(.001,statistics.fmean(works))
        finite=all(math.isfinite(c) for c in critical)
        return {'supported':True,'probability':sum(timely)/len(timely),'remaining_work':work,
                'critical_mean_seconds':statistics.fmean(critical) if finite else None,
                'score':expected/work,'expected_credited_tokens':expected,'mean_chain_tokens':statistics.fmean(rewards),
                'donors':len(critical),'survival_backoff':state['survival_backoff'],
                'estimated_unsubmitted_members':state['estimated_unsubmitted_members'],
                'stage_waves_mean':statistics.fmean(wave_counts),'wave_model':'selected first, then fixed capacity/token-budget waves'}

    def revalue(self,goal,first,background=()):
        if '_forecast' not in goal:return goal
        self.stats['selected_wave_revalues']+=1
        return {**goal,**self._value(goal['_forecast'],first,background)}
