"""Remaining chain-work estimates from independent FIT graphs and observed prefixes."""
import json
import math
from pathlib import Path
import statistics

class StageForecaster:
    def __init__(self,model):
        self.model=model;self.records=model['records'];self.k=model['neighbors']
        self.prefill_unit=model['prefill_tokens_per_decode_unit'];self.gap_units=model['interstage_gap_decode_units']
        self.guard_units=model['forecast_guard_decode_units'];self.factor=model['service_safety_factor']
        self.features={r['id']:[(math.log1p(sum(z['inputs'])),math.log1p(max(z['outputs'])),len(z['outputs'])) for z in r['stages']] for r in self.records}

    @classmethod
    def load(cls,path):return cls(json.loads(Path(path).read_text()))

    def neighbors(self,meta,stage,views):
        observed=meta['observed'];widths=meta['widths']
        current={i:z['declared_input'] for i,z in observed.get(stage,{}).items()}
        current.update({v['member']:v['declared_input'] for v in views})
        # Compound members can arrive after the first planning frame. Reuse only
        # when the actually visible declared inputs and prior observations agree.
        signature=(tuple(sorted(current.items())),tuple((sid,tuple((i,z['declared_input'],z['output'])
                    for i,z in sorted(members.items()))) for sid,members in sorted(observed.items()) if sid<stage))
        cached=meta['neighbor_cache'].get(stage)
        if cached is not None and cached[0]==signature:return cached[1]
        current_log=math.log1p(sum(current.values())*widths[stage]/max(1,len(current)))
        pool=[r for r in self.records if len(r['stages'])==meta['stages']]
        if not pool:pool=[r for r in self.records if len(r['stages'])>stage]
        if not pool:raise ValueError('No training support for current stage')
        def distance(row):
            fs=self.features[row['id']];score=abs(fs[stage][0]-current_log)+.35*sum(abs(a-b[2]) for a,b in zip(widths,fs))
            for sid,members in observed.items():
                if sid>=stage or sid>=len(fs):continue
                score+=.5*abs(math.log1p(sum(x['declared_input'] for x in members.values()))-fs[sid][0])
                score+=.5*abs(math.log1p(max(x['output'] for x in members.values()))-fs[sid][1])
            return score,row['id']
        result=sorted(pool,key=distance)[:self.k];meta['neighbor_cache'][stage]=(signature,result)
        return result

    @staticmethod
    def member_target(row,stage,member,width,cap):
        outputs=row['stages'][stage]['outputs'];index=min(len(outputs)-1,member*len(outputs)//max(1,width))
        return min(cap,outputs[index])

    def estimate(self,meta,stage,views,slack,token_seconds):
        donors=self.neighbors(meta,stage,views);cap=min(v['max_tokens'] for v in views);width=meta['widths'][stage]
        survivors=[r for r in donors if all(self.member_target(r,stage,v['member'],width,cap)>v['generated'] for v in views)]
        selected=survivors or donors;work=[];critical=[];current=[]
        for row in selected:
            remaining=[max(0.,self.member_target(row,stage,v['member'],width,cap)-v['generated']) for v in views]
            if not survivors:remaining=[max(50.,x) for x in remaining]
            w=sum(remaining)+sum(v['prefill_tokens'] for v in views)/self.prefill_unit
            c=max(remaining)+max(v['prefill_tokens'] for v in views)/self.prefill_unit;current.append(max(remaining))
            for sid in range(stage+1,min(meta['stages'],len(row['stages']))):
                z=row['stages'][sid];outs=[min(cap,x) for x in z['outputs']];ratio=meta['widths'][sid]/max(1,len(outs))
                w+=sum(outs)*ratio+sum(z['inputs'])*ratio/self.prefill_unit
                c+=max(outs)+max(z['inputs'])/self.prefill_unit+self.gap_units
            work.append(w);critical.append((self.factor*c+self.guard_units)*token_seconds)
        probability=sum(t<=slack for t in critical)/len(critical) if slack>0 else 0.
        mean_work=max(1.,statistics.fmean(work))
        return {'probability':probability,'remaining_work':mean_work,'critical_mean_seconds':statistics.fmean(critical),
                'current_remaining_tokens':statistics.fmean(current),'score':probability/mean_work,'donors':len(selected),'survival_backoff':not bool(survivors)}
