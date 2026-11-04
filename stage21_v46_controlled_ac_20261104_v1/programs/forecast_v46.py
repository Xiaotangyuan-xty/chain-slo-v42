"""FIT-only remaining-work estimates with full declared-stage coverage."""
import json
import math
from pathlib import Path
import statistics

class StageForecaster:
    def __init__(self,model):
        self.model=model;self.records=model['records'];self.k=model['neighbors']
        self.alpha=model['prefill_tokens_per_decode_unit'];self.gap=model['interstage_gap_decode_units']
        self.guard=model['forecast_guard_decode_units'];self.factor=model['service_safety_factor']
        self.features={r['id']:[(math.log1p(sum(s['inputs'])),math.log1p(max(s['outputs'])),len(s['outputs']))for s in r['stages']]for r in self.records}
    @classmethod
    def load(cls,path):return cls(json.loads(Path(path).read_text()))
    def neighbors(self,meta,stage,views):
        observed=meta['observed'];widths=meta['widths'];known={i:v['declared_input']for i,v in observed.get(stage,{}).items()}
        known.update({v['member']:v['declared_input']for v in views})
        signature=(tuple(sorted(known.items())),tuple((s,tuple((i,x['declared_input'],x['output'])for i,x in sorted(ms.items())))for s,ms in sorted(observed.items())))
        cached=meta['neighbor_cache'].get(stage)
        if cached and cached[0]==signature:return cached[1]
        pool=[r for r in self.records if len(r['stages'])==meta['stages']]
        if not pool:pool=[r for r in self.records if len(r['stages'])>=meta['stages']]
        if not pool:return []
        current=math.log1p(sum(known.values())*widths[stage]/max(1,len(known)))
        def distance(row):
            fs=self.features[row['id']];value=abs(fs[stage][0]-current)+.35*sum(abs(w-f[2])for w,f in zip(widths,fs))
            for s,members in observed.items():
                if s>stage or not members:continue
                if s<stage:value+=.5*abs(math.log1p(sum(v['declared_input']for v in members.values()))-fs[s][0])
                value+=.5*abs(math.log1p(max(v['output']for v in members.values()))-fs[s][1])
            return value,row['id']
        donors=sorted(pool,key=distance)[:self.k];meta['neighbor_cache'][stage]=(signature,donors);return donors
    @staticmethod
    def index(row,stage,member,width):return min(len(row['stages'][stage]['outputs'])-1,member*len(row['stages'][stage]['outputs'])//width)
    def estimate(self,meta,stage,views,slack,token_seconds,capacity=507,max_seqs=32,transfer_per_block=0.):
        donors=self.neighbors(meta,stage,views)
        if not donors:return {'supported':False,'probability':0.,'score':0.,'remaining_work':None,'critical_mean_seconds':None,'donors':0,'reason':'no_full_depth_training_support','estimated_unsubmitted_members':0}
        width=meta['widths'][stage];future_cap=meta.get('future_max_tokens',1024)
        def target(row,v):return min(v['max_tokens'],row['stages'][stage]['outputs'][self.index(row,stage,v['member'],width)])
        survivors=[row for row in donors if all(target(row,v)>v['generated']or v['generated']>=v['max_tokens']for v in views)]
        chosen=survivors or donors;live={v['member']:v for v in views};done=meta['observed'].get(stage,{})
        missing=[i for i in range(width)if i not in live and i not in done]
        work=[];times=[]
        def wave_time(components,block_charges):
            slots=max(1,min(max_seqs,int(capacity/max(1,statistics.fmean(block_charges)))))
            return max(max(components,default=0.),sum(components)/slots)
        for row in chosen:
            costs=[];charges=[]
            for v in views:
                room=max(0,v['max_tokens']-v['generated']);residual=max(0.,target(row,v)-v['generated'])
                if not survivors:residual=min(room,max(50.,residual))
                costs.append(residual+v['prefill_tokens']/self.alpha)
                charges.append(math.ceil((v['prompt']+min(v['max_tokens'],v['generated']+50))/16))
            for member in missing:
                idx=self.index(row,stage,member,width);s=row['stages'][stage]
                inp=s['inputs'][min(idx,len(s['inputs'])-1)];out=min(future_cap,s['outputs'][idx])
                costs.append(out+inp/self.alpha);charges.append(math.ceil((inp+min(out,50))/16))
            w=sum(costs);critical=wave_time(costs,charges)
            for s in range(stage+1,meta['stages']):
                donor=row['stages'][s];future_cost=[];future_blocks=[]
                for member in range(meta['widths'][s]):
                    idx=self.index(row,s,member,meta['widths'][s]);inp=donor['inputs'][min(idx,len(donor['inputs'])-1)];out=min(future_cap,donor['outputs'][idx])
                    future_cost.append(out+inp/self.alpha);future_blocks.append(math.ceil((inp+min(out,50))/16))
                w+=sum(future_cost);critical+=wave_time(future_cost,future_blocks)+self.gap
            restore=sum(v.get('restore_blocks',0)for v in views)*transfer_per_block
            work.append(w);times.append(token_seconds*(self.factor*critical+self.guard)+restore)
        mean=max(1.,statistics.fmean(work));prob=sum(t<=slack for t in times)/len(times)if slack>0 else 0.
        return {'supported':True,'probability':prob,'score':prob/mean,'remaining_work':mean,'critical_mean_seconds':statistics.fmean(times),
          'donors':len(chosen),'survival_backoff':not bool(survivors),'estimated_unsubmitted_members':len(missing)}
