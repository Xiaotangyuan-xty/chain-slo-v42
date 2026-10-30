"""Two physical model copies on GPU0, separate KV ledgers, true batched decode.

GPU0 time-slices the replicas; this is NOT an independent multi-GPU fleet.
"""
import time
from native_backend import NativeBackend

def pack_metadata(lengths):
    if not lengths or min(lengths)<=0:raise ValueError('positive produced prefixes required')
    p=max(lengths)+1
    return p,[[1]*n+[0]*(p-n)+[1] for n in lengths]

class BatchedBackend(NativeBackend):
    def step_batch(self, items):
        if len(items)>4 or len({k for k,p,e in items})!=len(items):raise ValueError('batch boundary')
        logits={};ready=[];t=self.torch
        # First-token prefills are real forwards, not fictitious decode tokens.
        for key,prompt,emitted in items:
            self.ledger.commit('touch',key)
        for key,prompt,emitted in items:
            if not emitted:
                logits[key]=self.step(key,prompt,emitted);continue
            full=tuple(prompt)+tuple(emitted[:-1])
            self._ensure(key,list(full),True)
            self.ledger.commit('touch',key)
            ready.append((key,full,emitted[-1]))
        if not ready:return logits
        if sum(len(full)+1 for _,full,_ in ready)>self.gpu_capacity:raise RuntimeError('batch cannot fit common KV quota')
        for key,full,token in ready:self._reserve(key,len(full)+1)
        if any(self.ledger.view().get(k).gpu!=len(full) for k,full,_ in ready):raise RuntimeError('batch prefix evicted during preparation')
        pasts=[self.gpu[k] for k,_,_ in ready];lengths=[len(f) for _,f,_ in ready]
        p,masks=pack_metadata(lengths)
        with t.inference_mode():
            pads=[]
            for layer in range(len(pasts[0])):
                pair=[]
                for v in range(2):
                    pair.append(t.cat([t.nn.functional.pad(c[layer][v],(0,0,0,p-lengths[i])) for i,c in enumerate(pasts)],0))
                pads.append(tuple(pair))
            from transformers.cache_utils import DynamicCache
            pk=DynamicCache.from_legacy_cache(tuple(pads))
            ids=t.tensor([[token] for _,_,token in ready],device='cuda:0',dtype=t.long)
            mask=t.tensor(masks,device='cuda:0',dtype=t.long)
            pos=t.tensor([[n] for n in lengths],device='cuda:0',dtype=t.long)
            begin=t.cuda.Event(enable_timing=True);end=t.cuda.Event(enable_timing=True);st=time.perf_counter()
            begin.record();o=self.model.model(input_ids=ids,past_key_values=pk,attention_mask=mask,
                position_ids=pos,use_cache=True,return_dict=True)
            values=self.model.lm_head(o.last_hidden_state[:,-1,:]);end.record();end.synchronize()
            wall=time.perf_counter()-st;span=begin.elapsed_time(end)/1000
            produced=o.past_key_values.to_legacy_cache()
            for i,(key,full,token) in enumerate(ready):
                n=lengths[i]
                kv=tuple(tuple(t.cat((x[i:i+1,...,:n,:],x[i:i+1,...,p:p+1,:]),-2).contiguous() for x in pair) for pair in produced)
                self.gpu[key]=kv;self.gpu_tokens[key]=full+(token,);self.tokens[key]=full+(token,)
                self.ledger.commit('gpu_produced',key,n+1);self.ledger.commit('touch',key)
                logits[key]=values[i:i+1].clone()
                self.events.append({'event':'native_batch_query_commit','program':key,'cached':n,'queries':1,
                    'computed':n+1,'batch_size':len(ready),'batch_wall':wall,'batch_cuda':span})
            self.stats['productive_decode_queries']+=len(ready);self.stats['logical_outputs']+=len(ready)
            self.stats['actual_forward_query_tokens']+=len(ready);self.stats['forward_wall_seconds']+=wall
            self.stats.setdefault('native_decode_batches',[]).append({'B':len(ready),'S':sum(lengths),'wall':wall,'cuda':span})
        return logits

class Pool:
    def __init__(self):
        self.instances=[BatchedBackend(8192,8192),BatchedBackend(8192,8192)]
    def reset(self):
        for b in self.instances:b.reset(8192,8192)
    def before_call(self,i,key,prompt,observed_prefix):
        b=self.instances[i]
        if observed_prefix:
            previous=tuple(prompt[:observed_prefix])
            produced=b.gpu_tokens.get(key,())
            if previous[:len(produced)]!=produced:raise RuntimeError('migrated prefix identity')
            b.tokens[key]=previous
    def release(self,key):
        for b in self.instances:b.release(key)
