"""Single active Qwen forward with explicit KV residency and teacher forcing.

No torch/transformers import until NativeBackend() after the GPU0 supervisor gate.
No production batching or native vLLM performance claim. All tensors stay remote.
"""
import json, os, time
from contracts import CacheLedger

class NativeBackend:
    def __init__(self, gpu_capacity=2048, host_capacity=2048):
        from study_gate import GPU
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(GPU):raise RuntimeError('registered physical GPU mask required')
        import torch
        torch.cuda._device_count_nvml=lambda:-1
        from scoped_cuda_platform import cuda_driver_uuid
        from study_gate import EXPECTED_UUID
        if cuda_driver_uuid()!=EXPECTED_UUID or torch.cuda.device_count()!=1:
            raise RuntimeError('logical CUDA0 differs from registered physical GPU')
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(4)
        self.torch=torch
        self.model=AutoModelForCausalLM.from_pretrained('/home/taig/models/Qwen2.5-7B-Instruct',
            torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,
            trust_remote_code=False).eval().to('cuda:0')
        self.stream=torch.cuda.Stream(device=0)
        self.reset(gpu_capacity,host_capacity)

    def reset(self, gpu_capacity=None, host_capacity=None):
        t=self.torch;t.cuda.synchronize(0)
        self.gpu={};self.host={};self.pending={};self.tokens={};self.gpu_tokens={};self.host_tokens={}
        self.ledger=CacheLedger(gpu_capacity or getattr(self,'gpu_capacity',2048),
                                host_capacity if host_capacity is not None else getattr(self,'host_capacity',2048))
        self.gpu_capacity,self.host_capacity=self.ledger.gpu_capacity,self.ledger.host_capacity
        self.stats={'recompute_tokens':0,'recompute_wall_seconds':0.,'recompute_cuda_span_seconds':0.,
                    'swap_bytes_out':0,'swap_bytes_in':0,'copy_cuda_span_seconds':0.,
                    'exposed_transfer_seconds':0.,'necessary_prefill_tokens':0,
                    'copy_launch_blocked_seconds':0.,'copy_wait_blocked_seconds':0.,
                    'productive_decode_queries':0,'logical_outputs':0,'evictions':0,'resident_resumes':0,
                    'prefetches':0,'actual_forward_query_tokens':0,'forward_wall_seconds':0.}
        self.events=[]
        t.cuda.reset_peak_memory_stats(0)

    @staticmethod
    def nbytes(cache):return sum(x.numel()*x.element_size() for pair in cache for x in pair)

    def _forward(self, ids, past=None):
        if not ids:raise RuntimeError('empty query is not computation')
        from transformers.cache_utils import DynamicCache
        t=self.torch;begin=t.cuda.Event(enable_timing=True);end=t.cuda.Event(enable_timing=True)
        start=time.perf_counter()
        with t.inference_mode():
            x=t.tensor([ids],device='cuda:0',dtype=t.long)
            pk=DynamicCache.from_legacy_cache(past) if past else DynamicCache()
            begin.record()
            out=self.model.model(input_ids=x,past_key_values=pk,use_cache=True,return_dict=True)
            logits=self.model.lm_head(out.last_hidden_state[:,-1,:])
            end.record();t.cuda.current_stream(0).synchronize()
            kv=out.past_key_values.to_legacy_cache()
        wall=time.perf_counter()-start
        self.stats['actual_forward_query_tokens']+=len(ids)
        self.stats['forward_wall_seconds']+=wall
        return kv,logits,wall,begin.elapsed_time(end)/1000

    def _copy(self, cache, to_host):
        wall_begin=time.perf_counter()
        t=self.torch;start=t.cuda.Event(enable_timing=True);end=t.cuda.Event(enable_timing=True)
        with t.cuda.stream(self.stream):
            start.record();out=[]
            for pair in cache:
                row=[]
                for x in pair:
                    if to_host:
                        y=t.empty_like(x,device='cpu',pin_memory=True);y.copy_(x,non_blocking=True)
                        x.record_stream(self.stream)
                    else:y=x.to('cuda:0',non_blocking=True)
                    row.append(y)
                out.append(tuple(row))
            end.record()
        # This single host thread has not yet submitted the next model forward.
        # Allocation and copy submission therefore block model progress too.
        launch=time.perf_counter()-wall_begin
        self.stats['copy_launch_blocked_seconds']+=launch
        self.stats['exposed_transfer_seconds']+=launch
        return tuple(out),start,end

    def _record_copy_wait(self,seconds):
        self.stats['copy_wait_blocked_seconds']+=seconds
        self.stats['exposed_transfer_seconds']+=seconds

    def resolve(self, key):
        if key not in self.pending:return
        cache,start,end=self.pending.pop(key);begin=time.perf_counter();end.synchronize()
        self._record_copy_wait(time.perf_counter()-begin)
        self.stats['copy_cuda_span_seconds']+=start.elapsed_time(end)/1000
        for pair in cache:
            for x in pair:x.record_stream(self.torch.cuda.current_stream(0))
        self.gpu[key]=cache;self.gpu_tokens[key]=self.host_tokens[key]
        self.ledger.commit('gpu_produced',key,len(self.gpu_tokens[key]))

    def _reserve(self,key,n):
        old=self.ledger.view().get(key)
        if self.ledger.view().free >= max(0,n-old.gpu):return
        for k in list(self.pending):self.resolve(k)
        actions=self.ledger.plan(key,n)
        for event,victim,count in actions:
            if event=='swap_out':
                cache=self.gpu[victim];copied,start,end=self._copy(cache,True)
                begin=time.perf_counter();end.synchronize()
                self._record_copy_wait(time.perf_counter()-begin)
                self.stats['copy_cuda_span_seconds']+=start.elapsed_time(end)/1000
                self.stats['swap_bytes_out']+=self.nbytes(cache)
                self.host[victim]=copied;self.host_tokens[victim]=self.gpu_tokens[victim]
                self.gpu.pop(victim);self.gpu_tokens.pop(victim);self.stats['evictions']+=1
            elif event=='drop_gpu':
                self.gpu.pop(victim,None);self.gpu_tokens.pop(victim,None);self.stats['evictions']+=1
            elif event=='drop_host':self.host.pop(victim,None);self.host_tokens.pop(victim,None)
            self.ledger.commit(event,victim,count)
            self.events.append({'event':event,'program':victim,'tokens':count,'cause':'common_LRU_capacity'})

    def prefetch(self,key):
        e=self.ledger.view().get(key)
        if e.gpu or e.pending or not e.host or self.ledger.view().free<e.host:return False
        cache,start,end=self._copy(self.host[key],False)
        self.pending[key]=(cache,start,end)
        self.ledger.commit('reserve_copy',key,e.host)
        self.stats['swap_bytes_in']+=self.nbytes(cache);self.stats['prefetches']+=1
        return True

    def _ensure(self,key,ids,recovery):
        self.resolve(key)
        self._reserve(key,len(ids))
        e=self.ledger.view().get(key)
        if e.host>e.gpu:
            # Prefix verification uses actual previously produced token IDs.
            if self.host_tokens[key] != tuple(ids[:e.host]):raise RuntimeError('host prefix witness mismatch')
            cache,start,end=self._copy(self.host[key],False)
            begin=time.perf_counter();end.synchronize()
            self._record_copy_wait(time.perf_counter()-begin)
            self.stats['copy_cuda_span_seconds']+=start.elapsed_time(end)/1000
            self.stats['swap_bytes_in']+=self.nbytes(cache)
            for pair in cache:
                for x in pair:x.record_stream(self.torch.cuda.current_stream(0))
            self.gpu[key]=cache;self.gpu_tokens[key]=self.host_tokens[key]
            self.ledger.commit('gpu_produced',key,e.host);e=self.ledger.view().get(key)
        cached=e.gpu
        if cached>len(ids) or (cached and self.gpu_tokens[key] != tuple(ids[:cached])):
            raise RuntimeError('GPU prefix witness mismatch')
        if cached==len(ids):return None
        kv,logits,wall,cuda=self._forward(ids[cached:],self.gpu.get(key))
        self.gpu[key]=kv;self.gpu_tokens[key]=tuple(ids)
        # Publish only after the native producer completed.
        if kv[0][0].shape[-2]!=len(ids):raise RuntimeError('native KV length mismatch')
        self.ledger.commit('gpu_produced',key,len(ids));self.tokens[key]=tuple(ids)
        if recovery:
            self.stats['recompute_tokens']+=len(ids)-cached
            self.stats['recompute_wall_seconds']+=wall;self.stats['recompute_cuda_span_seconds']+=cuda
        else:self.stats['necessary_prefill_tokens']+=len(ids)-cached
        self.events.append({'event':'native_query_commit','program':key,'cached':cached,'queries':len(ids)-cached,
                            'computed':len(ids),'recovery':recovery,'wall':wall,'cuda_span':cuda})
        return logits

    def step(self,key,prompt,emitted):
        """One output opportunity; emitted contains only already emitted tokens."""
        previous=self.tokens.get(key,())
        full=tuple(prompt)+tuple(emitted[:-1]) if emitted else tuple(prompt)
        if not emitted:
            common=0
            for a,b in zip(previous,full):
                if a!=b:break
                common+=1
            if common!=len(previous):raise RuntimeError('new call changed a previously computed prefix')
            if previous:self._ensure(key,list(previous),True)
            logits=self._ensure(key,list(full),False)
            if logits is None:raise RuntimeError('zero-new-token initial call not registered')
        else:
            if self.ledger.view().get(key).gpu==len(full):self.stats['resident_resumes']+=1
            self._ensure(key,list(full),True)
            self._reserve(key,len(full)+1)
            kv,logits,wall,cuda=self._forward([emitted[-1]],self.gpu.get(key))
            if kv[0][0].shape[-2]!=len(full)+1:raise RuntimeError('decode producer watermark mismatch')
            self.gpu[key]=kv;self.gpu_tokens[key]=full+(emitted[-1],);self.tokens[key]=self.gpu_tokens[key]
            self.ledger.commit('gpu_produced',key,len(full)+1)
            self.stats['productive_decode_queries']+=1
            self.events.append({'event':'native_query_commit','program':key,'cached':len(full),'queries':1,
                                'computed':len(full)+1,'recovery':False,'productive_decode':True,
                                'wall':wall,'cuda_span':cuda})
        self.ledger.commit('touch',key)
        self.stats['logical_outputs']+=1
        return logits

    def release(self,key):
        self.resolve(key)
        for d in (self.gpu,self.host,self.gpu_tokens,self.host_tokens,self.tokens):d.pop(key,None)
        self.ledger.commit('release',key)

    def summary(self):
        for k in list(self.pending):self.resolve(k)
        return {**self.stats,'native_peak_allocated_bytes':self.torch.cuda.max_memory_allocated(0),
                'persistent_quota_tokens':self.gpu_capacity,'host_quota_tokens':self.host_capacity,
                'restoration_exposure_seconds':self.stats['recompute_wall_seconds']+self.stats['exposed_transfer_seconds'],
                'CUDA_span_is_not_sum_of_kernel_times':True}
