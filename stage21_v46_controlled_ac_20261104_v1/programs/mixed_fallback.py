"""Calibration-only mixed adapter; frozen v45 bytes are never edited."""
import atexit
from collections import Counter
import json
from pathlib import Path

from jitserve.request_info import RequestType
from jitserve.scheduler.policy import SLOPolicy
from jitserve.scheduler.slo_scheduler import SLOScheduler
from native_chain import ChainCompletionPolicy, NativeChainScheduler

VARIANT='v45_mixed_native_fallback_calibration_v1'

class MixedPolicy(ChainCompletionPolicy):
    mixed_variant=VARIANT
    def __init__(self):
        super().__init__();self.native_mode=False
    def add_seq_group(self,seq_group):
        g=seq_group
        if g.request_type==RequestType.COLLECTIVE:return super().add_seq_group(g)
        return SLOPolicy.add_seq_group(self,g)
    def delete_seq_group(self,seq_group):
        g=seq_group
        if g.request_type==RequestType.COLLECTIVE:return super().delete_seq_group(g)
        return SLOPolicy.delete_seq_group(self,g)
    def get_priority(self,seq_group):
        g=seq_group
        if self.native_mode:return SLOPolicy.get_priority(self,g)
        return super().get_priority(g)

class MixedScheduler(NativeChainScheduler):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw);self.policy=MixedPolicy();self.mixed_counts=Counter();self.last_native=False
    def _schedule(self):
        groups=[g for g in list(self.running)+list(self.waiting)+list(self.swapped)if not g.is_finished()]
        foreground=[g for g in groups if g.request_type!=RequestType.COLLECTIVE]
        compounds=[g for g in groups if g.request_type==RequestType.COLLECTIVE]
        native=bool(foreground or getattr(self,'_pending_swap_out_queue',()))
        phase='main'if any(g.request_info.client_id!=900000 for g in groups)else'warmup'
        self.mixed_counts[phase+'_scheduler_calls']+=1
        self.mixed_counts['peak_foreground_live']=max(self.mixed_counts['peak_foreground_live'],len(foreground))
        if foreground and compounds:self.mixed_counts[phase+'_mixed_native_calls']+=1
        if native!=self.last_native:
            self.mixed_counts['mode_transitions']+=1;self.policy.seq_group_slo_dict={}
            if not native:
                self.last_plan=self.policy.num_schedule_count-50
                self.last_members=None;self.last_recorded_selected=None;self.leases={};self.selected=set()
        self.last_native=native;self.policy.native_mode=native
        if native:
            self.policy.draining=True;self.selected=None
            self.mixed_counts[phase+'_native_fallback_calls']+=1
            return SLOScheduler._schedule(self)
        self.mixed_counts[phase+'_v45_chain_calls']+=1
        return NativeChainScheduler._schedule(self)

def install(server,output):
    import vllm.engine.llm_engine as llm_engine
    llm_engine.SLOScheduler=MixedScheduler
    def receipt():
        engine=getattr(server,'engine',None)
        scheduler=engine.engine.scheduler[0]if engine is not None else None
        data={'variant':VARIANT,'rule':'Any live latency/deadline singleton, or pending native swap cleanup: original native global scheduling on all queues; otherwise frozen v45 scheduling on compound chains.',
              'purpose':'Short calibration of adapter compatibility and resource cost, not an evaluation of an optimized mixed-v45 design.',
              'counts':dict(scheduler.mixed_counts)if scheduler is not None else{},'frozen_core_bytes_modified':False}
        p=Path(output)/'MIXED_RUNTIME.json';p.write_text(json.dumps(data,indent=2)+'\n')
    atexit.register(receipt)
