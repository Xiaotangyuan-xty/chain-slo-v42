"""v44: predicted completion density, short KV horizon and bounded residence."""
from collections import deque
from pathlib import Path
import time

from vllm.core.scheduler import PreemptionMode,SchedulerOutputs
from vllm.sequence import SequenceStatus
from jitserve.scheduler.policy import SLOPolicy
from jitserve.scheduler.slo_scheduler import SLOScheduler
from chain_forecast import StageForecaster
from chain_admission import select,blocks

MODEL=StageForecaster.load(Path(__file__).resolve().parents[1]/'assets/CHAIN_MODEL.json')

class ChainCompletionPolicy(SLOPolicy):
    version='v44_native_chain_completion_density'
    def __init__(self):
        super().__init__(schedule_interval=50,penalty_factor=100)
        self.chains={};self.priorities={};self.draining=True
    def add_seq_group(self,seq_group):
        info=seq_group.request_info;cid=seq_group.collection_id
        if cid not in self.chains or (info.stage_id==0 and self.chains[cid]['finished']):
            stages=getattr(info,'chain_stage_count',6)
            self.chains[cid]={'stages':stages,'widths':getattr(info,'chain_stage_widths',[1]*stages),
                              'deadline':info.chain_deadline,'observed':{},'neighbor_cache':{},'finished':False}
        self.chains[cid]['deadline']=min(self.chains[cid]['deadline'],info.chain_deadline)
        super().add_seq_group(seq_group)
    def delete_seq_group(self,seq_group):
        info=seq_group.request_info;meta=self.chains[seq_group.collection_id]
        if seq_group.first_seq.status!=SequenceStatus.FINISHED_ABORTED:
            stage=meta['observed'].setdefault(info.stage_id,{})
            stage[info.request_id]={'declared_input':info.input_len,'output':seq_group.first_seq.get_output_len()}
            if info.stage_id==meta['stages']-1 and len(stage)>=meta['widths'][info.stage_id]:meta['finished']=True
        self.priorities.pop(seq_group.request_id,None)
        super().delete_seq_group(seq_group)
    def get_priority(self,group):
        if self.draining:return (2,group.arrival_time)
        return self.priorities.get(group.request_id,(1,group.request_info.chain_deadline,group.arrival_time))
    def goals(self,groups,now):
        grouped={}
        for group in groups:grouped.setdefault(group.collection_id,[]).append(group)
        result=[];seconds=max(.01,min(.2,self._get_interval_time()));self.priorities={}
        for cid,gs in grouped.items():
            stage=gs[0].request_info.stage_id
            if any(g.request_info.stage_id!=stage for g in gs):raise RuntimeError('Concurrent stages in one chain')
            meta=self.chains[cid]
            views=[{'request':g.request_id,'member':g.request_info.request_id,'declared_input':g.request_info.input_len,
                    'prompt':g.first_seq.get_prompt_len(),'generated':g.first_seq.get_output_len(),'max_tokens':g.sampling_params.max_tokens,
                    'prefill_tokens':g.get_num_uncomputed_tokens() if g.is_prefill() else 0,'arrival':g.arrival_time} for g in gs]
            estimate=MODEL.estimate(meta,stage,views,meta['deadline']-now,seconds)
            incumbent=any(g.first_seq.status==SequenceStatus.RUNNING for g in gs)
            score=estimate['score']*(1.2 if incumbent else 1.)
            goal={**estimate,'chain':cid,'stage':stage,'score':score,'deadline':meta['deadline'],
                  'arrival':min(g.arrival_time for g in gs),'members':views}
            for g in gs:self.priorities[g.request_id]=(0 if estimate['probability']>0 else 1,-score,meta['deadline'],g.arrival_time)
            result.append(goal)
        return result

class NativeChainScheduler(SLOScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.policy=ChainCompletionPolicy();self.reservation_events=[];self.selected=set();self.rows=[];self.leases={}
        self.last_plan=-50;self.last_members=None;self.last_recorded_selected=None
    def _schedule(self):
        self.policy.update_schedule_count();self.free_finished_seq_groups()
        groups=list(self.running)+list(self.waiting)+list(self.swapped);live={g.request_id for g in groups};now=time.perf_counter()
        for rid in list(self.leases):
            if rid not in live:self.leases.pop(rid,None)
        for g in self.running:self.leases.setdefault(g.request_id,g.first_seq.get_output_len())
        tick=self.policy.num_schedule_count
        sibling=False
        if self.selected:
            selected_chains={g.collection_id for g in groups if g.request_id in self.selected}
            sibling=any(g.request_id not in (self.last_members or set()) and g.collection_id in selected_chains for g in groups)
        replan=(tick-self.last_plan>=50 or (self.selected is not None and not (self.selected&live))
                or (not self.last_members and bool(live)) or sibling)
        capacity=self.block_manager.num_total_gpu_blocks-self.block_manager.watermark_blocks
        if replan:
            goals=self.policy.goals(groups,now) if groups else []
            self.selected,self.rows=select(goals,capacity,self.scheduler_config.max_num_seqs,self.cache_config.block_size)
            self.policy.draining=self.selected is None;self.last_plan=tick
        if self.policy.draining:selected=set(live);holds=[]
        else:
            selected=set(self.selected)&live;holds=[]
            for g in self.running:
                if (g.request_id not in selected and g.request_info.chain_deadline>now
                    and g.first_seq.get_output_len()-self.leases[g.request_id]<50):
                    selected.add(g.request_id);holds.append(g.request_id)
        if replan or live!=self.last_members or frozenset(selected)!=self.last_recorded_selected:
            self.reservation_events.append({'kind':'native_chain_reservation','monotonic':now,'tick':tick,
                'mode':'native_work_conserving_drain' if self.policy.draining else 'completion_density_horizon',
                'rows':self.rows,'selected':sorted(selected),'lease_holds':holds,
                'reserved_blocks':sum(blocks({'prompt':g.first_seq.get_prompt_len(),'generated':g.first_seq.get_output_len(),
                    'max_tokens':g.sampling_params.max_tokens},self.cache_config.block_size) for g in groups if g.request_id in selected),
                'capacity':capacity,'free_gpu':self.block_manager.get_num_free_gpu_blocks(),'free_cpu':self.block_manager.get_num_free_cpu_blocks()})
            self.last_members=live;self.last_recorded_selected=frozenset(selected)
        swaps=[];preemptions=0
        if not self.policy.draining:
            for g in sorted(list(self.running),key=self.policy.get_priority,reverse=True):
                if g.request_id in selected or preemptions>=self.max_swaps_per_iter:continue
                mode=PreemptionMode.SWAP if self.block_manager.can_swap_out(g) else PreemptionMode.RECOMPUTE
                mode=self._preempt(g,swaps,mode);self.running.remove(g);self.leases.pop(g.request_id,None)
                (self.swapped if mode==PreemptionMode.SWAP else self.waiting).append(g);preemptions+=1
        if swaps or preemptions:return SchedulerOutputs([],0,0,[],swaps,[],[],0,len(self.running),preemptions)
        hidden_waiting=[g for g in self.waiting if g.request_id not in selected]
        hidden_swapped=[g for g in self.swapped if g.request_id not in selected]
        self.waiting=deque(g for g in self.waiting if g.request_id in selected)
        self.swapped=deque(sorted((g for g in self.swapped if g.request_id in selected),key=self.policy.get_priority))
        self.running=deque(sorted(self.running,key=self.policy.get_priority))
        try:result=self._schedule_chunked_prefill()
        finally:
            self.waiting.extend(hidden_waiting);self.swapped.extend(hidden_swapped)
        return result

def install(server):
    import vllm.engine.llm_engine as llm_engine
    llm_engine.SLOScheduler=NativeChainScheduler
    original=server._handle_collective_request
    def collective(info,request_dict,prompt):
        stages=request_dict.get('num_stages',6)
        info.chain_deadline=time.perf_counter()+info.slo_constraint[2]*stages-request_dict.get('served_time',0.)
        info.chain_stage_count=stages;info.chain_stage_widths=list(request_dict.get('requests_per_stage') or [1]*stages)
        return original(info,request_dict,prompt)
    server._handle_collective_request=collective
