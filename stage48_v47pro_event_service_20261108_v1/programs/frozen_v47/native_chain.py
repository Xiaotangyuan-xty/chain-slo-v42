"""v47: token credit, reversible expired-only drain and capacity-aware batching."""
from collections import Counter, deque
import heapq
from pathlib import Path
import time

from vllm.core.scheduler import PreemptionMode,SchedulerOutputs
from vllm.sequence import SequenceStatus
from jitserve.scheduler.policy import SLOPolicy
from jitserve.scheduler.slo_scheduler import SLOScheduler
from .chain_forecast import StageForecaster
from .chain_admission import select,blocks
from .token_reward import REWARD

MODEL=StageForecaster.load(Path(__file__).resolve().parents[2]/'assets/CHAIN_MODEL.json')

class ChainCompletionPolicy(SLOPolicy):
    version='v47_token_reward_deadline_drain_cached_forecast'
    def __init__(self):
        super().__init__(schedule_interval=50,penalty_factor=100)
        self.chains={};self.priorities={};self.draining=True
        self.deadline_heap=[];self.membership_epoch=0
    def add_seq_group(self,seq_group):
        info=seq_group.request_info;cid=seq_group.collection_id
        fresh=cid not in self.chains or (info.stage_id==0 and self.chains[cid]['finished'])
        if fresh:
            prior_version=self.chains.get(cid,{}).get('deadline_version',0)
            stages=getattr(info,'chain_stage_count',6)
            self.chains[cid]={'stages':stages,'widths':getattr(info,'chain_stage_widths',[1]*stages),
                              'deadline':info.chain_deadline,'observed':{},'neighbor_cache':{},
                              'finished':False,'aborted':False,'deadline_version':prior_version}
        meta=self.chains[cid]
        deadline=min(meta['deadline'],info.chain_deadline)
        if fresh or deadline!=meta['deadline']:
            meta['deadline']=deadline;meta['deadline_version']+=1
            heapq.heappush(self.deadline_heap,(-deadline,cid,meta['deadline_version']))
        self.membership_epoch+=1
        if len(self.deadline_heap)>4*len(self.chains)+64:
            self.deadline_heap=[(-m['deadline'],c,m['deadline_version'])
                                for c,m in self.chains.items() if not m['finished']]
            heapq.heapify(self.deadline_heap)
        super().add_seq_group(seq_group)

    def has_live_credit(self,now):
        # An unfinished root remains registered during a stage barrier. This is
        # reversible: a newly arriving root immediately adds a fresh deadline.
        while self.deadline_heap:
            negative,cid,version=self.deadline_heap[0];meta=self.chains[cid]
            if meta['finished'] or meta.get('aborted') or version!=meta['deadline_version']:
                heapq.heappop(self.deadline_heap);continue
            return -negative>now
        return False

    def delete_seq_group(self,seq_group):
        info=seq_group.request_info;meta=self.chains[seq_group.collection_id]
        if seq_group.first_seq.status!=SequenceStatus.FINISHED_ABORTED:
            stage=meta['observed'].setdefault(info.stage_id,{})
            seq=seq_group.first_seq
            output=seq.get_output_len()
            stage[info.request_id]={'declared_input':info.input_len,'output':output,
                'reward_input':getattr(info,'chain_reward_input_tokens',seq.get_prompt_len()),
                'reward_output':REWARD.count(seq.output_text,output) if REWARD.tokenizer is not None else output}
            if info.stage_id==meta['stages']-1 and len(stage)>=meta['widths'][info.stage_id]:
                meta['finished']=True;meta['neighbor_cache'].clear()
        else:
            meta['aborted']=True;meta['deadline_version']+=1
        self.membership_epoch+=1
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
                    'prefill_tokens':g.get_num_uncomputed_tokens() if g.is_prefill() else 0,'arrival':g.arrival_time,
                    'reward_input':getattr(g.request_info,'chain_reward_input_tokens',g.first_seq.get_prompt_len())} for g in gs]
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
        self.base_max_num_seqs=self.scheduler_config.max_num_seqs
        self.fast_expired_drain=False;self.drain_key=None;self.v47_counts=Counter()
    def _restore_drain_limits(self):
        if self.fast_expired_drain:
            self.scheduler_config.max_num_seqs=self.pre_drain_max_num_seqs
            self.fast_expired_drain=False;self.drain_key=None
            self.selected=set();self.last_plan=self.policy.num_schedule_count-50
            self.last_members=None;self.v47_counts['expired_drain_exits']+=1

    def _expired_drain(self,groups,live,tick):
        if not self.fast_expired_drain:
            self.pre_drain_max_num_seqs=self.scheduler_config.max_num_seqs
            self.fast_expired_drain=True;self.leases.clear();self.policy.priorities.clear()
            self.v47_counts['expired_drain_entries']+=1
        self.policy.draining=True;self.selected=None;self.rows=[]
        key=(self.policy.membership_epoch,tick//50)
        if key!=self.drain_key:
            capacity=self.block_manager.num_total_gpu_blocks-self.block_manager.watermark_blocks
            used=0;slots=0
            def cost(g):
                return blocks({'prompt':g.first_seq.get_prompt_len(),
                               'generated':g.first_seq.get_output_len(),
                               'max_tokens':g.sampling_params.max_tokens},self.cache_config.block_size)
            for g in self.running:
                used+=cost(g);slots+=1
            limit=max(self.base_max_num_seqs,min(128,self.scheduler_config.max_num_batched_tokens))
            # Match the native swapped-before-waiting admission order. Never
            # reserve extra KV or replace the physical allocator.
            for g in list(self.swapped)+list(self.waiting):
                charge=cost(g)
                if slots>=limit or used+charge>capacity:break
                used+=charge;slots+=1
            self.scheduler_config.max_num_seqs=min(limit,max(self.base_max_num_seqs,slots))
            self.drain_key=key;self.v47_counts['drain_capacity_refreshes']+=1
            self.v47_counts['max_drain_seq_limit']=max(self.v47_counts['max_drain_seq_limit'],self.scheduler_config.max_num_seqs)
        self.last_members=live;self.last_recorded_selected=frozenset(live)
        self.v47_counts['expired_drain_ticks']+=1
        return self._schedule_chunked_prefill()

    def _schedule(self):
        self.policy.update_schedule_count();self.free_finished_seq_groups()
        groups=list(self.running)+list(self.waiting)+list(self.swapped);live={g.request_id for g in groups};now=time.perf_counter()
        if groups and not self.policy.has_live_credit(now):
            return self._expired_drain(groups,live,self.policy.num_schedule_count)
        self._restore_drain_limits()
        for rid in list(self.leases):
            if rid not in live:self.leases.pop(rid,None)
        for g in self.running:self.leases.setdefault(g.request_id,g.first_seq.get_output_len())
        tick=self.policy.num_schedule_count
        sibling=False
        if self.selected:
            selected_chains={g.collection_id for g in groups if g.request_id in self.selected}
            sibling=any(g.request_id not in (self.last_members or set()) and g.collection_id in selected_chains for g in groups)
        arrival=self.last_members is not None and bool(live-self.last_members)
        replan=(tick-self.last_plan>=50 or (self.selected is not None and not (self.selected&live))
                or (not self.last_members and bool(live)) or sibling or arrival)
        capacity=self.block_manager.num_total_gpu_blocks-self.block_manager.watermark_blocks
        protected={g.request_id for g in self.running if g.request_info.chain_deadline>now
                   and g.first_seq.get_output_len()-self.leases[g.request_id]<50}
        if replan:
            goals=self.policy.goals(groups,now) if groups else []
            self.selected,self.rows=select(goals,capacity,self.scheduler_config.max_num_seqs,self.cache_config.block_size,protected)
            self.policy.draining=self.selected is None;self.last_plan=tick
        if self.policy.draining:selected=set(live);holds=[]
        else:
            selected=set(self.selected)&live;holds=[]
            for g in self.running:
                if (g.request_id not in selected and g.request_info.chain_deadline>now
                    and g.first_seq.get_output_len()-self.leases[g.request_id]<50):
                    selected.add(g.request_id);holds.append(g.request_id)
        if replan or live!=self.last_members or frozenset(selected)!=self.last_recorded_selected:
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
        tokenizer=getattr(getattr(server,'graph_ctx',None),'generation_tokenizer',None)
        REWARD.bind(tokenizer)
        if tokenizer is not None:
            info.chain_reward_input_tokens=REWARD.count(prompt,getattr(info,'input_len',0))
        return original(info,request_dict,prompt)
    server._handle_collective_request=collective
