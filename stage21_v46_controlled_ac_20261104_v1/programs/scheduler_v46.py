"""v46: complete remaining-work accounting and joint foreground/chain admission."""
from collections import deque,Counter
from pathlib import Path
import time,math
from vllm.core.scheduler import PreemptionMode,SchedulerOutputs
from vllm.sequence import SequenceStatus
from jitserve.request_info import RequestType
from jitserve.scheduler.policy import SLOPolicy
from jitserve.scheduler.slo_scheduler import SLOScheduler
from forecast_v46 import StageForecaster
from admission_v46 import choose,blocks
from observation_v46 import ExecutionCadence
MODEL=StageForecaster.load(Path(__file__).resolve().parents[1]/'assets/CHAIN_MODEL.json')
VERSION='v46_complete_work_joint_residency'

class WorkflowPolicy(SLOPolicy):
    version=VERSION
    mixed_variant='v46_joint_foreground_budget_v1'
    def __init__(self):
        super().__init__(schedule_interval=50,penalty_factor=100);self.chains={};self.priorities={};self.draining=False
        self.foreground_deadlines={}
    def add_seq_group(self,seq_group):
        g=seq_group;info=g.request_info
        if g.request_type==RequestType.COLLECTIVE:
            cid=g.collection_id
            if cid not in self.chains or(info.stage_id==0 and self.chains[cid]['finished']):
                self.chains[cid]={'stages':info.chain_stage_count,'widths':list(info.chain_stage_widths),'deadline':info.chain_deadline,
                   'observed':{},'neighbor_cache':{},'finished':False,'future_max_tokens':1024}
            self.chains[cid]['deadline']=min(self.chains[cid]['deadline'],info.chain_deadline)
        else:self.foreground_deadlines[g.request_id]=time.perf_counter()+max(0.,g.deadline)
        return super().add_seq_group(g)
    def delete_seq_group(self,seq_group):
        g=seq_group;info=g.request_info
        if g.request_type==RequestType.COLLECTIVE and g.first_seq.status!=SequenceStatus.FINISHED_ABORTED:
            meta=self.chains[g.collection_id];observed=meta['observed'].setdefault(info.stage_id,{})
            observed[info.request_id]={'declared_input':info.input_len,'output':g.first_seq.get_output_len()}
            if info.stage_id==meta['stages']-1 and len(observed)==meta['widths'][info.stage_id]:meta['finished']=True
        self.priorities.pop(g.request_id,None);self.foreground_deadlines.pop(g.request_id,None)
        return super().delete_seq_group(g)
    def get_priority(self,seq_group):
        g=seq_group
        if g.request_type!=RequestType.COLLECTIVE:return(0,)+tuple(SLOPolicy.get_priority(self,g))+(g.arrival_time,)
        return self.priorities.get(g.request_id,(2,self.chains[g.collection_id]['deadline'],g.arrival_time))
    def workflow_deadline(self,g):
        return self.chains[g.collection_id]['deadline']if g.request_type==RequestType.COLLECTIVE else self.foreground_deadlines[g.request_id]

class WorkflowScheduler(SLOScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.policy=WorkflowPolicy();self.cadence=ExecutionCadence();self.decision_log=None
        self.leases={};self.last_plan=-50;self.last_live=set();self.selected=set();self.last_recorded_selected=frozenset()
        self.last_plan_key=None;self.reservation_events=[];self.mixed_counts=Counter();self.rows=[];self.initial_max_seqs=self.scheduler_config.max_num_seqs
    def emit(self,kind,**fields):
        if self.decision_log:self.decision_log.write(kind,**fields)
    def view(self,g):
        seq=g.first_seq;generated=seq.get_output_len();prompt=seq.get_prompt_len()
        return {'request':g.request_id,'member':g.request_info.request_id,'declared_input':g.request_info.input_len,'prompt':prompt,
          'generated':generated,'max_tokens':g.sampling_params.max_tokens,'prefill_tokens':g.get_num_uncomputed_tokens()if g.is_prefill()else 0,
          'arrival':g.arrival_time,'restore_blocks':math.ceil((prompt+generated)/self.cache_config.block_size)if seq.status==SequenceStatus.SWAPPED else 0}
    def protected(self,groups,now):
        live={g.request_id for g in groups}
        for rid in set(self.leases)-live:self.leases.pop(rid)
        protected=set()
        for g in self.running:
            v=self.view(g);seq=g.first_seq;c=seq.data.get_num_computed_tokens()
            lease=self.leases.setdefault(g.request_id,{'generated':v['generated'],'computed':c,'context':v['prompt']+v['generated'],
               'prefill_paid':0.,'created':now,'wall_budget':min(10.,max(.5,75*self.cadence.token_seconds))})
            lease['prefill_paid']=max(lease['prefill_paid'],max(0,min(c,lease['context'])-lease['computed'])/256.)
            progress=max(0,v['generated']-lease['generated'])+lease['prefill_paid']
            if self.policy.workflow_deadline(g)>now and progress<50 and now-lease['created']<lease['wall_budget']:protected.add(g.request_id)
        return protected
    def plan(self,groups,now,protected):
        capacity=self.block_manager.num_total_gpu_blocks-self.block_manager.watermark_blocks;slots=self.initial_max_seqs
        foreground=[g for g in groups if g.request_type!=RequestType.COLLECTIVE];chains=[g for g in groups if g.request_type==RequestType.COLLECTIVE]
        fg_sorted=sorted(foreground,key=lambda g:(g.request_id not in protected,self.policy.get_priority(g)))
        fg_selected=set();fg_used=0;quota=capacity//2 if chains else capacity;fg_slots=max(1,slots//2)if chains else slots
        for g in fg_sorted:
            charge=blocks(self.view(g),self.cache_config.block_size)
            if fg_used+charge<=quota and len(fg_selected)<fg_slots:fg_selected.add(g.request_id);fg_used+=charge
        # A large foreground call can borrow, but must fit the entire logical pool.
        if foreground and not fg_selected:
            for g in fg_sorted:
                charge=blocks(self.view(g),self.cache_config.block_size)
                if charge<=capacity:fg_selected.add(g.request_id);fg_used=charge;break
        goals=[];bychain={};self.policy.priorities={}
        for g in chains:bychain.setdefault(g.collection_id,[]).append(g)
        for cid,members in bychain.items():
            stage=members[0].request_info.stage_id
            if any(g.request_info.stage_id!=stage for g in members):raise RuntimeError('Concurrent workflow stages violate the declared barrier')
            meta=self.policy.chains[cid];views=[self.view(g)for g in members]
            estimate=MODEL.estimate(meta,stage,views,meta['deadline']-now,self.cadence.token_seconds,
               max(1,capacity-fg_used),max(1,slots-len(fg_selected)),self.cadence.transfer_per_block)
            score=estimate['score']*(1.2 if any(g.first_seq.status==SequenceStatus.RUNNING for g in members)else 1.)
            goal={**estimate,'chain':cid,'stage':stage,'score':score,'deadline':meta['deadline'],'arrival':min(g.arrival_time for g in members),'members':views}
            goals.append(goal)
            for g in members:self.policy.priorities[g.request_id]=(1,0 if estimate['probability']>0 else 1,-score,meta['deadline'],g.arrival_time)
        chain_selected,chain_used,rows=choose(goals,capacity-fg_used,slots-len(fg_selected),self.cache_config.block_size,protected)
        selected=fg_selected|chain_selected;used=fg_used+chain_used
        # Borrow unused chain budget for foreground requests without changing chain reward.
        for g in fg_sorted:
            if g.request_id in selected:continue
            charge=blocks(self.view(g),self.cache_config.block_size)
            if used+charge<=capacity and len(selected)<slots:selected.add(g.request_id);used+=charge
        exception=None
        if not selected and groups:
            # Progress fallback only for physically representable current contexts.
            # The full envelope is disclosed; this is never an uncharged admission.
            for g in sorted(groups,key=self.policy.get_priority):
                v=self.view(g);current=math.ceil((v['prompt']+v['generated'])/self.cache_config.block_size)
                if current<=self.block_manager.num_total_gpu_blocks:
                    selected={g.request_id};used=blocks(v,self.cache_config.block_size);exception={'request':g.request_id,'current_blocks':current,'full_envelope':used};break
            if not selected:raise RuntimeError('No physically representable request; preserve invalid input contract')
        self.rows=rows;self.selected=selected;self.last_recorded_selected=frozenset(selected);self.last_plan=self.policy.num_schedule_count
        self.emit('plan',tick=self.last_plan,capacity_blocks=capacity,sequence_limit=slots,charged_blocks=used,
          foreground_quota_blocks=quota,foreground_selected=sorted(fg_selected),protected_candidates=sorted(protected),
          selected=sorted(selected),goals=[{k:v for k,v in g.items()if k!='members'}for g in goals],cohorts=rows,
          physical_progress_exception=exception,decode_step_seconds=self.cadence.token_seconds,
          transfer_seconds_per_block=self.cadence.transfer_per_block,
          member_states=[{'request':g.request_id,'chain':g.collection_id,'stage':g.request_info.stage_id,
             'status':g.first_seq.status.name,**self.view(g)}for g in groups])
    def _schedule(self):
        self.policy.update_schedule_count();self.free_finished_seq_groups();now=time.perf_counter()
        groups=[g for g in list(self.running)+list(self.waiting)+list(self.swapped)if not g.is_finished()]
        live={g.request_id for g in groups};protected=self.protected(groups,now)
        phase='main'if any(g.request_info.client_id!=900000 for g in groups)else'warmup'
        self.mixed_counts[phase+'_scheduler_calls']+=1
        self.mixed_counts[phase+'_joint_calls']+=1
        self.mixed_counts[phase+'_with_foreground_calls']+=int(any(g.request_type!=RequestType.COLLECTIVE for g in groups))
        plan_key=tuple(sorted((g.request_id,g.first_seq.status.name,g.first_seq.get_output_len()//50,g.request_id in protected,
                              self.policy.workflow_deadline(g)>now)for g in groups))
        structural_change=live!=self.last_live
        if self.policy.num_schedule_count-self.last_plan>=50 or structural_change or plan_key!=self.last_plan_key:
            self.plan(groups,now,protected);self.last_plan_key=plan_key
        self.last_live=live
        selected=self.selected&live;swaps=[];count=0
        for g in sorted(list(self.running),key=self.policy.get_priority,reverse=True):
            if g.request_id in selected or count>=self.max_swaps_per_iter:continue
            mode=PreemptionMode.SWAP if self.block_manager.can_swap_out(g)else PreemptionMode.RECOMPUTE
            mode=self._preempt(g,swaps,mode);self.running.remove(g);self.leases.pop(g.request_id,None)
            (self.swapped if mode==PreemptionMode.SWAP else self.waiting).append(g);count+=1
            self.emit('proactive_preemption',request=g.request_id,chain=g.collection_id,stage=g.request_info.stage_id,
              mode=mode.name,priority=self.policy.get_priority(g),generated=g.first_seq.get_output_len())
        if count:return SchedulerOutputs([],0,0,[],swaps,[],[],0,len(self.running),count)
        waiting=[g for g in self.waiting if g.request_id not in selected];swapped=[g for g in self.swapped if g.request_id not in selected]
        self.waiting=deque(sorted((g for g in self.waiting if g.request_id in selected),key=self.policy.get_priority))
        self.swapped=deque(sorted((g for g in self.swapped if g.request_id in selected),key=self.policy.get_priority))
        self.running=deque(sorted(self.running,key=self.policy.get_priority))
        try:return self._schedule_chunked_prefill()
        finally:self.waiting.extend(waiting);self.swapped.extend(swapped)

def install(server):
    import vllm.engine.llm_engine as engine
    engine.SLOScheduler=WorkflowScheduler
    original=server._handle_collective_request
    def collective(info,request_dict,prompt):
        stages=request_dict.get('num_stages',6);widths=list(request_dict.get('requests_per_stage')or[1]*stages)
        if stages<1 or len(widths)!=stages or any(not isinstance(w,int)or w<1 for w in widths):raise ValueError('Invalid declared workflow structure')
        if info.stage_id<0 or info.stage_id>=stages or info.request_id<0 or info.request_id>=widths[info.stage_id]:raise ValueError('Invalid workflow member identity')
        info.chain_deadline=time.perf_counter()+info.slo_constraint[2]*stages-request_dict.get('served_time',0.)
        info.chain_stage_count=stages;info.chain_stage_widths=widths
        return original(info,request_dict,prompt)
    server._handle_collective_request=collective
