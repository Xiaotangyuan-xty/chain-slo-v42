"""v47pro: v47 root reward with event replanning and isolated physical execution."""
from collections import Counter, deque
from copy import copy
import math
from replan_events import ReplanEvents
from service_model import ServiceModel
import heapq
from pathlib import Path
import time

from vllm.core.scheduler import PreemptionMode,SchedulerOutputs
from vllm.sequence import SequenceStatus
from jitserve.scheduler.policy import SLOPolicy
from jitserve.scheduler.slo_scheduler import SLOScheduler
from chain_forecast import StageForecaster
from chain_admission import select,blocks
from token_reward import REWARD

MODEL=StageForecaster.load(Path(__file__).resolve().parents[1]/'assets/CHAIN_MODEL.json')

class ChainCompletionPolicy(SLOPolicy):
    version='v47pro_event_wave_service_v1'
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
        meta['future_max_tokens']=getattr(info,'chain_future_max_tokens',meta.get('future_max_tokens',1024))
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
    def goals(self,groups,now,capacity=1000000,slots=32,block_size=16,token_budget=2048):
        grouped={}
        for group in groups:grouped.setdefault(group.collection_id,[]).append(group)
        result=[];seconds=max(.000001,self._get_interval_time());self.priorities={}
        for cid,gs in grouped.items():
            stage=gs[0].request_info.stage_id
            if any(g.request_info.stage_id!=stage for g in gs):raise RuntimeError('Concurrent stages in one chain')
            meta=self.chains[cid]
            views=[{'request':g.request_id,'member':g.request_info.request_id,'declared_input':g.request_info.input_len,
                    'prompt':g.first_seq.get_prompt_len(),'generated':g.first_seq.get_output_len(),'max_tokens':g.sampling_params.max_tokens,
                    'prefill_tokens':g.get_num_uncomputed_tokens() if g.is_prefill() else 0,'arrival':g.arrival_time,
                    'reward_input':getattr(g.request_info,'chain_reward_input_tokens',g.first_seq.get_prompt_len()),
                    'resident':g.first_seq.status==SequenceStatus.RUNNING,
                    'swapped':g.first_seq.status==SequenceStatus.SWAPPED} for g in gs]
            estimate=MODEL.estimate(meta,stage,views,meta['deadline']-now,seconds,service=getattr(self,'service_model',None),capacity=capacity,slots=slots,block_size=block_size,token_budget=token_budget)
            incumbent=any(g.first_seq.status==SequenceStatus.RUNNING for g in gs)
            score=estimate['score']*(1.2 if incumbent else 1.)
            goal={**estimate,'chain':cid,'stage':stage,'score':score,'deadline':meta['deadline'],
                  'arrival':min(g.arrival_time for g in gs),'members':views}
            for g in gs:self.priorities[g.request_id]=(0 if estimate['probability']>0 else 1,-score,meta['deadline'],g.arrival_time)
            result.append(goal)
        return result

class NativeChainScheduler(SLOScheduler):
    def __init__(self,scheduler_config,cache_config,lora_config,*args,**kwargs):
        # Native adjustment must not mutate the engine's shared config object.
        super().__init__(copy(scheduler_config),cache_config,lora_config,*args,**kwargs)
        self.policy=ChainCompletionPolicy();self.cost_model=ServiceModel(self.scheduler_config.max_num_batched_tokens);self.policy.service_model=self.cost_model
        self.base_max_num_seqs=scheduler_config.max_num_seqs
        self.selected=set();self.desired_selected=set();self.rows=[];self.leases={};self.reservation_events=[]
        self.last_plan=-50;self.last_members=None;self.last_recorded_selected=None
        self.fast_expired_drain=False;self.drain_key=None;self.v47_counts=Counter()
        self.events=ReplanEvents();self.mode=None;self.mode_epoch=0;self.last_joint_summary=None
        self.last_start=None;self.last_cycle_ids=set();self.last_nonempty=False;self.decision_sink=None;self.decision_id=0
        self._planned_this_step=False;self._lease_progress={};self.mask_id=0;self._logged_execution_mask=None
        self.last_protected=set()

    def _clear_plan(self):
        self.selected=set();self.desired_selected=set();self.rows=[];self.leases.clear();self._lease_progress.clear()
        self.last_members=None;self.last_recorded_selected=None;self.last_plan=-50
        self.policy.priorities.clear();self.policy.seq_group_slo_dict.clear()
        self._pending_swap_out_queue.clear();self._current_bin_queue.clear()
        self._exec_order=[];self._bins_by_index=[];self._bin_bounds=[]
        self._current_bounds=(0,0);self._current_is_last=False;self._cur_rank=0
        self.prev_no_swap_space=False;self.dull_iteration=0
        self.fast_expired_drain=False;self.drain_key=None;self.last_joint_summary=None;self._logged_execution_mask=None
        self.last_protected=set()

    def _enter_mode(self,mode):
        if self.mode!=mode:
            previous=self.mode;self._clear_plan();self.mode=mode;self.mode_epoch+=1
            now=time.perf_counter()
            for g in self.running:self.leases[g.request_id]={'generated':g.first_seq.get_output_len(),'started':now}
            self.events.invalidate('mode_epoch');self.v47_counts['mode_changes']+=1
            self._emit({'kind':'mode_change','previous':previous,'mode':mode})

    def _emit(self,event):
        if self.decision_sink:
            self.decision_sink({'monotonic':time.perf_counter(),'mode_epoch':self.mode_epoch,
                                'decision_id':self.decision_id,**event})

    def _stop_dull_scheduling(self):
        # Preserve pending requests during stalls; controller timeout preserves
        # the failure rather than turning idle rounds into request deletion.
        if not self.running and (self.waiting or self.swapped):self.v47_counts['idle_with_pending']+=1
        return []

    def _adjust_batch_limits(self,process_list,min_tbt_required,budget):
        if self.search_strategy!='sliding_window':return
        limit=max(self.max_batch_size,len(process_list))
        if self.policy.interval_time is not None and self.policy.interval_time>min_tbt_required:
            limit=max(1,limit-self.max_swaps_per_iter)
        budget.max_num_seqs=max(len(self.running),limit)
        # No write to scheduler_config. Pure/joint limits use base_max_num_seqs.

    def _chunked(self,limit=None):
        self._execution_flags.update(mode=self.mode,mode_epoch=self.mode_epoch,decision_id=self.decision_id)
        config=self.scheduler_config
        local=copy(config);local.max_num_seqs=limit or self.base_max_num_seqs
        self.scheduler_config=local
        try:return self._schedule_chunked_prefill()
        finally:self.scheduler_config=config

    def _preempt(self,seq_group,blocks_to_swap_out,preemption_mode=None):
        group=seq_group;swaps=blocks_to_swap_out;mode=preemption_mode
        # All physical/native/proactive preemption paths pass this boundary.
        self.leases.pop(group.request_id,None);self._lease_progress.pop(group.request_id,None)
        if self.selected is not None:self.selected.discard(group.request_id)
        if self.desired_selected is not None:self.desired_selected.discard(group.request_id)
        self.events.invalidate('physical_preemption');self.v47_counts['all_preemptions']+=1
        actual=super()._preempt(group,swaps,mode)
        self._emit({'kind':'preempt','request':group.request_id,'preemption':str(actual)})
        return actual

    def _expired_limit(self,groups):
        capacity=self.block_manager.num_total_gpu_blocks-self.block_manager.watermark_blocks
        used=slots=0
        for g in list(self.running)+list(self.swapped)+list(self.waiting):
            member=self._view(g);charge=blocks(member,self.cache_config.block_size)
            if slots>=max(self.base_max_num_seqs,min(128,self.scheduler_config.max_num_batched_tokens)):break
            if used+charge>capacity:break
            used+=charge;slots+=1
        return max(self.base_max_num_seqs,slots)

    def _view(self,g):
        return {'request':g.request_id,'prompt':g.first_seq.get_prompt_len(),
                'generated':g.first_seq.get_output_len(),'max_tokens':g.sampling_params.max_tokens,
                'prefill_tokens':g.get_num_uncomputed_tokens() if g.is_prefill() else 0,
                'arrival':g.arrival_time,'resident':g.first_seq.status==SequenceStatus.RUNNING,
                'swapped':g.first_seq.status==SequenceStatus.SWAPPED}

    def _begin(self):
        now=time.perf_counter()
        previous_start=self.last_start
        self.last_start=now;self._planned_this_step=False
        self._execution_flags={'decision_id':self.decision_id,'mode_epoch':self.mode_epoch,'candidate_filter':False,'mask_id':None}
        self.cost_model.set_token_budget(self.scheduler_config.max_num_batched_tokens)
        self.policy.update_schedule_count();self.free_finished_seq_groups()
        groups=[g for g in list(self.running)+list(self.waiting)+list(self.swapped) if not g.is_finished()]
        live={g.request_id for g in groups}
        if previous_start is not None and live&self.last_cycle_ids:self.cost_model.observe_round(now-previous_start)
        self.last_cycle_ids=live
        for rid in list(self.leases):
            if rid not in live:self.leases.pop(rid,None);self._lease_progress.pop(rid,None)
        for g in self.running:
            rid=g.request_id;n=g.first_seq.get_output_len()
            self.leases.setdefault(rid,{'generated':n,'started':now})
        self.last_nonempty=bool(groups)
        n=max(1,len(self.running));context=sum(g.first_seq.get_len() for g in self.running)
        self.policy.interval_time=self.cost_model.unit(n,context)
        return groups,now

    def _deadline(self,g,now,wall):
        if hasattr(g.request_info,'chain_deadline'):return g.request_info.chain_deadline
        if g.request_type.value==0:
            return now+g.arrival_time+g.TTFT_constraint+g.first_seq.get_output_len()*g.TBT_constraint-wall
        return now+g.arrival_time+g.deadline-wall

    def _protected(self,now):
        wall=time.time();protected=set()
        for g in self.running:
            lease=self.leases[g.request_id]
            if (self._deadline(g,now,wall)>now and now-lease['started']<1.5
                    and g.first_seq.get_output_len()-lease['generated']<50):protected.add(g.request_id)
        return protected

    def _event_state(self,groups,now):
        wall=time.time();deadlines=[];lead=max(.01,self.cost_model.last_round)
        for g in groups:
            root=self._deadline(g,now,wall);deadlines.extend([root-lead,root])
            if not hasattr(g.request_info,'chain_deadline'):
                n=g.first_seq.get_output_len();ttft=now+g.arrival_time+g.TTFT_constraint-wall
                if not n:deadlines.extend([ttft-lead,ttft])
                elif g.request_type.value==0:
                    token=ttft+n*g.TBT_constraint;deadlines.extend([token-lead,token])
            if g.request_id in self.leases:
                deadlines.append(self.leases[g.request_id]['started']+1.5)
        signature={
            'membership_stage':tuple(sorted((g.request_id,g.collection_id,g.request_info.stage_id) for g in groups)),
            'prediction_revision':tuple(sorted((g.request_id,getattr(g.request_info,'prediction_revision',0),g.request_info.output_len) for g in groups)),
            'physical_state':tuple(tuple(sorted((g.request_id,g.is_prefill()) for g in queue)) for queue in [self.running,self.waiting,self.swapped]),
            'free_capacity':(self.block_manager.get_num_free_gpu_blocks(),self.block_manager.get_num_free_cpu_blocks()),
            'configuration':(self.base_max_num_seqs,self.scheduler_config.max_num_seqs,self.scheduler_config.max_num_batched_tokens,self.cache_config.block_size,self.block_manager.num_total_gpu_blocks,self.block_manager.watermark_blocks),
            'service_quantum':tuple(sorted((g.request_id,(g.first_seq.get_output_len()-self.leases[g.request_id]['generated'])//50) for g in self.running)),
            'mode_epoch':self.mode_epoch,
        }
        return signature,deadlines

    def _choose_preemption(self,g):
        m=self._view(g)
        recompute=math.ceil((m['prompt']+m['generated'])/self.scheduler_config.max_num_batched_tokens)*self.cost_model.unit(1,m['prompt']+m['generated'],min(m['prompt']+m['generated'],self.scheduler_config.max_num_batched_tokens))
        swap=2*self.cost_model.unit(1,m['prompt']+m['generated'],0,True)
        if g.get_max_num_running_seqs()!=1 or (swap<recompute and self.block_manager.can_swap_out(g)):
            return PreemptionMode.SWAP if self.block_manager.can_swap_out(g) else PreemptionMode.RECOMPUTE
        return PreemptionMode.RECOMPUTE

    def _blocked(self,target):
        from vllm.core.interfaces import AllocStatus
        if len(self.running)>=self.base_max_num_seqs:return True
        if target in self.swapped:return self.block_manager.can_swap_in(target,0)!=AllocStatus.OK
        return self.block_manager.can_allocate(target)!=AllocStatus.OK

    def _execute_selected(self,selected):
        swaps=[];preemptions=0
        # Selected work and the current lease are protected; other residents
        # provide bounded turnover without requiring a blocked admission target.
        victims=[g for g in self.running if g.request_id not in selected and g.request_id not in self.last_protected]
        for victim in sorted(victims,key=self.policy.get_priority,reverse=True):
            if preemptions>=self.max_swaps_per_iter:break
            mode=self._preempt(victim,swaps,self._choose_preemption(victim))
            self.running.remove(victim)
            (self.swapped if mode==PreemptionMode.SWAP else self.waiting).append(victim)
            preemptions+=1
        effective=set(selected)|{g.request_id for g in self.running}
        self.selected=effective
        self.last_recorded_selected=frozenset(effective)
        mask=frozenset(effective)
        if mask!=self._logged_execution_mask:
            self._logged_execution_mask=mask;self.mask_id+=1
            self._emit({'kind':'execution_mask','mask_id':self.mask_id,'desired':sorted(selected),'allowed':sorted(mask),
                        'retained_residents':sorted(mask-set(selected))})
        self._execution_flags={'decision_id':self.decision_id,'mode_epoch':self.mode_epoch,'mode':self.mode,
                               'candidate_filter':True,'mask_id':self.mask_id}
        if swaps:
            self.v47_counts['swap_dependency_rounds']+=1
            return SchedulerOutputs([],0,0,[],swaps,[],[],0,len(self.running),preemptions)
        hidden_waiting=[g for g in self.waiting if g.request_id not in selected]
        hidden_swapped=[g for g in self.swapped if g.request_id not in selected]
        self.waiting=deque(sorted((g for g in self.waiting if g.request_id in selected),key=self.policy.get_priority))
        self.swapped=deque(sorted((g for g in self.swapped if g.request_id in selected),key=self.policy.get_priority))
        self.running=deque(sorted(self.running,key=self.policy.get_priority))
        try:
            out=self._chunked()
            out.preempted+=preemptions
            if preemptions:self.v47_counts['same_round_after_recompute']+=bool(out.scheduled_seq_groups)
            return out
        finally:self.waiting.extend(hidden_waiting);self.swapped.extend(hidden_swapped)

    def _plan_loop(self,groups,now,joint=False):
        live={g.request_id for g in groups};tick=self.policy.num_schedule_count
        chains=[g for g in groups if hasattr(g.request_info,'chain_deadline')]
        foreground=[g for g in groups if not hasattr(g.request_info,'chain_deadline')]
        signature,deadlines=self._event_state(groups,now)
        coalesce=max(.01,min(.05,self.cost_model.last_round))
        reasons=self.events.reasons(signature,tick,now,coalesce)
        protected=self._protected(now) if reasons else set()
        if reasons==['service_quantum'] and protected==self.last_protected:
            reasons=[];self.events.counts['quantum_without_protection_change']+=1
        if reasons:
            residents=[self._view(g) for g in self.running]
            capacity=self.block_manager.num_total_gpu_blocks-self.block_manager.watermark_blocks
            goals=self.policy.goals(chains,now,capacity,self.base_max_num_seqs,self.cache_config.block_size,self.scheduler_config.max_num_batched_tokens) if chains else []
            # Diagnostic only: preserve initial forecasts but skip selected and
            # final resident/service revaluation in both pure and joint modes.
            if joint:
                from joint_admission import select_joint
                fg_sorted=sorted(foreground,key=self.policy.get_priority)
                self.selected,self.rows,fg=select_joint(goals,[self._view(g) for g in fg_sorted],capacity,self.base_max_num_seqs,self.cache_config.block_size,protected,forecaster=None,residents=residents)
                self.last_joint_summary={'tick':tick,'chain_goals':len(goals),'foreground_selected':len(fg),'reasons':reasons}
            else:
                self.selected,self.rows=select(goals,capacity,self.base_max_num_seqs,self.cache_config.block_size,protected,forecaster=None,residents=residents)
            self.desired_selected=None if self.selected is None else set(self.selected)
            self.policy.draining=self.selected is None;self.last_plan=tick;self._planned_this_step=True;self.decision_id+=1
            self.last_protected=protected
            committed=time.perf_counter()
            self.events.commit(signature,tick,committed,deadlines,reasons,waiting=any(g not in self.running for g in groups),started=now)
            self.v47_counts['plans']+=1
            self._emit({'kind':'plan','mode':self.mode,'tick':tick,'reasons':reasons,'next_due':self.events.next_due,
                        'selected':sorted(self.selected) if self.selected is not None else None,'protected':sorted(protected),'rows':self.rows,
                        'running':[g.request_id for g in self.running],'waiting':len(self.waiting),'swapped':len(self.swapped)})
        selected=live if self.desired_selected is None else set(self.desired_selected)&live
        self.last_members=live;self.last_recorded_selected=frozenset(selected)
        return self._execute_selected(selected) if self.desired_selected is not None else self._chunked()

    def _schedule(self):
        groups,now=self._begin()
        mode='expired_drain' if groups and not self.policy.has_live_credit(now) else 'pure'
        self._enter_mode(mode)
        if mode=='expired_drain':
            self.fast_expired_drain=True;self.policy.draining=True;self.selected=None
            self.v47_counts['expired_drain_ticks']+=1
            return self._chunked(self._expired_limit(groups))
        return self._plan_loop(groups,now)


def install(server):
    import vllm.engine.llm_engine as llm_engine
    llm_engine.SLOScheduler=NativeChainScheduler
    original=server._handle_collective_request
    def collective(info,request_dict,prompt):
        stages=request_dict.get('num_stages',6)
        info.chain_deadline=time.perf_counter()+info.slo_constraint[2]*stages-request_dict.get('served_time',0.)
        info.chain_stage_count=stages;info.chain_stage_widths=list(request_dict.get('requests_per_stage') or [1]*stages)
        tokenizer=getattr(getattr(server,'graph_ctx',None),'generation_tokenizer',None);REWARD.bind(tokenizer)
        from frozen_v47.token_reward import REWARD as FROZEN_REWARD
        FROZEN_REWARD.bind(tokenizer)
        if tokenizer is not None:info.chain_reward_input_tokens=REWARD.count(prompt,getattr(info,'input_len',0))
        return original(info,request_dict,prompt)
    server._handle_collective_request=collective
