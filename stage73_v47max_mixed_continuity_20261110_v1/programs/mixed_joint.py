"""v47 chain reward/coexistence with event invalidation and isolated modes."""
from collections import Counter
import time
from jitserve.request_info import RequestType
from vllm.sequence import SequenceStatus
from jitserve.scheduler.policy import SLOPolicy
from native_chain import ChainCompletionPolicy,NativeChainScheduler
from pure_core_v47 import leave_pure,schedule_pure

VERSION='v47pro_event_wave_service_v1'
VARIANT='v47pro_v47_reward_event_wave_v1'


def is_chain(group):return group.request_type==RequestType.COLLECTIVE


class MixedPolicy(ChainCompletionPolicy):
    version=VERSION;mixed_variant=VARIANT
    def __init__(self):super().__init__();self.joint_mode=False
    def add_seq_group(self,seq_group):
        group=seq_group
        if is_chain(group):return super().add_seq_group(group)
        self.membership_epoch+=1
        return SLOPolicy.add_seq_group(self,group)
    def delete_seq_group(self,seq_group):
        group=seq_group
        if is_chain(group):return super().delete_seq_group(group)
        self.membership_epoch+=1;self.seq_group_slo_dict.pop(group.request_id,None)
        return SLOPolicy.delete_seq_group(self,group)
    def get_priority(self,seq_group):
        group=seq_group
        if is_chain(group):
            priority=super().get_priority(group)
            return (1,)+priority if self.joint_mode else priority
        # Preserve v47's foreground reward formula, but charge recovery/prefill
        # service to foreground too. This is not a candidate-only QRF change.
        reward,remain=SLOPolicy.reward_estimation(self,group)
        view={'prompt':group.first_seq.get_prompt_len(),'generated':group.first_seq.get_output_len(),
              'prefill_tokens':group.get_num_uncomputed_tokens() if group.is_prefill() else 0,
              'resident':group.first_seq.status==SequenceStatus.RUNNING,'swapped':group.first_seq.status==SequenceStatus.SWAPPED}
        recovery=self.service_model.recovery(view)
        priority=(-reward/max(.000001,remain+recovery),remain+recovery,group.arrival_time)
        return (0,)+priority if self.joint_mode else priority


class MixedScheduler(NativeChainScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.policy=MixedPolicy();self.policy.service_model=self.cost_model
        self.mixed_counts=Counter()
    def add_seq_group(self,seq_group):
        if not is_chain(seq_group) and getattr(self,'_pure_core_active',False):leave_pure(self)
        return super().add_seq_group(seq_group)
    def _restore_drain_limits(self):
        from frozen_v47.native_chain import NativeChainScheduler as frozen
        return frozen._restore_drain_limits(self)
    def _expired_drain(self,groups,live,tick):
        from frozen_v47.native_chain import NativeChainScheduler as frozen
        return frozen._expired_drain(self,groups,live,tick)
    def _schedule(self):
        queued=list(self.running)+list(self.waiting)+list(self.swapped)
        if all(is_chain(g) for g in queued):return schedule_pure(self)
        leave_pure(self)
        groups,now=self._begin();chains=[g for g in groups if is_chain(g)];foreground=[g for g in groups if not is_chain(g)]
        if foreground:mode='joint' if chains else 'foreground'
        else:mode='expired_drain' if groups and not self.policy.has_live_credit(now) else 'pure'
        self._enter_mode(mode);self.policy.joint_mode=mode=='joint'
        self._execution_flags.update(mode=mode,mode_epoch=self.mode_epoch,decision_id=self.decision_id)
        self.policy.physical_running=self.running;self.policy.physical_swapped=self.swapped
        self.mixed_counts[mode+'_calls']+=1
        if mode=='foreground':
            # No v47 candidate selection filter owns this native-only mode.
            self.policy.draining=True;self.selected=None;self.desired_selected=None
            self.last_recorded_selected=None
            signature,deadlines=self._event_state(groups,now)
            coalesce=max(.01,min(.05,self.cost_model.last_round))
            reasons=self.events.reasons(signature,self.policy.num_schedule_count,now,coalesce)
            if reasons:
                self.policy.seq_group_slo_dict.clear();self.last_plan=self.policy.num_schedule_count;self._planned_this_step=True
                if self._test_enough_swap_space():
                    result=self._schedule_slo()
                    self.events.commit(signature,self.last_plan,time.perf_counter(),deadlines,reasons,waiting=True,started=now)
                    return result
                self.events.commit(signature,self.last_plan,time.perf_counter(),deadlines,reasons,waiting=True,started=now)
            return self._chunked()
        if mode=='expired_drain':
            self.fast_expired_drain=True;self.policy.draining=True;self.selected=None
            self.v47_counts['expired_drain_ticks']+=1
            return self._chunked(self._expired_limit(groups))
        return self._plan_loop(groups,now,mode=='joint')
