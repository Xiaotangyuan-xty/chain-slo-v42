"""Read-only lifetime token timing, independent of native recompute resets."""
import atexit
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter as _clock, time as _wall

VERSION='lifetime_token_append_v1'

class LifetimeObservation:
    def __init__(self,output):
        self.output=Path(output);self.stream=(self.output/'LIFETIME_TIMINGS.jsonl').open('x')
        self.counts=Counter();self.closed=False;self.originals=None

    def appended(self,seq,before):
        after=seq.get_output_len()
        if after!=before+1:raise RuntimeError('Lifetime observer requires one appended token per callback')
        state=getattr(seq,'_lifetime_observation',None)
        if state is None:
            if before!=0:raise RuntimeError('Lifetime timing started after generation')
            state={'monotonic':[],'first_wall':_wall(),'recomputes':[],'exported':False}
            seq._lifetime_observation=state
        if len(state['monotonic'])!=before:raise RuntimeError('Missing or duplicate lifetime token capture')
        state['monotonic'].append(_clock());self.counts['tokens']+=1

    def exported(self,group,result):
        if result is None or not result.finished:return result
        if len(result.outputs)!=1:raise RuntimeError('This frozen observer supports n=1 only')
        seq=group.first_seq;state=getattr(seq,'_lifetime_observation',None)
        count=len(result.outputs[0].token_ids)
        if state is None or len(state['monotonic'])!=count or seq.get_output_len()!=count:
            raise RuntimeError('Final output and lifetime token timing counts differ')
        stamps=state['monotonic'];intervals=[b-a for a,b in zip(stamps,stamps[1:])]
        if any(x<0 for x in intervals):raise RuntimeError('Non-monotonic lifetime timestamp')
        ttft=state['first_wall']-group.concord_metrics.arrival_time
        if ttft<0:raise RuntimeError('Negative lifetime TTFT')
        if not state['exported']:
            info=group.request_info
            row={'schema':VERSION,'phase':'warmup'if info.client_id==900000 else'main',
                 'collection_id':info.collection_id,'stage_id':info.stage_id,'request_id':info.request_id,
                 'request_uuid':group.request_id,'output_tokens':count,
                 'arrival_wall':group.concord_metrics.arrival_time,'first_token_wall':state['first_wall'],
                 'token_append_monotonic':stamps,'TTFT':ttft,'TBT':intervals,
                 'native_final_TTFT':result.ttft,'native_final_TBT':list(result.tbt or []),
                 'recompute_prefix_counts':state['recomputes']}
            self.stream.write(json.dumps(row,allow_nan=False)+'\n');self.stream.flush()
            state['exported']=True;self.counts['requests']+=1;self.counts['intervals']+=len(intervals)
            self.counts['requests_with_recompute']+=int(bool(state['recomputes']))
        # Only serialized output fields change. Native policy metrics are intact.
        result.ttft=ttft;result.tbt=intervals
        return result

    def close(self):
        if self.closed:return
        self.stream.close();self.closed=True
        if self.output.exists():
            (self.output/'LIFETIME_OBSERVATION_SUMMARY.json').write_text(json.dumps({'version':VERSION,'counts':dict(self.counts),
                'native_policy_metrics_unchanged':True,'timestamps':'CPU token-append monotonic timestamps; TTFT from first token wall time and native request arrival.'},indent=2)+'\n')

    def restore_for_cpu_test(self):
        from vllm.sequence import Sequence
        from vllm.outputs import RequestOutput
        from vllm.core.scheduler import Scheduler
        Sequence.append_token_id,RequestOutput.from_seq_group,Scheduler._preempt_by_recompute=self.originals
        del Sequence._stage20_lifetime_installed
        self.close();atexit.unregister(self.close)

def install(root,stage,output):
    from vllm.sequence import Sequence
    from vllm.outputs import RequestOutput
    from vllm.core.scheduler import Scheduler
    if getattr(Sequence,'_stage20_lifetime_installed',False):raise RuntimeError('Lifetime observer installed twice')
    expected=json.loads((Path(stage)/'PROTOCOL.json').read_text())['lifetime_observation_sha256']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==expected
    observer=LifetimeObservation(output)
    append=Sequence.append_token_id;descriptor=RequestOutput.__dict__['from_seq_group'];recompute=Scheduler._preempt_by_recompute
    observer.originals=(append,descriptor,recompute)
    def append_token_id(self,*args,**kwargs):
        before=self.get_output_len();result=append(self,*args,**kwargs)
        observer.appended(self,before);return result
    def from_seq_group(cls,seq_group,*args,**kwargs):
        result=descriptor.__func__(cls,seq_group,*args,**kwargs)
        return observer.exported(seq_group,result)
    def preempt_by_recompute(self,group,*args,**kwargs):
        seq=group.first_seq;before=seq.get_output_len();state=getattr(seq,'_lifetime_observation',None)
        if state is not None:state['recomputes'].append(before)
        result=recompute(self,group,*args,**kwargs)
        if seq.get_output_len()!=before:raise RuntimeError('Native recompute changed emitted output history')
        observer.counts['recompute_resets']+=1
        return result
    Sequence.append_token_id=append_token_id;RequestOutput.from_seq_group=classmethod(from_seq_group)
    Scheduler._preempt_by_recompute=preempt_by_recompute;Sequence._stage20_lifetime_installed=True
    receipt={'version':VERSION,'module_sha256':expected,'native_source_bytes_changed':False,
             'native_metrics_reset_or_policy_logic_changed':False,'same_observer_for_A_C':True,
             'recompute_handling':'Keep a separate per-sequence emitted-token history; retain preemption gap; never reset or reconstruct missing timestamps.',
             'export':'Override only final RequestOutput TTFT/TBT; preserve native final metrics in a separate log.',
             'sources':{name:{'path':inspect.getfile(obj),'sha256':hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()}
                        for name,obj in [('sequence',Sequence),('outputs',RequestOutput),('scheduler',Scheduler)]}}
    (Path(output)/'LIFETIME_OBSERVER_RUNTIME.json').write_text(json.dumps(receipt,indent=2)+'\n')
    atexit.register(observer.close)
    return observer
