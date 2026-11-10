"""Fresh v47-derived process only. Installing never starts a model or GPU job."""
import atexit
import functools
import json
import queue
import threading
import time
from pathlib import Path


class _DecisionWriter:
    def __init__(self,stream,queue_size=4096,batch_size=256,flush_interval=1.):
        self.stream=stream;self.queue=queue.Queue(maxsize=queue_size);self.batch_size=batch_size;self.flush_interval=flush_interval
        self.lock=threading.Lock();self.closed=False;self.stop=object()
        self.counts={'queued':0,'written':0,'dropped':0,'writer_errors':0}
        self.thread=threading.Thread(target=self._run,name='v47pro-decision-writer',daemon=True);self.thread.start()

    def enqueue(self,scheduler,event):
        with self.lock:
            if self.closed:self.counts['dropped']+=1;return
            try:self.queue.put_nowait((scheduler,event))
            except queue.Full:self.counts['dropped']+=1
            else:self.counts['queued']+=1

    def snapshot(self):
        with self.lock:return dict(self.counts)

    def _error(self):
        with self.lock:self.counts['writer_errors']+=1

    def _write(self,batch):
        if not batch:return
        try:self.stream.write(''.join(batch))
        except Exception:self._error()
        else:
            with self.lock:self.counts['written']+=len(batch)

    def _flush(self):
        try:self.stream.flush()
        except Exception:self._error()

    def _run(self):
        batch=[];deadline=time.monotonic()+self.flush_interval
        try:
            while True:
                try:item=self.queue.get(timeout=max(0.,deadline-time.monotonic()))
                except queue.Empty:item=None
                if item is self.stop:break
                if item is not None:
                    try:
                        scheduler,event=item;batch.append(json.dumps({'scheduler':scheduler,**event},allow_nan=False)+'\n')
                    except Exception:self._error()
                if len(batch)>=self.batch_size:self._write(batch);batch=[]
                now=time.monotonic()
                if now>=deadline:
                    self._write(batch);batch=[];self._flush();deadline=now+self.flush_interval
        finally:
            self._write(batch);self._flush()
            try:self.stream.close()
            except Exception:self._error()

    def close(self):
        with self.lock:
            stopping=not self.closed;self.closed=True
        if stopping:self.queue.put(self.stop)
        self.thread.join()


def install(server,output,*,model_key='llama8',shared_prediction_config=None):
    from prediction_transport import PROTOCOL
    output=Path(output)
    if not output.is_dir() or (output/'V47PRO_RUNTIME.json').exists():raise ValueError('Fresh existing output directory required')
    if getattr(server,'_v47pro_installed',False):raise RuntimeError('Fresh server process required')
    if shared_prediction_config is not None and shared_prediction_config.get('predictor_protocol')!=PROTOCOL:raise ValueError('Matching framed prediction required')
    import native_chain
    from mixed_joint import MixedScheduler,VERSION,VARIANT
    import pure_core_v47
    from native_tot_adapter import install_server_annotations
    from token_reward import REWARD
    import vllm.engine.llm_engine as engine
    if getattr(MixedScheduler,'_v47pro_installed',False):raise RuntimeError('Fresh scheduler process required')
    server._v47pro_installed=True;MixedScheduler._v47pro_installed=True
    native_chain.install(server);annotations=install_server_annotations(server)
    engine.SLOScheduler=MixedScheduler
    stream=(output/'V47PRO_DECISIONS.jsonl').open('x',buffering=1024*1024)
    decisions=_DecisionWriter(stream);schedulers=[];closed=False
    previous_init=MixedScheduler.__init__
    @functools.wraps(previous_init)
    def init(self,*args,**kwargs):
        previous_init(self,*args,**kwargs);index=len(schedulers);schedulers.append(self)
        self.decision_sink=functools.partial(decisions.enqueue,index)
    MixedScheduler.__init__=init
    shared=None
    if shared_prediction_config is not None:
        from shared_prediction import install as install_shared
        from jitserve.scheduler.policy import SLOPolicy
        shared=install_shared(server,SLOPolicy,output,method='v47pro',**shared_prediction_config)
    from execution_observation import install as observe
    receipt={'version':VERSION,'variant':VARIANT,'scheduling_parent':'frozen v47 stage30, not v48/v49/v50/v50pro',
             'v47_source_commit':'082e98c7c05431110bd6cd74234f909487b963ee','model_key':model_key,'stage_budget_adapter':False,
             'native_ToT_annotations':annotations,'shared_prediction':shared,'prediction_comparison_ready':shared is not None,
             'execution_observer':observe(output),'installer_allocates_GPU':False,'installer_runs_model':False,
             'future_truth_access':False,'request_drops':False,'retries':False,
             'main_metric':'settlement-time fixed-hour weighted credit/3600; full replay/drain separately',
             'score':'v47 conditional whole-root credit / remaining service work; conservative capacity waves',
             'foreground':'v47 native foreground reward formula retained; recovery cost and leases apply to foreground too',
             'mode_limit':'initial independent sequence cap; native adjustments affect only this scheduling budget'}
    def write(final=False):
        nonlocal closed
        if closed:return
        if final:closed=True;decisions.close()
        data={**receipt,'decision_log':decisions.snapshot(),
               'schedulers':[{'mode':s.mode,'mode_epoch':s.mode_epoch,'v47_counts':dict(s.v47_counts),
               'mixed_counts':dict(s.mixed_counts),'replan_reasons':dict(s.events.counts),'service_model':s.cost_model.describe(),
               'pure_core_transitions':getattr(s,'pure_core_transitions',0),
               'last_joint_summary':s.last_joint_summary} for s in schedulers],
               'forecast_stats':dict(native_chain.MODEL.stats),'reward_count_stats':dict(REWARD.stats),
               'pure_core_control':'frozen_v47_commit_082e98c7_pure_only',
               'pure_core_forecast_stats':dict(pure_core_v47.MODEL.stats)}
        for name in ['V47PRO_RUNTIME.json','MIXED_RUNTIME.json']:
            if output.is_dir():(output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    write();atexit.register(write,True)
    return receipt
