"""Execution cadence and buffered decision events; no token filling or policy edits."""
import atexit
from collections import deque
import functools,json,time
from pathlib import Path

class ExecutionCadence:
    def __init__(self):self.decode=deque(maxlen=64);self.transfer=deque(maxlen=64);self.decode_batches=0;self.transition_batches=0
    @property
    def token_seconds(self):return min(1.,max(.001,sum(self.decode)/len(self.decode)))if self.decode else .02
    @property
    def transfer_per_block(self):return sum(self.transfer)/len(self.transfer)if self.transfer else .0001
    def observe(self,seconds,groups,transfer_blocks):
        if any(not g.is_prompt for g in groups):self.decode.append(seconds);self.decode_batches+=1
        if not groups and transfer_blocks:self.transfer.append(seconds/transfer_blocks);self.transition_batches+=1

class DecisionLog:
    def __init__(self,output,limit=100000):
        self.path=Path(output);self.file=(self.path/'V46_DECISIONS.jsonl').open('x',buffering=1024*1024);self.limit=limit;self.count=0;self.closed=False
    def write(self,kind,**fields):
        if self.count>=self.limit:raise RuntimeError('Decision evidence bound exceeded; preserve run')
        self.file.write(json.dumps({'kind':kind,'monotonic':time.perf_counter(),**fields},allow_nan=False)+'\n');self.count+=1
    def close(self):
        if not self.closed:self.file.close();self.closed=True

def install_execution_observation(server,output,version,baseline_variant):
    original=server.init_app;created=[]
    async def init_app(*args,**kwargs):
        app=await original(*args,**kwargs);engine=server.engine.engine;executor=engine.model_executor;scheduler=engine.scheduler[0]
        cadence=getattr(scheduler,'cadence',ExecutionCadence());scheduler.cadence=cadence
        execute=executor.execute_model_async
        @functools.wraps(execute)
        async def measured(request,*a,**kw):
            groups=request.seq_group_metadata_list if request is not None else[]
            transfers=sum(len(getattr(request,key,())or())for key in ['blocks_to_swap_in','blocks_to_swap_out'])if request is not None else 0
            begin=time.perf_counter();result=await execute(request,*a,**kw);cadence.observe(time.perf_counter()-begin,groups,transfers);return result
        executor.execute_model_async=measured;created.append(scheduler)
        if hasattr(scheduler,'decision_log'):scheduler.decision_log=DecisionLog(output)
        return app
    server.init_app=init_app
    def close():
        for scheduler in created:
            if getattr(scheduler,'decision_log',None):
                scheduler.decision_log.close()
                (Path(output)/'MIXED_RUNTIME.json').write_text(json.dumps({'variant':'v46_joint_foreground_budget_v1','rule':'Joint charged foreground/chain admission; foreground half-pool target with work-conserving borrowing; no global native bypass.','counts':dict(scheduler.mixed_counts),'frozen_v45_core_bytes_modified':False},indent=2)+'\n')
            c=scheduler.cadence
            (Path(output)/'V46_EXECUTION_OBSERVATION.json').write_text(json.dumps({'version':version,'baseline_variant':baseline_variant,
             'decode_batches':c.decode_batches,'transition_batches':c.transition_batches,'final_decode_step_seconds':c.token_seconds,
             'transfer_seconds_per_block':c.transfer_per_block,'decision_events':getattr(getattr(scheduler,'decision_log',None),'count',0)},indent=2)+'\n')
    atexit.register(close)
