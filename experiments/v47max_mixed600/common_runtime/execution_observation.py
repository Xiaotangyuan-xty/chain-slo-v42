"""Observe actual sync/async executor boundaries without synchronizing CUDA."""
import functools
import json
import time


def features(request, scheduler):
    rows = request.seq_group_metadata_list or []
    context = sum(seq.get_len() for row in rows for seq in row.seq_data.values())
    prefills = sum((row.token_chunk_size or 0)*len(row.seq_data) for row in rows if row.is_prompt)
    flags = getattr(scheduler, '_execution_flags', {})
    return {'members':sum(len(row.seq_data) for row in rows),'context_tokens':context,
            'prefill_tokens':prefills,'prefill_groups':sum(row.is_prompt for row in rows),
            'transition':bool(request.blocks_to_swap_in or request.blocks_to_swap_out or request.blocks_to_copy),
            'swap_in_blocks':len(request.blocks_to_swap_in),'swap_out_blocks':len(request.blocks_to_swap_out),
            'copy_blocks':len(request.blocks_to_copy),**flags}


def wrap_executor(executor, schedulers, emit=lambda event: None):
    if getattr(executor,'_v49pro_observed',False):raise RuntimeError('Executor observer already installed')
    executor._v49pro_observed=True
    def before(request):
        scheduler=schedulers[request.virtual_engine]
        return scheduler,features(request,scheduler),time.perf_counter()
    def after(scheduler, values, start, error=None):
        end=time.perf_counter();values={**values,'start_monotonic':start,'end_monotonic':end,
                                     'seconds':end-start,'error':error}
        model=getattr(scheduler,'cost_model',None)
        if model is not None and hasattr(model,'observe_execution') and error is None:model.observe_execution(values,end-start)
        emit(values)
    if hasattr(executor,'execute_model_async'):
        prior=executor.execute_model_async
        @functools.wraps(prior)
        async def asynchronous(execute_model_req=None,**kw):
            request=execute_model_req or kw.pop('request',None)
            scheduler,values,start=before(request)
            try:result=await prior(execute_model_req=request,**kw)
            except BaseException as e:after(scheduler,values,start,type(e).__name__);raise
            after(scheduler,values,start);return result
        executor.execute_model_async=asynchronous
    if hasattr(executor,'execute_model'):
        prior_sync=executor.execute_model
        @functools.wraps(prior_sync)
        def synchronous(execute_model_req=None,**kw):
            request=execute_model_req or kw.pop('request',None)
            scheduler,values,start=before(request)
            try:result=prior_sync(execute_model_req=request,**kw)
            except BaseException as e:after(scheduler,values,start,type(e).__name__);raise
            after(scheduler,values,start);return result
        executor.execute_model=synchronous


def install(output):
    """One fresh-process installation, common to A/Astar/v49pro."""
    import atexit
    from pathlib import Path
    import vllm.engine.llm_engine as engine
    cls=engine.LLMEngine
    if getattr(cls,'_v49pro_execution_observation',False):raise RuntimeError('Fresh engine process required')
    cls._v49pro_execution_observation=True
    stream=(Path(output)/'EXECUTOR_EVENTS.jsonl').open('x',buffering=1024*1024)
    receipt={'version':'executor_host_boundary_v1','CUDA_synchronization_added':False,
             'engine_or_model_started_by_installer':False,'same_observer_required_all_methods':True,
             'events':0,'failed_calls':0}
    def emit(event):
        receipt['events']+=1;receipt['failed_calls']+=event['error']is not None
        stream.write(json.dumps(event,allow_nan=False)+'\n')
    def close():
        stream.flush()
        if Path(output).is_dir():
            (Path(output)/'EXECUTOR_OBSERVATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
        stream.close()
    atexit.register(close);prior=cls.__init__
    @functools.wraps(prior)
    def initialized(self,*args,**kwargs):
        prior(self,*args,**kwargs)
        for scheduler in self.scheduler:
            original_schedule=scheduler.schedule
            def measured_schedule(*args,_scheduler=scheduler,_prior=original_schedule,**kwargs):
                start=time.perf_counter();before=getattr(_scheduler,'last_plan',None)
                result=_prior(*args,**kwargs)
                duration=max(0.,time.perf_counter()-start);out=result[1]
                preempted=getattr(out,'preempted',0)
                preempted=len(preempted)if hasattr(preempted,'__len__')else preempted
                prior_flags=getattr(_scheduler,'_execution_flags',{})
                # All paths record scheduler host time. Pure/foreground plan
                # changes are charged just as joint changes; no path-specific
                # sample exclusion is used for executor observations.
                planned=before!=getattr(_scheduler,'last_plan',None)or getattr(_scheduler,'_planned_this_step',False)
                _scheduler._execution_flags={**prior_flags,'scheduler_seconds':duration,
                    'planning_seconds':duration if planned else 0.,
                    'preempted':int(preempted or 0),'ignored':len(out.ignored_seq_groups)}
                return result
            scheduler.schedule=measured_schedule
        wrap_executor(self.model_executor,self.scheduler,emit)
    cls.__init__=initialized
    return receipt
