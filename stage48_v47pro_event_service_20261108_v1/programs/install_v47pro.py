"""Fresh v47-derived process only. Installing never starts a model or GPU job."""
import atexit
import functools
import json
from pathlib import Path


def install(server,output,*,model_key='llama8',shared_prediction_config=None):
    from prediction_transport import PROTOCOL
    output=Path(output)
    if not output.is_dir() or (output/'V47PRO_RUNTIME.json').exists():raise ValueError('Fresh existing output directory required')
    if getattr(server,'_v47pro_installed',False):raise RuntimeError('Fresh server process required')
    if shared_prediction_config is not None and shared_prediction_config.get('predictor_protocol')!=PROTOCOL:raise ValueError('Matching framed prediction required')
    import native_chain
    from mixed_joint import MixedScheduler,VERSION,VARIANT
    from native_tot_adapter import install_server_annotations
    from token_reward import REWARD
    import vllm.engine.llm_engine as engine
    if getattr(MixedScheduler,'_v47pro_installed',False):raise RuntimeError('Fresh scheduler process required')
    server._v47pro_installed=True;MixedScheduler._v47pro_installed=True
    native_chain.install(server);annotations=install_server_annotations(server)
    engine.SLOScheduler=MixedScheduler
    stream=(output/'V47PRO_DECISIONS.jsonl').open('x',buffering=1)
    schedulers=[];closed=False
    previous_init=MixedScheduler.__init__
    @functools.wraps(previous_init)
    def init(self,*args,**kwargs):
        previous_init(self,*args,**kwargs);index=len(schedulers);schedulers.append(self)
        self.decision_sink=lambda event:stream.write(json.dumps({'scheduler':index,**event},allow_nan=False)+'\n')
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
        stream.flush()
        data={**receipt,'schedulers':[{'mode':s.mode,'mode_epoch':s.mode_epoch,'v47_counts':dict(s.v47_counts),
               'mixed_counts':dict(s.mixed_counts),'replan_reasons':dict(s.events.counts),'service_model':s.cost_model.describe(),
               'last_joint_summary':s.last_joint_summary} for s in schedulers],
               'forecast_stats':dict(native_chain.MODEL.stats),'reward_count_stats':dict(REWARD.stats)}
        for name in ['V47PRO_RUNTIME.json','MIXED_RUNTIME.json']:
            if output.is_dir():(output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        if final:closed=True;stream.close()
    write();atexit.register(write,True)
    return receipt
