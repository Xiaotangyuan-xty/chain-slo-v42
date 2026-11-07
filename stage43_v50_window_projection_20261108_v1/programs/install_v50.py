"""Fresh-process installer; constructs no engine, device context or replay."""
import atexit
import functools
import hashlib
import json
import math
from pathlib import Path
from planner_config import CONFIG,contract


def install(server, output=None, *, model_key=None, shared_prediction_config=None):
    if output is None:raise ValueError('v50 requires a fresh output directory for budget and observation receipts')
    if model_key not in [None, 'llama8', 'qwen14']:
        raise ValueError('Unknown calibration domain; None is explicit transfer mode')
    if getattr(server, '_v50_candidate_installed', False):
        raise RuntimeError('v50 requires one installer in a fresh serving process')
    if shared_prediction_config is not None:
        from prediction_transport import PROTOCOL
        config=shared_prediction_config
        if (not {'port','predictor_protocol'}<=set(config)
                or set(config)-{'port','predictor_protocol','timeout_seconds'}
                or config['predictor_protocol']!=PROTOCOL
                or not isinstance(config['port'],int)or not 1024<=config['port']<=65535
                or not math.isfinite(config.get('timeout_seconds',10.))
                or config.get('timeout_seconds',10.)<=0):
            raise ValueError('Explicit matching shared prediction configuration required')
    if output is not None:
        path=Path(output)
        if not path.is_dir()or any((path/n).exists()for n in ['V50_RUNTIME.json','V50_DECISIONS.jsonl']):
            raise ValueError('Fresh existing output directory required')
    import native_chain
    from mixed_joint import MixedScheduler, VERSION, VARIANT
    from token_reward import REWARD
    import vllm.engine.llm_engine as engine
    if getattr(MixedScheduler,'_v50_runtime_installed',False):
        raise RuntimeError('v50 comparison requires a fresh serving process')
    MixedScheduler._v50_runtime_installed=True
    server._v50_candidate_installed = True
    from candidate_budget import install as install_budget
    budget=install_budget(output)
    native_chain.install(server)
    previous = server._handle_collective_request
    def collective(info, data, prompt):
        implicit = 'requests_per_stage' not in data
        depth = data.get('num_stages', 6)
        if implicit and depth != 4:
            raise ValueError('Only native3-thought/2-round ToT is qualified')
        result = previous(info, data, prompt)
        if implicit:
            info.application='tot_native';info.chain_stage_count=4
            info.chain_stage_widths=[3,1,3,1];info.chain_future_max_tokens=256
            info.mixed_model_key=model_key
        return result
    server._handle_collective_request = collective
    engine.SLOScheduler = MixedScheduler
    output = Path(output) if output is not None else None
    stream = (output/'V50_DECISIONS.jsonl').open('x',buffering=1024*1024) if output else None
    instances=[];logged=0;closed=False
    previous_init=MixedScheduler.__init__
    @functools.wraps(previous_init)
    def init(self,*args,**kwargs):
        previous_init(self,*args,**kwargs);index=len(instances);instances.append(self)
        if stream:
            def sink(event):
                nonlocal logged
                stream.write(json.dumps({'scheduler':index,**event},allow_nan=False)+'\n')
                logged+=1
            self.decision_sink=sink
    MixedScheduler.__init__=init
    shared=None
    if shared_prediction_config is not None:
        from shared_prediction import install as shared_install
        from jitserve.scheduler.policy import SLOPolicy
        shared=shared_install(server,SLOPolicy,output,method='v50',**shared_prediction_config)
    root=Path(__file__).resolve().parents[1]
    receipt={'version':VERSION,'variant':VARIANT,'parent_commit':'077436cfa01cd3774d324e20891b6af01877f846',
             'installer_runs_model':False,'installer_allocates_GPU':False,'stage_budget_adapter':True,'candidate_budget':budget,
             'model_key':model_key,'ToT_path':'native implicit [3,1,3,1], cap256',
             'shared_prediction':shared,'prediction_comparison_ready':shared is not None,
             'prediction_comparison_requirement':'Identical shared protocol/weights/snapshot semantics for A/Astar/v50; do not reuse unrepaired baselines.',
             'request_drops':False,'future_truth_access':False,'benchmark_end_time_used':False,
             'step_cost_model':'indexed executor host cost strata; not CUDA kernel latency',
             'planning_model':'cached same-window counterfactual stage envelope; approximate common continuation',
             'planning_budget_seconds':CONFIG.max_seconds,'deadline_planning_reserve_seconds':CONFIG.max_seconds+CONFIG.final_margin_seconds,
             'planner_contract':contract(),
             'anytime_keeps_completed_positive_gain':True,
             'decision_logging':'all plans, bounded candidate summaries; executor measurements in separate buffered stream',
             'pure_files':{n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in
                           ['programs/native_chain.py','programs/chain_forecast.py','programs/chain_admission.py','programs/token_reward.py','assets/CHAIN_MODEL.json']}}
    def write(final=False):
        nonlocal closed
        if closed:return
        if stream:stream.flush()
        if output and output.exists():
            from mixed_forecast import BANK
            runtime={**receipt,'logged_plan_events':logged,'schedulers':[
                {'mixed_counts':dict(s.mixed_counts),'v47_pure_drain_counts':dict(s.v47_counts),
                 'unstreamed_plan_events_lost':s.plan_events_dropped,
                 'step_cost_stats':dict(s.cost_model.stats),'step_cost_bins':s.cost_model.snapshot(.02,1).describe(),
                 'last_joint_summary':s.last_joint_summary}for s in instances],
                 'DR_forecast_stats':dict(native_chain.MODEL.stats),
                 'ToT_forecast_stats':{k:dict(v.stats)for k,v in(BANK or{}).items()},
                 'reward_count_stats':dict(REWARD.stats)}
            for name in ['V50_RUNTIME.json','MIXED_RUNTIME.json']:
                (output/name).write_text(json.dumps(runtime,indent=2,allow_nan=False)+'\n')
        if final:
            closed=True
            if stream:stream.close()
    from execution_observation import install as observe_execution
    receipt['execution_observer']=observe_execution(output) if output else None
    write();atexit.register(write,True)
    return receipt
