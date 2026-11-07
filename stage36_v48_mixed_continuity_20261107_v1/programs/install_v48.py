"""Fresh-process candidate installer; never constructs an engine or starts a job."""
import atexit
import functools
import hashlib
import json
from pathlib import Path


def install(server, output=None, *, model_key=None, shared_prediction_config=None):
    if model_key not in [None,'llama8','qwen14']:
        raise ValueError('Unknown calibration domain; use None for explicit transfer mode')
    if getattr(server,'_v48_candidate_installed',False):
        raise RuntimeError('v48 requires one installer in a fresh serving process')
    import native_chain
    from mixed_joint import MixedScheduler, VERSION, VARIANT
    from token_reward import REWARD
    import vllm.engine.llm_engine as engine
    server._v48_candidate_installed=True
    native_chain.install(server)
    previous=server._handle_collective_request
    def collective(info,data,prompt):
        implicit='requests_per_stage' not in data
        depth=data.get('num_stages',6)
        if implicit and depth!=4:
            raise ValueError('Only native3-thought/2-round ToT is qualified')
        result=previous(info,data,prompt)
        if implicit:
            info.application='tot_native';info.chain_stage_count=4
            info.chain_stage_widths=[3,1,3,1];info.chain_future_max_tokens=256
            info.v48_model_key=model_key
        return result
    server._handle_collective_request=collective
    engine.SLOScheduler=MixedScheduler
    instances=[];previous_init=MixedScheduler.__init__
    @functools.wraps(previous_init)
    def init(self,*args,**kwargs):
        previous_init(self,*args,**kwargs);instances.append(self)
    MixedScheduler.__init__=init
    shared=None
    if shared_prediction_config is not None:
        from shared_prediction import install as shared_install
        from jitserve.scheduler.policy import SLOPolicy
        shared=shared_install(server,SLOPolicy,output,method='v48',**shared_prediction_config)
    root=Path(__file__).resolve().parents[1]
    receipt={'version':VERSION,'variant':VARIANT,'parent_commit':'082e98c7c05431110bd6cd74234f909487b963ee',
             'installer_runs_model':False,'installer_allocates_GPU':False,'stage_budget_adapter':False,
             'model_key':model_key,'ToT_path':'native implicit [3,1,3,1], cap256',
             'shared_prediction':shared,'prediction_comparison_ready':shared is not None,
             'prediction_comparison_requirement':'Apply the identical shared protocol to A and Astar too, or label all arms legacy transport. Never compare repaired-v48 with unrepaired historical baselines.',
             'request_drops':False,'future_truth_access':False,'benchmark_end_time_used':False,
             'pure_files':{n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in
                           ['programs/native_chain.py','programs/chain_forecast.py','programs/chain_admission.py','programs/token_reward.py','assets/CHAIN_MODEL.json']}}
    if output is not None:
        output=Path(output)
        def write():
            from mixed_forecast import BANK
            runtime={**receipt,'schedulers':[{'mixed_counts':dict(s.mixed_counts),
                       'v47_pure_drain_counts':dict(s.v47_counts),
                       'plan_events_dropped':s.plan_events_dropped,
                       'last_joint_summary':s.last_joint_summary}for s in instances],
                     'DR_forecast_stats':dict(native_chain.MODEL.stats),
                     'ToT_forecast_stats':{k:dict(v.stats)for k,v in (BANK or {}).items()},
                     'reward_count_stats':dict(REWARD.stats)}
            if not output.exists():return
            for name in ['V48_RUNTIME.json','MIXED_RUNTIME.json']:
                (output/name).write_text(json.dumps(runtime,indent=2,allow_nan=False)+'\n')
            with(output/'V48_DECISIONS.jsonl').open('w')as stream:
                for s in instances:
                    for event in s.plan_events:stream.write(json.dumps(event,allow_nan=False)+'\n')
        write();atexit.register(write)
    return receipt
