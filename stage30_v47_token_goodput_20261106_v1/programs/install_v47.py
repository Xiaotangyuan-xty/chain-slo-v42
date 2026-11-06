"""Fresh-process entry point for v47. Does not launch an engine or GPU job."""
import atexit
import json
from pathlib import Path


def install(server, output=None):
    import native_chain
    from mixed_joint import MixedScheduler, VERSION, VARIANT
    from token_reward import REWARD
    import vllm.engine.llm_engine as llm_engine
    native_chain.install(server)
    llm_engine.SLOScheduler = MixedScheduler
    receipt = {'version': VERSION, 'variant': VARIANT,
               'parent': 'v46_v45_mixed_and_work_bounds_only',
               'score': 'conditional expected full-chain prompt+8*output credit / remaining work',
               'stage_budget_adapter': False, 'detailed_decision_log': False,
               'expired_drain': 'reversible; known unfinished root deadlines, including stage barriers',
               'drain_seq_limit': 'capacity-aware, at most max(initial_limit,min(128,batch_token_budget))',
               'installer_allocates_GPU': False, 'installer_runs_model': False,
               'future_truth_access': False, 'request_drops': False}
    if output is not None:
        output = Path(output)
        def write():
            if not output.exists():
                return
            engine = getattr(server, 'engine', None)
            scheduler = engine.engine.scheduler[0] if engine is not None else None
            data = {**receipt, 'counts': dict(scheduler.v47_counts) if scheduler else {},
                    'mixed_counts': dict(scheduler.mixed_counts) if scheduler else {},
                    'last_joint_summary': scheduler.last_joint_summary if scheduler else None,
                    'forecast_stats': dict(getattr(native_chain.MODEL, 'stats', {})),
                    'reward_count_stats': dict(REWARD.stats)}
            (output / 'V47_RUNTIME.json').write_text(json.dumps(data, indent=2) + '\n')
            mixed = {**data, 'v47_counts': data['counts'], 'counts': data['mixed_counts']}
            (output / 'MIXED_RUNTIME.json').write_text(json.dumps(mixed, indent=2) + '\n')
        write()
        atexit.register(write)
    return receipt
