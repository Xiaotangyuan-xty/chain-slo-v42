"""Explicit server entry point for the revised, v45-based v46 candidate."""
import atexit
import json
from pathlib import Path


def install(server, output=None, minimal_observation=True):
    stage = Path(__file__).resolve().parents[1]
    observation = None
    if minimal_observation:
        from low_observation import install as install_low_observation
        observation = install_low_observation(stage)
    import native_chain
    native_chain.install(server)
    import vllm.engine.llm_engine as llm_engine
    from mixed_joint import MixedScheduler, VERSION, VARIANT
    llm_engine.SLOScheduler = MixedScheduler
    receipt = {'version': VERSION, 'variant': VARIANT,
               'v45_scheduler_and_chain_admission_source_unchanged': True,
               'minimal_observation': observation,
               'scope': ['mixed coexistence', 'remaining-work boundary fixes'],
               'stage_budget_adapter': False, 'new_execution_cadence': False,
               'wall_clock_lease': False, 'event_driven_replanning': False}
    if output is not None:
        output = Path(output)
        (output / 'V46_MINIMAL_RUNTIME.json').write_text(json.dumps(receipt, indent=2) + '\n')

        def finalize():
            engine = getattr(server, 'engine', None)
            scheduler = engine.engine.scheduler[0] if engine is not None else None
            data = {**receipt, 'counts': dict(scheduler.mixed_counts) if scheduler else {},
                    'last_joint_summary': scheduler.last_joint_summary if scheduler else None}
            (output / 'MIXED_RUNTIME.json').write_text(json.dumps(data, indent=2) + '\n')
        atexit.register(finalize)
    return receipt
