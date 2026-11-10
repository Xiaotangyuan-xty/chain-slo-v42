"""Shared runtime concurrency ceiling; preserve native scoring and KV allocation."""
import hashlib,inspect,json
from pathlib import Path

VERSION='shared_initial32_dynamic128_v1'


def install(output, initial=32, ceiling=128):
    from jitserve.scheduler.slo_scheduler import SLOScheduler
    if getattr(SLOScheduler,'_shared_budget_contract',False):
        raise RuntimeError('Fresh scheduler process required')
    previous=SLOScheduler._adjust_batch_limits
    def adjust(self,process_list,min_tbt_required,budget):
        previous(self,process_list,min_tbt_required,budget)
        budget.max_num_seqs=min(ceiling,max(1,budget.max_num_seqs))
        self.scheduler_config.max_num_seqs=min(ceiling,max(1,self.scheduler_config.max_num_seqs))
    SLOScheduler._adjust_batch_limits=adjust
    SLOScheduler._shared_budget_contract=True
    receipt={'version':VERSION,'initial_max_num_seqs':initial,'runtime_sequence_ceiling':ceiling,
        'scope':'Same ceiling for all methods; native A/Astar scoring and budget choice below ceiling unchanged; candidate pure code unchanged.',
        'physical_KV_modified':False,'scores_modified':False,'CUDA_synchronization_added':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (Path(output)/'RUNTIME_BUDGET_CONTRACT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
