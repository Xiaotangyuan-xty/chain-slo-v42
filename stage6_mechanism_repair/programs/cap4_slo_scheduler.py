from jitserve.scheduler.slo_scheduler import SLOScheduler
from scheduler_recompute import RecomputeInvalidatingScheduler


class Cap4SLOScheduler(RecomputeInvalidatingScheduler, SLOScheduler):
    def __init__(self, scheduler_config, cache_config, lora_config,
                 pipeline_parallel_size=1, output_proc_callback=None, batch_slots=4):
        if not isinstance(batch_slots, int) or batch_slots <= 0:
            raise ValueError("invalid fixed request-slot capacity")
        super().__init__(scheduler_config, cache_config, lora_config,
                         pipeline_parallel_size, output_proc_callback)
        self.batch_slots = batch_slots
        self.scheduler_config.max_num_seqs = batch_slots

    def _adjust_batch_limits(self, process_list, min_tbt_required, budget):
        self.scheduler_config.max_num_seqs = self.batch_slots
        budget.max_num_seqs = self.batch_slots
