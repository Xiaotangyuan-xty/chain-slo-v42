from vllm.core.scheduler import PreemptionMode


class RecomputeInvalidatingScheduler:
    recompute_invalidate = None

    def _preempt(self, seq_group, blocks_to_swap_out, preemption_mode=None):
        mode = super()._preempt(seq_group, blocks_to_swap_out, preemption_mode)
        if mode == PreemptionMode.RECOMPUTE and self.recompute_invalidate is not None:
            self.recompute_invalidate(seq_group.request_id)
        return mode
