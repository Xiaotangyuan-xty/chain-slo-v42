from vllm.core.scheduler import PreemptionMode


class HostCapacityDefer(RuntimeError):
    def __init__(self, request_id, free_blocks, free_tokens):
        super().__init__("host capacity deferred")
        self.request_id = request_id
        self.free_blocks = free_blocks
        self.free_tokens = free_tokens


class RecomputeInvalidatingScheduler:
    recompute_invalidate = None

    def _preempt(self, seq_group, blocks_to_swap_out, preemption_mode=None):
        resident = self.block_manager.compute_running_seq_group_blocks([seq_group])
        free_blocks = self.block_manager.get_num_free_cpu_blocks()
        insufficient = resident > free_blocks
        free_tokens = None
        ledger = getattr(self, "v2_ledger", None)
        if ledger is not None:
            block_size = getattr(getattr(self, "cache_config", None), "block_size", 16)
            free_tokens = ledger.host_capacity - sum(entry.host for entry in ledger.entries.values())
            insufficient = insufficient or resident * block_size > free_tokens
        if insufficient:
            raise HostCapacityDefer(seq_group.request_id, free_blocks, free_tokens)
        mode = super()._preempt(seq_group, blocks_to_swap_out, preemption_mode)
        if mode == PreemptionMode.RECOMPUTE and self.recompute_invalidate is not None:
            self.recompute_invalidate(seq_group.request_id)
        return mode
