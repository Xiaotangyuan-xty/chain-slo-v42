from dataclasses import dataclass

from v2_contracts import View


@dataclass(frozen=True)
class ExecutionView(View):
    chain_virtual: tuple = ()
    declared_inputs: tuple = ()
    compound_members: tuple = ()
    resource_units: tuple = ()
    free_kv_blocks: int = 0
    watermark_blocks: int = 0
    block_size: int = 16
    max_swaps_per_iter: int = 3
    token_budget: int = 2048
    native_priorities: tuple = ()
    compound_progress: tuple = ()
    schedule_count: int | None = None
    schedule_interval: int = 50
    official_frame: bool | None = None
    incumbent_action: tuple | None = None
    host_free_kv_blocks: int | None = None
    host_kv_capacity_blocks: int | None = None
    decision_overhead_seconds: float = 0.0
    custom_forced_enabled: bool = True
    defer_hopeless: bool = False
    defer_safe_later: bool = False
    admission_frame_reserve: float = 0.0

    def virtual_for(self, program):
        return dict(self.chain_virtual).get(program, 0.0)

    def declared_for(self, key):
        return dict(self.declared_inputs).get(key)

    def resource_for(self, key):
        return dict(self.resource_units).get(key)

    def member_progress_for(self, key):
        value = dict(self.compound_progress).get(key)
        if value is None:
            return None
        return dict(value) if isinstance(value, dict) else tuple(value)

    def resource_contract(self):
        if not self.resource_units:
            return None
        contract = {
            "free_blocks": self.free_kv_blocks,
            "watermark_blocks": self.watermark_blocks,
            "block_size": self.block_size,
            "max_swaps": self.max_swaps_per_iter,
            "token_budget": self.token_budget,
        }
        if self.host_free_kv_blocks is not None:
            contract["host_free_blocks"] = self.host_free_kv_blocks
        if self.host_kv_capacity_blocks is not None:
            contract["host_capacity_blocks"] = self.host_kv_capacity_blocks
        return contract

    def native_priority_for(self, key):
        return dict(self.native_priorities).get(key)

    @property
    def is_official_frame(self):
        if self.official_frame is not None:
            return self.official_frame
        if self.schedule_count is None:
            return True
        if self.schedule_interval <= 0:
            raise ValueError("invalid official frame interval")
        return self.schedule_count % self.schedule_interval == 0

    @property
    def frame_index(self):
        if self.schedule_count is None:
            return None
        return self.schedule_count // self.schedule_interval
