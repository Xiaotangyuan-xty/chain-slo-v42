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

    def virtual_for(self, program):
        return dict(self.chain_virtual).get(program, 0.0)

    def declared_for(self, key):
        return dict(self.declared_inputs).get(key)

    def resource_for(self, key):
        return dict(self.resource_units).get(key)

    def resource_contract(self):
        if not self.resource_units:
            return None
        return {
            "free_blocks": self.free_kv_blocks,
            "watermark_blocks": self.watermark_blocks,
            "block_size": self.block_size,
            "max_swaps": self.max_swaps_per_iter,
            "token_budget": self.token_budget,
        }

    def native_priority_for(self, key):
        return dict(self.native_priorities).get(key)
