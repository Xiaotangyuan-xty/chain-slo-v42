from pool_backend import BatchedBackend


class SchedulerManagedBackend(BatchedBackend):
    def _reserve(self, key, n):
        self.resolve(key)
        old = self.ledger.view().get(key)
        required = max(0, n - old.gpu)
        if self.ledger.view().free < required:
            raise RuntimeError(
                f"Official scheduler and backend KV ledger diverged for {key}: "
                f"required={required}, free={self.ledger.view().free}")
