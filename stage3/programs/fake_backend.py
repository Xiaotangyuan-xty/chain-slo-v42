from dataclasses import asdict

from contracts import CacheLedger


class FakeEvent:
    def synchronize(self):
        return None

    def elapsed_time(self, other):
        return 0.0


class FakeLogits:
    def __init__(self, token):
        self.token = token

    def argmax(self, dim=-1):
        return self

    def item(self):
        return self.token


class FakeBackend:
    def __init__(self):
        self.gpu = {}
        self.host = {}
        self.gpu_tokens = {}
        self.host_tokens = {}
        self.tokens = {}
        self.pending = {}
        self.events = []
        self.stats = {
            "forward_wall_seconds": 0.0,
            "actual_forward_query_tokens": 0,
            "logical_outputs": 0,
            "copy_wait_blocked_seconds": 0.0,
            "copy_cuda_span_seconds": 0.0,
            "swap_bytes_out": 0,
            "recompute_tokens": 0,
        }
        self.ledger = CacheLedger(8192, 8192)

    def reset(self, gpu_capacity, host_capacity):
        self.__init__()
        self.ledger = CacheLedger(gpu_capacity, host_capacity)
        self.gpu_capacity = gpu_capacity
        self.host_capacity = host_capacity

    def resolve(self, key):
        return None

    def _record_copy_wait(self, seconds):
        self.stats["copy_wait_blocked_seconds"] += seconds

    def _copy(self, cache, to_host):
        return cache, FakeEvent(), FakeEvent()

    def nbytes(self, cache):
        return 0

    def _restore(self, key):
        if key not in self.host or key in self.gpu:
            return
        self.gpu[key] = self.host[key]
        self.gpu_tokens[key] = self.host_tokens[key]
        self.ledger.commit("gpu_produced", key, len(self.gpu_tokens[key]))
        self.events.append({"event": "fake_swap_in", "program": key, "tokens": len(self.gpu_tokens[key])})

    def step_batch(self, items):
        if len(items) > 4:
            raise RuntimeError("fake backend batch limit")
        logits = {}
        for key, prompt, emitted in items:
            self._restore(key)
            if emitted:
                computed = tuple(prompt) + tuple(emitted)
                queries = 1
            else:
                computed = tuple(prompt)
                previous = self.gpu_tokens.get(key, ())
                if computed[:len(previous)] != previous:
                    raise RuntimeError("partial-prefill prefix mismatch")
                queries = len(computed) - len(previous)
            if queries < 0:
                raise RuntimeError("negative fake forward query count")
            self.gpu[key] = object()
            self.gpu_tokens[key] = computed
            self.tokens[key] = computed
            self.ledger.commit("gpu_produced", key, len(computed))
            self.ledger.commit("touch", key)
            self.stats["actual_forward_query_tokens"] += queries
            self.stats["logical_outputs"] += 1
            token = 1000 + ((len(computed) * 31 + len(items)) % 1000)
            logits[key] = FakeLogits(token)
            self.events.append({"event": "fake_forward", "program": key, "queries": queries, "computed": len(computed)})
        self.stats["forward_wall_seconds"] += 0.001
        return logits

    def release(self, key):
        for mapping in [self.gpu, self.host, self.gpu_tokens, self.host_tokens, self.tokens]:
            mapping.pop(key, None)
        self.ledger.commit("release", key)
        self.events.append({"event": "fake_release", "program": key})

    def summary(self):
        return {
            **self.stats,
            "persistent_quota_tokens": self.gpu_capacity,
            "host_quota_tokens": self.host_capacity,
            "ledger": [asdict(entry) for entry in self.ledger.view().entries],
            "events": list(self.events),
        }
