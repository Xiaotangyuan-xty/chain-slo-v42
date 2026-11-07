"""Bounded counts of currently known text, never future test outputs."""
from collections import Counter, OrderedDict
import hashlib


class TokenReward:
    def __init__(self, capacity=2048):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.tokenizer = None
        self.stats = Counter()

    def bind(self, tokenizer):
        if tokenizer is not self.tokenizer:
            self.cache.clear()
            self.tokenizer = tokenizer

    def count(self, text, fallback):
        if self.tokenizer is None:
            self.stats['engine_token_fallbacks'] += 1
            return int(fallback)
        key = hashlib.sha256(text.encode()).digest()
        if key in self.cache:
            self.stats['token_cache_hits'] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.stats['token_cache_misses'] += 1
        count = len(self.tokenizer.encode(text, add_special_tokens=False))
        self.cache[key] = count
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return count


REWARD = TokenReward()
