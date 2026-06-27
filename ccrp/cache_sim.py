from dataclasses import dataclass


@dataclass
class AccessResult:
    hit: bool
    cached_prefix_tokens: int
    wrote_prefix: bool


class CacheSim:
    def __init__(self, params):
        self.params = params
        self._warm_until = {}   # prefix_key -> timestamp

    def is_warm(self, prefix_key, now_s):
        until = self._warm_until.get(prefix_key)
        return until is not None and until > now_s

    def access(self, prefix_key, prefix_tokens, now_s):
        if prefix_tokens < self.params.min_prefix_tokens:
            return AccessResult(False, 0, False)
        if self.is_warm(prefix_key, now_s):
            self._warm_until[prefix_key] = now_s + self.params.ttl_s
            return AccessResult(True, prefix_tokens, False)
        self._warm_until[prefix_key] = now_s + self.params.ttl_s
        return AccessResult(False, 0, True)
