from ccrp.cost_model import CacheParams
from ccrp.cache_sim import CacheSim

PARAMS = CacheParams(300.0, 1024, 0.1, 1.25, 3.0, 15.0)


def test_short_prefix_never_cacheable():
    sim = CacheSim(PARAMS)
    r = sim.access("k", prefix_tokens=500, now_s=0.0)
    assert (r.hit, r.cached_prefix_tokens, r.wrote_prefix) == (False, 0, False)
    assert sim.is_warm("k", 0.0) is False


def test_cold_miss_writes_then_warm_hit():
    sim = CacheSim(PARAMS)
    first = sim.access("k", 2000, now_s=0.0)
    assert (first.hit, first.wrote_prefix) == (False, True)
    second = sim.access("k", 2000, now_s=10.0)
    assert (second.hit, second.cached_prefix_tokens, second.wrote_prefix) == (True, 2000, False)


def test_expiry_after_ttl_is_a_miss_again():
    sim = CacheSim(PARAMS)
    sim.access("k", 2000, now_s=0.0)
    assert sim.is_warm("k", 299.0) is True
    assert sim.is_warm("k", 301.0) is False
    third = sim.access("k", 2000, now_s=400.0)
    assert (third.hit, third.wrote_prefix) == (False, True)


def test_access_refreshes_ttl():
    sim = CacheSim(PARAMS)
    sim.access("k", 2000, now_s=0.0)
    sim.access("k", 2000, now_s=200.0)   # refresh
    assert sim.is_warm("k", 450.0) is True  # 200 + 300 = 500 > 450
