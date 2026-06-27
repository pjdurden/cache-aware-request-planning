import math
from ccrp.cost_model import CacheParams, Request, billed_cost
from ccrp.baselines import compression_cost, semantic_cache_cost

PARAMS = CacheParams(300.0, 1024, 0.1, 1.25, 3.0, 15.0)


def _r(rid, arrival):
    return Request(rid, "k", 2000, 200, 100, arrival)


def test_compression_shrinks_input_and_adds_local_cost():
    reqs = [_r("a", 0.0)]
    server, local = compression_cost(reqs, PARAMS, ratio=0.5, local_cost_per_request=0.01)
    # half the input tokens billed, no caching, output unchanged
    expected_server = (1000 + 100) * 3.0 / 1000 + 100 * 15.0 / 1000
    assert math.isclose(server, expected_server)
    assert math.isclose(local, 0.01)


def test_semantic_cache_skips_hits():
    reqs = [_r("a", 0.0), _r("b", 1.0)]
    server, local = semantic_cache_cost(reqs, PARAMS, hit_rate=0.5, local_cost_per_request=0.002)
    # 1 hit (skipped), 1 miss (full, no provider cache)
    miss = billed_cost(reqs[0], 0, False, PARAMS)
    assert math.isclose(server, miss)
    assert math.isclose(local, 2 * 0.002)
