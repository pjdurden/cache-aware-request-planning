import math
from ccrp.cost_model import CacheParams, Request, billed_cost
from ccrp.simulate import simulate_order

PARAMS = CacheParams(300.0, 1024, 0.1, 1.25, 3.0, 15.0)


def _r(rid, key, arrival):
    return Request(rid, key, 2000, 100, 50, arrival)


def test_repeated_prefix_back_to_back_gets_a_hit_on_second():
    order = [_r("a", "k", 0.0), _r("b", "k", 1.0)]
    total = simulate_order(order, PARAMS, service_time_s=1.0)
    first = billed_cost(order[0], 0, True, PARAMS)        # write
    second = billed_cost(order[1], 2000, False, PARAMS)   # read hit
    assert math.isclose(total, first + second)


def test_distinct_prefixes_never_hit():
    order = [_r("a", "k1", 0.0), _r("b", "k2", 1.0)]
    total = simulate_order(order, PARAMS, service_time_s=1.0)
    each = billed_cost(order[0], 0, True, PARAMS)
    assert math.isclose(total, 2 * each)
