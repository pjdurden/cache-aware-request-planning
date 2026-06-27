import math
from ccrp.cost_model import (
    CacheParams,
    Request,
    billed_cost,
    intervention_net_savings,
)

PARAMS = CacheParams(
    ttl_s=300.0,
    min_prefix_tokens=1024,
    read_discount=0.1,
    write_multiplier=1.25,
    input_price_per_1k=3.0,
    output_price_per_1k=15.0,
)


def _req(prefix=2000, suffix=200, output=100):
    return Request("r", "k", prefix, suffix, output, 0.0)


def test_full_miss_no_cache_write():
    # nothing cached, nothing written: pay full input + output
    r = _req()
    cost = billed_cost(r, cached_prefix_tokens=0, wrote_prefix=False, params=PARAMS)
    expected = (2000 + 200) * 3.0 / 1000 + 100 * 15.0 / 1000
    assert math.isclose(cost, expected)


def test_full_write_applies_multiplier_to_prefix():
    r = _req()
    cost = billed_cost(r, cached_prefix_tokens=0, wrote_prefix=True, params=PARAMS)
    expected = (2000 * 1.25 + 200) * 3.0 / 1000 + 100 * 15.0 / 1000
    assert math.isclose(cost, expected)


def test_full_hit_applies_read_discount_to_prefix():
    r = _req()
    cost = billed_cost(r, cached_prefix_tokens=2000, wrote_prefix=False, params=PARAMS)
    expected = (2000 * 0.1 + 200) * 3.0 / 1000 + 100 * 15.0 / 1000
    assert math.isclose(cost, expected)


def test_net_savings_subtracts_local_cost():
    assert math.isclose(
        intervention_net_savings(baseline_cost=1.0, intervention_cost=0.6, local_cost=0.1),
        0.3,
    )
