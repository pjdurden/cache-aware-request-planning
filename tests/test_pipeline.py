from ccrp.cost_model import CacheParams
from ccrp.pipeline import build_requests, plan

PARAMS = CacheParams(300.0, 1024, 0.1, 1.25, 3.0, 15.0)


def _raw(rid, parts, arrival):
    return {"id": rid, "prefix_parts": parts, "prefix_tokens": 2000,
            "suffix_tokens": 100, "output_tokens": 50, "arrival_s": arrival}


def test_canonicalization_groups_despite_key_order():
    # same tool schema, different dict key order -> same canonical prefix_key
    raw = [
        _raw("a", [{"name": "f", "args": {"x": 1, "y": 2}}], 0.0),
        _raw("b", [{"args": {"y": 2, "x": 1}, "name": "f"}], 1.0),
    ]
    reqs = build_requests(raw)
    assert reqs[0].prefix_key == reqs[1].prefix_key
    result = plan(raw, PARAMS, service_time_s=1.0, max_slack_s=100.0)
    assert len(result["cluster_sizes"]) == 1
    assert list(result["cluster_sizes"].values())[0] == 2


def test_full_chain_shaping_beats_naive_when_ttl_busts():
    # two distinct prefixes interleaved; service_time 200 > TTL/2 so naive
    # re-access gap (~400s) exceeds the 300s TTL. Grouping keeps them warm.
    k1 = [{"sys": "agent one"}]
    k2 = [{"sys": "agent two"}]
    raw = [_raw("a", k1, 0.0), _raw("b", k2, 0.0), _raw("c", k1, 0.0), _raw("d", k2, 0.0)]
    result = plan(raw, PARAMS, service_time_s=200.0, max_slack_s=10000.0)
    assert result["shaped_cost"] < result["naive_cost"]
    assert len(result["cluster_sizes"]) == 2
