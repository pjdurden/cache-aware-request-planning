from ccrp.cost_model import CacheParams, Request
from ccrp.scheduler import greedy_schedule
from ccrp.simulate import simulate_order

PARAMS = CacheParams(300.0, 1024, 0.1, 1.25, 3.0, 15.0)


def _r(rid, key, arrival):
    return Request(rid, key, 2000, 100, 50, arrival)


def test_interleaved_prefixes_get_grouped_when_slack_allows():
    # arrival order interleaves k1/k2; grouping yields more hits
    reqs = [_r("a", "k1", 0.0), _r("b", "k2", 0.0), _r("c", "k1", 0.0), _r("d", "k2", 0.0)]
    order = greedy_schedule(reqs, PARAMS, service_time_s=200.0, max_slack_s=10000.0)
    keys = [r.prefix_key for r in order]
    # same-key requests end up adjacent
    assert keys in (["k1", "k1", "k2", "k2"], ["k2", "k2", "k1", "k1"])
    assert simulate_order(order, PARAMS, 200.0) < simulate_order(reqs, PARAMS, 200.0)


def test_deadline_guard_prevents_starvation():
    # b has zero slack and must be served before grouping a's together
    reqs = [_r("a1", "k1", 0.0), _r("b", "k2", 0.0), _r("a2", "k1", 0.0)]
    order = greedy_schedule(reqs, PARAMS, service_time_s=1.0, max_slack_s=0.0)
    # with zero slack nothing can be deferred, so order respects arrival/no reorder past deadline
    served_at = {r.id: i for i, r in enumerate(order)}
    assert served_at["b"] <= 1


def test_all_requests_scheduled_once():
    reqs = [_r("a", "k1", 0.0), _r("b", "k2", 1.0), _r("c", "k1", 2.0)]
    order = greedy_schedule(reqs, PARAMS, service_time_s=1.0, max_slack_s=10.0)
    assert sorted(r.id for r in order) == ["a", "b", "c"]
