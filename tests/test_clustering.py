from ccrp.cost_model import Request
from ccrp.clustering import cluster_by_prefix


def _r(rid, key, arrival):
    return Request(rid, key, 2000, 100, 50, arrival)


def test_groups_by_prefix_key():
    reqs = [_r("a", "k1", 0.0), _r("b", "k2", 1.0), _r("c", "k1", 2.0)]
    groups = cluster_by_prefix(reqs)
    assert set(groups.keys()) == {"k1", "k2"}
    assert [r.id for r in groups["k1"]] == ["a", "c"]


def test_within_group_sorted_by_arrival():
    reqs = [_r("late", "k", 9.0), _r("early", "k", 1.0)]
    groups = cluster_by_prefix(reqs)
    assert [r.id for r in groups["k"]] == ["early", "late"]
