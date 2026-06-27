from ccrp.canonicalize import canonical_prefix
from ccrp.clustering import cluster_by_prefix
from ccrp.cost_model import Request
from ccrp.scheduler import greedy_schedule
from ccrp.simulate import simulate_order


def build_requests(raw):
    """Build Request objects, deriving each cache key from prefix content.

    Each raw item is a dict with: id, prefix_parts (list of str/dict),
    prefix_tokens, suffix_tokens, output_tokens, arrival_s. The prefix_key is
    the canonical form of prefix_parts, so identical prefix content (regardless
    of dict key order or whitespace) maps to the same key.
    """
    out = []
    for item in raw:
        out.append(Request(
            id=item["id"],
            prefix_key=canonical_prefix(item["prefix_parts"]),
            prefix_tokens=item["prefix_tokens"],
            suffix_tokens=item["suffix_tokens"],
            output_tokens=item["output_tokens"],
            arrival_s=item["arrival_s"],
        ))
    return out


def plan(raw, params, service_time_s, max_slack_s):
    """Chain canonicalize, cluster, schedule, and price into one result."""
    requests = build_requests(raw)
    clusters = cluster_by_prefix(requests)
    naive_order = sorted(requests, key=lambda r: r.arrival_s)
    shaped_order = greedy_schedule(requests, params, service_time_s, max_slack_s)
    return {
        "cluster_sizes": {k: len(v) for k, v in clusters.items()},
        "order": [r.id for r in shaped_order],
        "naive_cost": simulate_order(naive_order, params, service_time_s),
        "shaped_cost": simulate_order(shaped_order, params, service_time_s),
    }
