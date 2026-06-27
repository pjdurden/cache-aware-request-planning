from collections import defaultdict


def cluster_by_prefix(requests):
    groups = defaultdict(list)
    for r in requests:
        groups[r.prefix_key].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r.arrival_s)
    return dict(groups)
