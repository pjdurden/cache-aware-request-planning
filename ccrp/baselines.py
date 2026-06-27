from ccrp.cost_model import Request, billed_cost


def compression_cost(requests, params, ratio, local_cost_per_request):
    """Calculate cost of compression baseline.

    Shrinks prefix and suffix input tokens by ratio (e.g. 0.4 means 40% removed).
    No provider caching. Output tokens unchanged. Adds flat local cost per request.

    Args:
        requests: List of Request objects
        params: CacheParams
        ratio: Compression ratio (0.0 to 1.0)
        local_cost_per_request: Fixed cost per request

    Returns:
        (server_cost, local_cost) tuple
    """
    server = 0.0
    for r in requests:
        shrunk = Request(
            r.id, r.prefix_key,
            prefix_tokens=int(round(r.prefix_tokens * (1.0 - ratio))),
            suffix_tokens=int(round(r.suffix_tokens * (1.0 - ratio))),
            output_tokens=r.output_tokens,
            arrival_s=r.arrival_s,
        )
        server += billed_cost(shrunk, 0, False, params)
    local = local_cost_per_request * len(requests)
    return server, local


def semantic_cache_cost(requests, params, hit_rate, local_cost_per_request):
    """Calculate cost of semantic cache baseline.

    Deterministically treats first round(hit_rate * n) requests (by arrival time)
    as cache hits (skipped, 0 cost). Rest are full-cost misses with no provider caching.
    Adds flat local cost per request.

    Args:
        requests: List of Request objects
        params: CacheParams
        hit_rate: Cache hit rate (0.0 to 1.0)
        local_cost_per_request: Fixed cost per request

    Returns:
        (server_cost, local_cost) tuple
    """
    ordered = sorted(requests, key=lambda r: r.arrival_s)
    n_hits = round(hit_rate * len(ordered))
    server = 0.0
    for i, r in enumerate(ordered):
        if i < n_hits:
            continue  # hit: call skipped
        server += billed_cost(r, 0, False, params)
    local = local_cost_per_request * len(requests)
    return server, local
