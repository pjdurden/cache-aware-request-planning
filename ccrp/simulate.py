from ccrp.cost_model import billed_cost
from ccrp.cache_sim import CacheSim


def simulate_order(order, params, service_time_s):
    if not order:
        return 0.0
    sim = CacheSim(params)
    now = order[0].arrival_s
    total = 0.0
    for req in order:
        now = max(now, req.arrival_s)
        res = sim.access(req.prefix_key, req.prefix_tokens, now)
        total += billed_cost(req, res.cached_prefix_tokens, res.wrote_prefix, params)
        now += service_time_s
    return total
