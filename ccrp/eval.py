from dataclasses import dataclass
from ccrp.simulate import simulate_order
from ccrp.scheduler import greedy_schedule
from ccrp.baselines import compression_cost, semantic_cache_cost


@dataclass
class Point:
    method: str
    server_cost: float
    local_cost: float
    quality_cost: float


def run_experiment(requests, params, service_time_s, max_slack_s, *,
                   compression_ratio, compression_quality_cost, compression_local,
                   semantic_hit_rate, semantic_quality_cost, semantic_local):
    naive_order = sorted(requests, key=lambda r: r.arrival_s)
    naive_cost = simulate_order(naive_order, params, service_time_s)

    shaped = greedy_schedule(requests, params, service_time_s, max_slack_s)
    shaped_cost = simulate_order(shaped, params, service_time_s)

    comp_server, comp_local = compression_cost(
        requests, params, compression_ratio, compression_local)
    sem_server, sem_local = semantic_cache_cost(
        requests, params, semantic_hit_rate, semantic_local)

    return [
        Point("naive", naive_cost, 0.0, 0.0),
        Point("cache_shaping", shaped_cost, 0.0, 0.0),
        Point("compression", comp_server, comp_local, compression_quality_cost),
        Point("semantic_cache", sem_server, sem_local, semantic_quality_cost),
    ]
