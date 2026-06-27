from ccrp.cost_model import CacheParams
from ccrp.workloads import agentic_tooluse
from ccrp.eval import run_experiment, Point


def test_run_experiment_returns_all_methods_and_shaping_beats_naive():
    params = CacheParams(300.0, 1024, 0.1, 1.25, 3.0, 15.0)
    reqs = agentic_tooluse(n_runs=4, steps_per_run=4, schema_tokens=3000,
                           step_suffix_tokens=150, output_tokens=80, seed=3)
    points = run_experiment(
        reqs, params, service_time_s=100.0, max_slack_s=100000.0,
        compression_ratio=0.4, compression_quality_cost=0.15, compression_local=0.01,
        semantic_hit_rate=0.3, semantic_quality_cost=0.1, semantic_local=0.002,
    )
    by = {p.method: p for p in points}
    assert set(by) == {"naive", "cache_shaping", "compression", "semantic_cache"}
    assert by["cache_shaping"].server_cost < by["naive"].server_cost
    assert by["cache_shaping"].quality_cost == 0.0
