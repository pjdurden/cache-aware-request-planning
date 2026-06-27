"""Reproduce the paper's tables and figure from the ccrp modules.

Run from anywhere after installing the package in editable mode:

    /home/pjsump/.venv/bin/python -m pip install -e ".[figures]"
    /home/pjsump/.venv/bin/python docs/paper/run_eval.py

All numbers are deterministic (fixed workload seeds), so this regenerates the
exact tables in cache-aware-client-request-planning-paper.md and writes the
pareto figure next to this script. No network access and no live model are
used: the study is simulation under representative public cache parameters.
"""
from pathlib import Path

from ccrp.cost_model import CacheParams
from ccrp.workloads import agentic_tooluse, multi_turn_chat
from ccrp.eval import run_experiment
from ccrp.metrics import reduction_pct
from ccrp.simulate import simulate_order
from ccrp.scheduler import greedy_schedule

# Representative public cache parameters (Anthropic-like: explicit cache, 5 min
# TTL, 1024 min prefix, 0.1x read, 1.25x write). Prices are dollars per 1k
# tokens (3.0 in, 15.0 out), i.e. 3.0 and 15.0 dollars per million.
PARAMS = CacheParams(
    ttl_s=300.0, min_prefix_tokens=1024, read_discount=0.1,
    write_multiplier=1.25, input_price_per_1k=3.0, output_price_per_1k=15.0,
)

# Large slack so the scheduler may reorder freely; the deadline guard is not the
# variable under study here.
SLACK = 1_000_000.0


def naive_cost(reqs, service_time_s):
    order = sorted(reqs, key=lambda r: r.arrival_s)
    return simulate_order(order, PARAMS, service_time_s)


def shaped_cost(reqs, service_time_s):
    order = greedy_schedule(reqs, PARAMS, service_time_s, SLACK)
    return simulate_order(order, PARAMS, service_time_s)


def agentic_sweep():
    print("=" * 64)
    print("AGENTIC TOOL-USE: sweep per-step service time (s).")
    print("n_runs=4, steps_per_run=4, schema=3000 tok. In naive arrival")
    print("order a run's steps are 4*service_time apart. TTL=300.")
    print("=" * 64)
    reqs = agentic_tooluse(n_runs=4, steps_per_run=4, schema_tokens=3000,
                           step_suffix_tokens=150, output_tokens=80, seed=3)
    print(f"{'service_s':>10} {'naive$':>10} {'shaped$':>10} {'reduction%':>12}")
    for st in [10, 40, 75, 100, 150, 300]:
        n = naive_cost(reqs, st)
        s = shaped_cost(reqs, st)
        print(f"{st:>10} {n:>10.3f} {s:>10.3f} {reduction_pct(n, s):>12.1f}")
    return reqs


def chat_sweep():
    print()
    print("=" * 64)
    print("MULTI-TURN CHAT: sweep user think-time (s) between turns.")
    print("n_sessions=6, turns=5, system=1500 tok. Sessions interleaved.")
    print("=" * 64)
    print(f"{'think_s':>10} {'naive$':>10} {'shaped$':>10} {'reduction%':>12}")
    for tt in [10, 60, 120, 300, 600]:
        chat = multi_turn_chat(n_sessions=6, turns_per_session=5,
                               system_tokens=1500, turn_growth_tokens=200,
                               think_time_s=float(tt), seed=2)
        # Service time (model latency) is small relative to think time.
        n = naive_cost(chat, 2.0)
        s = shaped_cost(chat, 2.0)
        print(f"{tt:>10} {n:>10.3f} {s:>10.3f} {reduction_pct(n, s):>12.1f}")


def comparison_plane(reqs):
    print()
    print("=" * 64)
    print("COMPARISON PLANE at the agentic operating point (service=100s).")
    print("=" * 64)
    points = run_experiment(
        reqs, PARAMS, service_time_s=100.0, max_slack_s=SLACK,
        compression_ratio=0.4, compression_quality_cost=0.15,
        compression_local=0.01, semantic_hit_rate=0.3,
        semantic_quality_cost=0.1, semantic_local=0.002,
    )
    print(f"{'method':>16} {'server$':>10} {'local$':>8} {'quality':>8}")
    for p in sorted(points, key=lambda p: p.server_cost):
        print(f"{p.method:>16} {p.server_cost:>10.3f} "
              f"{p.local_cost:>8.3f} {p.quality_cost:>8.2f}")
    return points


def write_figure(points):
    try:
        from ccrp.figures import pareto_plane
    except Exception as e:
        print(f"\nfigure skipped (matplotlib not installed?): {e}")
        return
    out = Path(__file__).resolve().parent / "pareto_plane.png"
    pareto_plane(points, str(out))
    print(f"\nfigure written: {out}")


def main():
    reqs = agentic_sweep()
    chat_sweep()
    points = comparison_plane(reqs)
    write_figure(points)


if __name__ == "__main__":
    main()
