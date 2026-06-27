from ccrp.workloads import agentic_tooluse, multi_turn_chat


def test_agentic_shares_prefix_within_run():
    reqs = agentic_tooluse(n_runs=2, steps_per_run=3, schema_tokens=2000,
                           step_suffix_tokens=150, output_tokens=80, seed=1)
    assert len(reqs) == 6
    keys = {r.prefix_key for r in reqs}
    assert keys == {"run0", "run1"}
    run0 = [r for r in reqs if r.prefix_key == "run0"]
    assert all(r.prefix_tokens == 2000 for r in run0)


def test_agentic_is_deterministic():
    a = agentic_tooluse(2, 3, 2000, 150, 80, seed=7)
    b = agentic_tooluse(2, 3, 2000, 150, 80, seed=7)
    assert [r.arrival_s for r in a] == [r.arrival_s for r in b]


def test_chat_prefix_grows_per_turn():
    reqs = multi_turn_chat(n_sessions=1, turns_per_session=3, system_tokens=1500,
                           turn_growth_tokens=200, think_time_s=30.0, seed=2)
    s0 = sorted([r for r in reqs if r.prefix_key == "sess0"], key=lambda r: r.arrival_s)
    prefixes = [r.prefix_tokens for r in s0]
    assert prefixes == sorted(prefixes)
    assert prefixes[0] == 1500
