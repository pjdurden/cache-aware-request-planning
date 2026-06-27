import random
from ccrp.cost_model import Request


def agentic_tooluse(n_runs, steps_per_run, schema_tokens, step_suffix_tokens,
                    output_tokens, seed):
    rng = random.Random(seed)
    reqs = []
    # Interleave steps across runs so arrival order does not match prefix order.
    for step in range(steps_per_run):
        for run in range(n_runs):
            arrival = float(step * n_runs + run) + rng.random()
            reqs.append(Request(
                id=f"run{run}-step{step}",
                prefix_key=f"run{run}",
                prefix_tokens=schema_tokens,
                suffix_tokens=step_suffix_tokens,
                output_tokens=output_tokens,
                arrival_s=arrival,
            ))
    return reqs


def multi_turn_chat(n_sessions, turns_per_session, system_tokens, turn_growth_tokens,
                    think_time_s, seed):
    rng = random.Random(seed)
    reqs = []
    for sess in range(n_sessions):
        base = rng.random()
        for turn in range(turns_per_session):
            arrival = base + turn * think_time_s
            reqs.append(Request(
                id=f"sess{sess}-turn{turn}",
                prefix_key=f"sess{sess}",
                prefix_tokens=system_tokens + turn * turn_growth_tokens,
                suffix_tokens=turn_growth_tokens,
                output_tokens=turn_growth_tokens,
                arrival_s=arrival,
            ))
    return reqs
