from ccrp.cache_sim import CacheSim


def greedy_schedule(requests, params, service_time_s, max_slack_s):
    sim = CacheSim(params)
    remaining = list(requests)
    order = []
    now = min((r.arrival_s for r in remaining), default=0.0)
    last_key = None

    while remaining:
        arrived = [r for r in remaining if r.arrival_s <= now]
        if not arrived:
            now = min(r.arrival_s for r in remaining)
            continue

        # Deadline guard: anything at or past its deadline must go first.
        due = [r for r in arrived if r.arrival_s + max_slack_s <= now]
        if due:
            pick = min(due, key=lambda r: (r.arrival_s + max_slack_s, r.arrival_s))
        else:
            warm = [r for r in arrived if sim.is_warm(r.prefix_key, now)]
            if warm:
                pick = min(warm, key=lambda r: r.arrival_s)
            elif last_key is not None and any(r.prefix_key == last_key for r in arrived):
                pick = min((r for r in arrived if r.prefix_key == last_key),
                           key=lambda r: r.arrival_s)
            else:
                pick = min(arrived, key=lambda r: r.arrival_s)

        sim.access(pick.prefix_key, pick.prefix_tokens, now)
        order.append(pick)
        remaining.remove(pick)
        last_key = pick.prefix_key
        now += service_time_s

    return order
