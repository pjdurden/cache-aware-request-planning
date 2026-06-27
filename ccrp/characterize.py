"""Provider cache characterization harness (telemetry parsing)."""


def parse_anthropic_usage(usage):
    """Parse Anthropic usage dict to normalized format.

    Maps cache_read_input_tokens and cache_creation_input_tokens
    to cache_read_tokens and cache_write_tokens.
    """
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_write_tokens": usage.get("cache_creation_input_tokens", 0),
    }


def parse_openai_usage(usage):
    """Parse OpenAI usage dict to normalized format.

    Maps prompt_tokens and nested prompt_tokens_details.cached_tokens
    to input_tokens and cache_read_tokens.
    """
    details = usage.get("prompt_tokens_details", {})
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "cache_read_tokens": details.get("cached_tokens", 0),
    }


def infer_min_prefix_tokens(probe_results):
    """Infer minimum prefix tokens needed for cache hit.

    Given a list of (prefix_tokens, observed_hit_on_repeat) tuples,
    return the smallest prefix length that ever produced a cache hit,
    or 0 if no hits observed.
    """
    hits = [n for n, hit in probe_results if hit]
    return min(hits) if hits else 0


def probe_provider(send_fn, prefix_tokens_list):
    """Probe provider for cache behavior at different prefix lengths.

    For each prefix length, calls send_fn twice and reports whether
    the second call showed a cache read. send_fn is injected
    (no real network in tests).
    """
    results = []
    for n in prefix_tokens_list:
        send_fn(n)                      # warm
        usage = send_fn(n)              # repeat
        parsed = usage.get("cache_read_tokens", 0)
        results.append((n, parsed > 0))
    return results
