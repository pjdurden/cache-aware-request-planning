from dataclasses import dataclass


@dataclass(frozen=True)
class CacheParams:
    ttl_s: float
    min_prefix_tokens: int
    read_discount: float       # fraction of full price for a cache read, e.g. 0.1
    write_multiplier: float    # multiple of full price for a cache write, e.g. 1.25
    input_price_per_1k: float
    output_price_per_1k: float


@dataclass
class Request:
    id: str
    prefix_key: str            # canonical identity of the cacheable prefix
    prefix_tokens: int         # length of the cacheable prefix
    suffix_tokens: int         # variable tail, never cacheable
    output_tokens: int
    arrival_s: float


def billed_cost(req, cached_prefix_tokens, wrote_prefix, params):
    read = cached_prefix_tokens
    written = req.prefix_tokens - cached_prefix_tokens if wrote_prefix else 0
    full = req.prefix_tokens - cached_prefix_tokens - written
    prefix_units = read * params.read_discount + written * params.write_multiplier + full
    input_units = prefix_units + req.suffix_tokens
    input_cost = input_units * params.input_price_per_1k / 1000.0
    output_cost = req.output_tokens * params.output_price_per_1k / 1000.0
    return input_cost + output_cost


def intervention_net_savings(baseline_cost, intervention_cost, local_cost):
    """Dollars saved by an intervention after charging its local-compute cost."""
    return baseline_cost - intervention_cost - local_cost
