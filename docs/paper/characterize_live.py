"""Live provider cache characterization (the calibration step for the paper).

Recovers the cache parameters that ARE measurable from real provider telemetry
(minimum cacheable prefix, and optionally the TTL) by sending tiny probe
requests and reading the usage payload. The discount and per-token price are
published pricing, not measurable from token counts, so they are pulled from
the pricing table and printed alongside the recovered values.

This costs a small amount of real money (a few cents on Haiku) and uses YOUR
API keys, so you run it, not the harness. Set the keys in the environment:

    ANTHROPIC_API_KEY=...  (required for the Anthropic probe)
    OPENAI_API_KEY=...     (optional, enables the OpenAI probe with --openai)

    .venv/bin/python -m pip install -e ".[providers]"
    ANTHROPIC_API_KEY=sk-ant-... .venv/bin/python docs/paper/characterize_live.py

Add --ttl to also bound the TTL (this waits several minutes). Add --openai to
also probe OpenAI (best effort: automatic caching, adjust --openai-model if the
default is unavailable on your account).
"""
import argparse
import math
import os
import sys
import time

from ccrp.characterize import (
    parse_anthropic_usage,
    parse_openai_usage,
    infer_min_prefix_tokens,
    probe_provider,
)

# Probe sizes bracket the documented Haiku/Opus minimum (4096 tokens) so the
# recovered threshold is meaningful, not just confirmed.
TARGET_PREFIX_TOKENS = [1024, 2048, 4096, 6144, 8192]

# Published pricing (dollars per 1k tokens) and cache economics, from provider
# docs. These are NOT measurable from telemetry; the probe confirms caching
# works and recovers the prefix/TTL, and these fill in the rest of CacheParams.
HAIKU_INPUT_PER_1K = 0.001      # $1.00 / 1M
HAIKU_OUTPUT_PER_1K = 0.005     # $5.00 / 1M
ANTHROPIC_READ_DISCOUNT = 0.1   # cache read is ~0.1x input price
ANTHROPIC_WRITE_MULTIPLIER = 1.25  # 5-minute cache write is ~1.25x


def make_anthropic_filler(client, model, target_tokens):
    """Build filler text of approximately target_tokens, sized via count_tokens."""
    word = "characterization "
    sample_repeats = 200
    sample = word * sample_repeats
    counted = client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": sample}]
    ).input_tokens
    tokens_per_repeat = counted / sample_repeats
    repeats = max(1, math.ceil(target_tokens / tokens_per_repeat))
    return word * repeats


def anthropic_send_fn(client, model, fillers):
    def send_fn(n):
        resp = client.messages.create(
            model=model,
            max_tokens=1,
            system=[{
                "type": "text",
                "text": fillers[n],
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": "hi"}],
        )
        u = resp.usage
        usage = {
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        return parse_anthropic_usage(usage)
    return send_fn


def probe_anthropic(model, do_ttl):
    import anthropic

    client = anthropic.Anthropic()
    print("=" * 66)
    print(f"ANTHROPIC cache characterization (model: {model})")
    print("=" * 66)

    print("Sizing probe prefixes via count_tokens ...")
    fillers = {}
    for target in TARGET_PREFIX_TOKENS:
        fillers[target] = make_anthropic_filler(client, model, target)

    send_fn = anthropic_send_fn(client, model, fillers)
    print(f"{'prefix_tokens':>14} {'hit_on_repeat':>14}")
    results = probe_provider(send_fn, TARGET_PREFIX_TOKENS)
    for n, hit in results:
        print(f"{n:>14} {str(hit):>14}")

    min_prefix = infer_min_prefix_tokens(results)
    print(f"\nRecovered minimum cacheable prefix: {min_prefix} tokens")

    # Confirm read/write magnitudes at the largest size (warm then read).
    large = TARGET_PREFIX_TOKENS[-1]
    first = send_fn(large)
    second = send_fn(large)
    print(f"At {large} tokens: first call wrote {first['cache_write_tokens']} "
          f"tokens, repeat read {second['cache_read_tokens']} tokens from cache.")

    ttl_note = "300 (from docs; not probed)"
    if do_ttl:
        ttl_note = probe_anthropic_ttl(send_fn, large)

    print("\nSuggested calibrated CacheParams (recovered + published pricing):")
    print(
        "CacheParams("
        "ttl_s=300.0, "
        f"min_prefix_tokens={min_prefix or 4096}, "
        f"read_discount={ANTHROPIC_READ_DISCOUNT}, "
        f"write_multiplier={ANTHROPIC_WRITE_MULTIPLIER}, "
        f"input_price_per_1k={HAIKU_INPUT_PER_1K}, "
        f"output_price_per_1k={HAIKU_OUTPUT_PER_1K})"
    )
    print(f"(TTL: {ttl_note}. read_discount/write_multiplier/prices are published "
          "values, not telemetry-recoverable.)")


def probe_anthropic_ttl(send_fn, size):
    """Coarsely bound the TTL: confirm warm at +30s, check expiry at +330s."""
    print("\nProbing TTL (this waits about 6 minutes) ...")
    send_fn(size)  # warm
    time.sleep(30)
    warm = send_fn(size)["cache_read_tokens"] > 0
    print(f"  +30s: still warm = {warm}")
    time.sleep(330)
    expired = send_fn(size)["cache_read_tokens"] == 0
    print(f"  +360s: expired = {expired}")
    if warm and expired:
        return "300 (bounded: warm at 30s, expired by 360s)"
    if warm and not expired:
        return ">360 (still warm at 360s; TTL longer than probed)"
    return "uncertain (no warm hit at 30s; rerun)"


def probe_openai(model):
    import openai

    client = openai.OpenAI()
    print("\n" + "=" * 66)
    print(f"OPENAI cache characterization (model: {model}; best effort)")
    print("=" * 66)
    print("OpenAI caches automatically for prompts of ~1024+ tokens.")

    filler = "characterization " * 6000  # safely over the ~1024-token floor

    def send_once():
        resp = client.chat.completions.create(
            model=model,
            max_tokens=1,
            messages=[
                {"role": "system", "content": filler},
                {"role": "user", "content": "hi"},
            ],
        )
        u = resp.usage
        details = getattr(u, "prompt_tokens_details", None)
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "prompt_tokens_details": {
                "cached_tokens": getattr(details, "cached_tokens", 0) or 0
            },
        }
        return parse_openai_usage(usage)

    send_once()  # warm
    second = send_once()
    print(f"Repeat call read {second['cache_read_tokens']} cached tokens "
          f"(of {second['input_tokens']} input tokens).")
    print("OpenAI cached-input discount and TTL are published values; consult "
          "current OpenAI pricing for the exact discount.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anthropic-model", default="claude-haiku-4-5")
    parser.add_argument("--openai", action="store_true", help="also probe OpenAI")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--ttl", action="store_true",
                        help="also bound the TTL (waits ~6 minutes)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in the environment first.", file=sys.stderr)
        return 1

    probe_anthropic(args.anthropic_model, args.ttl)

    if args.openai:
        if not os.environ.get("OPENAI_API_KEY"):
            print("\nSkipping OpenAI: OPENAI_API_KEY not set.", file=sys.stderr)
        else:
            probe_openai(args.openai_model)

    print("\nDone. Paste the recovered min_prefix_tokens (and TTL if probed) into "
          "PARAMS in docs/paper/run_eval.py, then rerun it to refresh the tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
