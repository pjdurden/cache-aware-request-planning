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

Reproducibility (record once, replay forever)
-----------------------------------------------
Plain runs (no flags) are live and record nothing, same as always. Two extra
flags make a run's provider traffic reproducible offline via `cassette-fn`
(https://pypi.org/project/cassette-fn/), a function-boundary VCR:

    --record            run live as usual, AND save every provider response
                         used to recover the numbers to a cassette file.
    --replay            skip the network entirely and answer every probe from
                         a previously recorded cassette. Needs no API key and
                         makes no request. The recovered numbers are exactly
                         what the recorded run measured.
    --cassette PATH     cassette file to record to / replay from (default:
                         docs/paper/characterize_live.cassette.json).

Typical workflow to make a paper number checkable without spending money
again:

    .venv/bin/python -m pip install -e ".[providers]"
    ANTHROPIC_API_KEY=sk-ant-... .venv/bin/python docs/paper/characterize_live.py --ttl --record
    # commit docs/paper/characterize_live.cassette.json
    .venv/bin/python docs/paper/characterize_live.py --ttl --replay   # no key, no network

--replay prints a banner making clear the numbers are replayed, not fresh, and
skips the multi-minute TTL sleeps (there is nothing to wait for when the
responses are already recorded). CASSETTE_FN_MODE in the environment overrides
--record/--replay if both happen to be set; see the cassette-fn docs.
"""
import argparse
import math
import os
import sys
import time

# Cassette file for --record/--replay, resolved relative to this script so it
# does not depend on the caller's working directory.
DEFAULT_CASSETTE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "characterize_live.cassette.json")

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


def probe_anthropic(model, do_ttl, tape=None, replaying=False):
    import anthropic

    client = anthropic.Anthropic()
    print("=" * 66)
    print(f"ANTHROPIC cache characterization (model: {model})")
    print("=" * 66)

    print("Sizing probe prefixes via count_tokens ...")

    # The SDK client is bound via closure (not passed as a call argument) so
    # that a cassette wrap never has to serialize it: `model` and
    # `target_tokens` stay the plain, hashed, recorded arguments; `client` is
    # captured the same way `anthropic_send_fn` below already captures it for
    # `send_fn`. (cassette-fn's Tape.wrap records the exact call arguments it
    # is given, so an unserializable SDK client must never be one of them --
    # `normalize` only affects the cache *key*, not what gets stored.)
    def _make_filler(model, target_tokens):
        return make_anthropic_filler(client, model, target_tokens)

    filler_call = tape.wrap(_make_filler) if tape is not None else _make_filler

    fillers = {}
    for target in TARGET_PREFIX_TOKENS:
        fillers[target] = filler_call(model, target)

    send_fn = anthropic_send_fn(client, model, fillers)
    if tape is not None:
        send_fn = tape.wrap(send_fn)
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
        ttl_note = probe_anthropic_ttl(send_fn, large, replaying=replaying)

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


def probe_anthropic_ttl(send_fn, size, replaying=False):
    """Coarsely bound the TTL: confirm warm at +30s, check expiry at +330s.

    In replay mode there is nothing to wait for -- the responses already
    happened during the recorded run -- so the sleeps are skipped and the
    three `send_fn(size)` calls are made back-to-back, consuming the next
    three recorded entries for this cassette key in order.
    """
    if replaying:
        print("\nProbing TTL (replayed from cassette; sleeps skipped) ...")
    else:
        print("\nProbing TTL (this waits about 6 minutes) ...")
    send_fn(size)  # warm
    if not replaying:
        time.sleep(30)
    warm = send_fn(size)["cache_read_tokens"] > 0
    print(f"  +30s: still warm = {warm}")
    if not replaying:
        time.sleep(330)
    expired = send_fn(size)["cache_read_tokens"] == 0
    print(f"  +360s: expired = {expired}")
    if warm and expired:
        return "300 (bounded: warm at 30s, expired by 360s)"
    if warm and not expired:
        return ">360 (still warm at 360s; TTL longer than probed)"
    return "uncertain (no warm hit at 30s; rerun)"


def probe_openai(model, tape=None, replaying=False):
    import openai

    # Same reasoning as the Anthropic client above: in replay mode every
    # provider call is served from the cassette, so a placeholder key is
    # enough to construct the client (the OpenAI SDK, unlike the Anthropic
    # one, refuses to construct with no key at all).
    if replaying:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "cassette-replay"))
    else:
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

    if tape is not None:
        send_once = tape.wrap(send_once)

    send_once()  # warm
    second = send_once()
    print(f"Repeat call read {second['cache_read_tokens']} cached tokens "
          f"(of {second['input_tokens']} input tokens).")
    print("OpenAI cached-input discount and TTL are published values; consult "
          "current OpenAI pricing for the exact discount.")


def _cassette_dir_and_name(path):
    """Split a cassette file path into cassette-fn's (dir, name) arguments.

    cassette-fn builds the on-disk path as ``{dir}/{name}.json``; this just
    inverts that so callers can hand the script one plain --cassette path.
    """
    directory = os.path.dirname(path) or "."
    base = os.path.basename(path)
    name = base[:-len(".json")] if base.endswith(".json") else base
    return directory, name


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--anthropic-model", default="claude-haiku-4-5")
    parser.add_argument("--openai", action="store_true", help="also probe OpenAI")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--ttl", action="store_true",
                        help="also bound the TTL (waits ~6 minutes)")
    parser.add_argument("--record", action="store_true",
                        help="run live as usual, and also save provider responses "
                             "to --cassette so the run can be replayed offline later")
    parser.add_argument("--replay", action="store_true",
                        help="answer every probe from --cassette instead of the network; "
                             "needs no API key and makes no request")
    parser.add_argument("--cassette", default=DEFAULT_CASSETTE,
                        help="cassette file for --record/--replay "
                             "(default: %(default)s)")
    args = parser.parse_args()

    if args.record and args.replay:
        parser.error("--record and --replay are mutually exclusive")

    tape = None
    if args.record or args.replay:
        import cassette_fn

        cassette_dir, cassette_name = _cassette_dir_and_name(args.cassette)
        tape = cassette_fn.tape(
            dir=cassette_dir,
            name=cassette_name,
            mode="record" if args.record else "replay",
        )
        if args.replay:
            print("=" * 66)
            print(f"REPLAYING from cassette: {args.cassette}")
            print("These are numbers from a previously recorded live run, NOT a "
                  "fresh measurement. No network request is made.")
            print("=" * 66)

    if not args.replay and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in the environment first.", file=sys.stderr)
        return 1

    probe_anthropic(args.anthropic_model, args.ttl, tape=tape, replaying=args.replay)

    if args.openai:
        if not args.replay and not os.environ.get("OPENAI_API_KEY"):
            print("\nSkipping OpenAI: OPENAI_API_KEY not set.", file=sys.stderr)
        else:
            probe_openai(args.openai_model, tape=tape, replaying=args.replay)

    print("\nDone. Paste the recovered min_prefix_tokens (and TTL if probed) into "
          "PARAMS in docs/paper/run_eval.py, then rerun it to refresh the tables.")

    if args.record:
        tape.save()
        print(f"\nRecorded cassette written to {args.cassette}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
