import json
import pathlib
from ccrp.characterize import (
    parse_anthropic_usage,
    parse_openai_usage,
    infer_min_prefix_tokens,
    probe_provider,
)

FIX = pathlib.Path(__file__).parent / "fixtures"


def test_parse_anthropic_usage():
    usage = json.loads((FIX / "anthropic_usage.json").read_text())
    out = parse_anthropic_usage(usage)
    assert out == {"input_tokens": 50, "cache_read_tokens": 2000, "cache_write_tokens": 0}


def test_parse_openai_usage():
    usage = json.loads((FIX / "openai_usage.json").read_text())
    out = parse_openai_usage(usage)
    assert out == {"input_tokens": 2200, "cache_read_tokens": 2048}


def test_infer_min_prefix_tokens():
    probes = [(512, False), (1024, True), (2048, True)]
    assert infer_min_prefix_tokens(probes) == 1024


def test_probe_provider_uses_injected_send_fn():
    # send_fn: second identical call returns a cache read when prefix >= 1024
    calls = {"n": 0}

    def send_fn(prefix_tokens):
        calls["n"] += 1
        second = calls["n"] % 2 == 0
        read = 2000 if (second and prefix_tokens >= 1024) else 0
        return {"input_tokens": prefix_tokens, "cache_read_tokens": read}

    results = probe_provider(send_fn, [512, 1024])
    assert results == [(512, False), (1024, True)]
