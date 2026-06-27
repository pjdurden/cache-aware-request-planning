# Cache-Aware Client-Side Request Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the research artifact that produces the paper's numbers and figures: a calibrated provider-cache simulator, a greedy prefix-clustering request scheduler, two baselines, and an evaluation harness.

**Architecture:** Pure-Python package working in token counts (tokenizer-agnostic). A `CostModel` prices a request given cache state. A `CacheSim` models a provider's opaque prompt cache (TTL, minimum prefix length, read discount, write multiplier). A `GreedyScheduler` reorders a request stream to maximize cache hits under TTL and latency-slack constraints. Workload generators, analytic baselines, and an eval/metrics/figures layer sit on top. A separate `characterize` module recovers `CacheParams` from real-API usage telemetry (tested against recorded payloads).

**Tech Stack:** Python 3.11+, dataclasses (no pydantic), pytest, matplotlib (figures only). Optional extras `anthropic` / `openai` used only by the characterization harness, never imported by core.

## Global Constraints

- Python 3.10+ (environment max is 3.10.12; code uses no 3.11-only features). Build/run inside the project-local `.venv`.
- Core modules (`cost_model`, `canonicalize`, `cache_sim`, `clustering`, `scheduler`, `workloads`, `baselines`, `metrics`) MUST NOT import `anthropic`, `openai`, or any network library. Only `characterize` may, behind a function parameter.
- Money is `float` dollars throughout. Token counts are `int`.
- Package name: `ccrp` (cache-aware client request planning).
- No em dashes in any docstring or printed string (author preference; use commas or parentheses).
- Every task ends green: `pytest -q` passes before commit.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `ccrp/__init__.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `ccrp` with `__version__: str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
import ccrp


def test_package_imports_and_has_version():
    assert isinstance(ccrp.__version__, str)
    assert ccrp.__version__ != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ccrp'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "ccrp"
version = "0.1.0"
description = "Cache-aware client-side request planning for black-box LLM APIs"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
figures = ["matplotlib>=3.8"]
providers = ["anthropic>=0.40", "openai>=1.40"]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["ccrp*"]
```

```python
# ccrp/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e ".[dev]" && pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml ccrp/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold ccrp package"
```

---

### Task 2: Cost model and core data types

**Files:**
- Create: `ccrp/cost_model.py`
- Test: `tests/test_cost_model.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass(frozen=True) CacheParams(ttl_s: float, min_prefix_tokens: int, read_discount: float, write_multiplier: float, input_price_per_1k: float, output_price_per_1k: float)`
  - `@dataclass Request(id: str, prefix_key: str, prefix_tokens: int, suffix_tokens: int, output_tokens: int, arrival_s: float)`
  - `billed_cost(req: Request, cached_prefix_tokens: int, wrote_prefix: bool, params: CacheParams) -> float`
  - `BREAK_EVEN_HELP` not needed; add `intervention_net_savings(baseline_cost: float, intervention_cost: float, local_cost: float) -> float`

Cost rules:
- Input has a cacheable `prefix_tokens` and a variable `suffix_tokens`.
- `cached_prefix_tokens` is how many prefix tokens were served from cache (a read, discounted). `wrote_prefix` is True when the remaining prefix was written to cache this call (write multiplier applies to written tokens).
- Suffix is always full price. Output is always full price.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cost_model.py
import math
from ccrp.cost_model import (
    CacheParams,
    Request,
    billed_cost,
    intervention_net_savings,
)

PARAMS = CacheParams(
    ttl_s=300.0,
    min_prefix_tokens=1024,
    read_discount=0.1,
    write_multiplier=1.25,
    input_price_per_1k=3.0,
    output_price_per_1k=15.0,
)


def _req(prefix=2000, suffix=200, output=100):
    return Request("r", "k", prefix, suffix, output, 0.0)


def test_full_miss_no_cache_write():
    # nothing cached, nothing written: pay full input + output
    r = _req()
    cost = billed_cost(r, cached_prefix_tokens=0, wrote_prefix=False, params=PARAMS)
    expected = (2000 + 200) * 3.0 / 1000 + 100 * 15.0 / 1000
    assert math.isclose(cost, expected)


def test_full_write_applies_multiplier_to_prefix():
    r = _req()
    cost = billed_cost(r, cached_prefix_tokens=0, wrote_prefix=True, params=PARAMS)
    expected = (2000 * 1.25 + 200) * 3.0 / 1000 + 100 * 15.0 / 1000
    assert math.isclose(cost, expected)


def test_full_hit_applies_read_discount_to_prefix():
    r = _req()
    cost = billed_cost(r, cached_prefix_tokens=2000, wrote_prefix=False, params=PARAMS)
    expected = (2000 * 0.1 + 200) * 3.0 / 1000 + 100 * 15.0 / 1000
    assert math.isclose(cost, expected)


def test_net_savings_subtracts_local_cost():
    assert math.isclose(
        intervention_net_savings(baseline_cost=1.0, intervention_cost=0.6, local_cost=0.1),
        0.3,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cost_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ccrp.cost_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# ccrp/cost_model.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cost_model.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add ccrp/cost_model.py tests/test_cost_model.py
git commit -m "feat: cost model and core request/cache-params types"
```

---

### Task 3: Prefix canonicalization

**Files:**
- Create: `ccrp/canonicalize.py`
- Test: `tests/test_canonicalize.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `canonical_prefix(parts: list[dict | str]) -> str` — deterministic string for a prefix made of system text and tool schemas, stable across nondeterministic key ordering and whitespace.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonicalize.py
from ccrp.canonicalize import canonical_prefix


def test_key_order_does_not_change_canonical_form():
    a = canonical_prefix([{"name": "f", "params": {"x": 1, "y": 2}}])
    b = canonical_prefix([{"params": {"y": 2, "x": 1}, "name": "f"}])
    assert a == b


def test_whitespace_is_normalized_in_text_parts():
    a = canonical_prefix(["You   are\n a   bot"])
    b = canonical_prefix(["You are a bot"])
    assert a == b


def test_different_content_differs():
    a = canonical_prefix(["system A", {"name": "f"}])
    b = canonical_prefix(["system B", {"name": "f"}])
    assert a != b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_canonicalize.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# ccrp/canonicalize.py
import json


def _normalize_text(s):
    return " ".join(s.split())


def canonical_prefix(parts):
    """Stable string identity for a cacheable prefix.

    Sorts dict keys (kills nondeterministic JSON ordering) and collapses
    whitespace in text parts (kills incidental formatting differences).
    """
    chunks = []
    for part in parts:
        if isinstance(part, str):
            chunks.append("T:" + _normalize_text(part))
        else:
            chunks.append("J:" + json.dumps(part, sort_keys=True, separators=(",", ":")))
    return "\n".join(chunks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_canonicalize.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add ccrp/canonicalize.py tests/test_canonicalize.py
git commit -m "feat: prefix canonicalization for byte-stable cache keys"
```

---

### Task 4: Provider cache simulator

**Files:**
- Create: `ccrp/cache_sim.py`
- Test: `tests/test_cache_sim.py`

**Interfaces:**
- Consumes: `CacheParams` from `ccrp.cost_model`.
- Produces:
  - `@dataclass AccessResult(hit: bool, cached_prefix_tokens: int, wrote_prefix: bool)`
  - `class CacheSim(params: CacheParams)` with:
    - `access(self, prefix_key: str, prefix_tokens: int, now_s: float) -> AccessResult`
    - `is_warm(self, prefix_key: str, now_s: float) -> bool`

Behavior:
- If `prefix_tokens < params.min_prefix_tokens`: never cacheable, returns `AccessResult(False, 0, False)`, no state change.
- If warm at `now_s` (stored `warm_until > now_s`): hit, `cached_prefix_tokens = prefix_tokens`, `wrote_prefix = False`. Refreshes `warm_until = now_s + ttl_s`.
- If cacheable but cold: miss, `cached_prefix_tokens = 0`, `wrote_prefix = True`. Sets `warm_until = now_s + ttl_s`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache_sim.py
from ccrp.cost_model import CacheParams
from ccrp.cache_sim import CacheSim

PARAMS = CacheParams(300.0, 1024, 0.1, 1.25, 3.0, 15.0)


def test_short_prefix_never_cacheable():
    sim = CacheSim(PARAMS)
    r = sim.access("k", prefix_tokens=500, now_s=0.0)
    assert (r.hit, r.cached_prefix_tokens, r.wrote_prefix) == (False, 0, False)
    assert sim.is_warm("k", 0.0) is False


def test_cold_miss_writes_then_warm_hit():
    sim = CacheSim(PARAMS)
    first = sim.access("k", 2000, now_s=0.0)
    assert (first.hit, first.wrote_prefix) == (False, True)
    second = sim.access("k", 2000, now_s=10.0)
    assert (second.hit, second.cached_prefix_tokens, second.wrote_prefix) == (True, 2000, False)


def test_expiry_after_ttl_is_a_miss_again():
    sim = CacheSim(PARAMS)
    sim.access("k", 2000, now_s=0.0)
    assert sim.is_warm("k", 299.0) is True
    assert sim.is_warm("k", 301.0) is False
    third = sim.access("k", 2000, now_s=400.0)
    assert (third.hit, third.wrote_prefix) == (False, True)


def test_access_refreshes_ttl():
    sim = CacheSim(PARAMS)
    sim.access("k", 2000, now_s=0.0)
    sim.access("k", 2000, now_s=200.0)   # refresh
    assert sim.is_warm("k", 450.0) is True  # 200 + 300 = 500 > 450
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cache_sim.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# ccrp/cache_sim.py
from dataclasses import dataclass


@dataclass
class AccessResult:
    hit: bool
    cached_prefix_tokens: int
    wrote_prefix: bool


class CacheSim:
    def __init__(self, params):
        self.params = params
        self._warm_until = {}   # prefix_key -> timestamp

    def is_warm(self, prefix_key, now_s):
        until = self._warm_until.get(prefix_key)
        return until is not None and until > now_s

    def access(self, prefix_key, prefix_tokens, now_s):
        if prefix_tokens < self.params.min_prefix_tokens:
            return AccessResult(False, 0, False)
        if self.is_warm(prefix_key, now_s):
            self._warm_until[prefix_key] = now_s + self.params.ttl_s
            return AccessResult(True, prefix_tokens, False)
        self._warm_until[prefix_key] = now_s + self.params.ttl_s
        return AccessResult(False, 0, True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cache_sim.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add ccrp/cache_sim.py tests/test_cache_sim.py
git commit -m "feat: provider prompt-cache simulator with TTL and min-prefix gate"
```

---

### Task 5: Prefix clustering

**Files:**
- Create: `ccrp/clustering.py`
- Test: `tests/test_clustering.py`

**Interfaces:**
- Consumes: `Request` from `ccrp.cost_model`.
- Produces: `cluster_by_prefix(requests: list[Request]) -> dict[str, list[Request]]` — groups by `prefix_key`, each group ordered by `arrival_s`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clustering.py
from ccrp.cost_model import Request
from ccrp.clustering import cluster_by_prefix


def _r(rid, key, arrival):
    return Request(rid, key, 2000, 100, 50, arrival)


def test_groups_by_prefix_key():
    reqs = [_r("a", "k1", 0.0), _r("b", "k2", 1.0), _r("c", "k1", 2.0)]
    groups = cluster_by_prefix(reqs)
    assert set(groups.keys()) == {"k1", "k2"}
    assert [r.id for r in groups["k1"]] == ["a", "c"]


def test_within_group_sorted_by_arrival():
    reqs = [_r("late", "k", 9.0), _r("early", "k", 1.0)]
    groups = cluster_by_prefix(reqs)
    assert [r.id for r in groups["k"]] == ["early", "late"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_clustering.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# ccrp/clustering.py
from collections import defaultdict


def cluster_by_prefix(requests):
    groups = defaultdict(list)
    for r in requests:
        groups[r.prefix_key].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r.arrival_s)
    return dict(groups)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_clustering.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add ccrp/clustering.py tests/test_clustering.py
git commit -m "feat: cluster requests by canonical prefix key"
```

---

### Task 6: Schedule simulation (cost of an order)

**Files:**
- Create: `ccrp/simulate.py`
- Test: `tests/test_simulate.py`

**Interfaces:**
- Consumes: `Request`, `billed_cost` from `ccrp.cost_model`; `CacheSim` from `ccrp.cache_sim`.
- Produces: `simulate_order(order: list[Request], params: CacheParams, service_time_s: float) -> float` — total billed dollars when requests are served back-to-back in the given order, advancing a clock by `service_time_s` per request, starting at the first request's `arrival_s`.

This isolates "given an order, what does it cost" so both the naive and scheduled orders are priced by identical logic.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_simulate.py
import math
from ccrp.cost_model import CacheParams, Request, billed_cost
from ccrp.simulate import simulate_order

PARAMS = CacheParams(300.0, 1024, 0.1, 1.25, 3.0, 15.0)


def _r(rid, key, arrival):
    return Request(rid, key, 2000, 100, 50, arrival)


def test_repeated_prefix_back_to_back_gets_a_hit_on_second():
    order = [_r("a", "k", 0.0), _r("b", "k", 1.0)]
    total = simulate_order(order, PARAMS, service_time_s=1.0)
    first = billed_cost(order[0], 0, True, PARAMS)        # write
    second = billed_cost(order[1], 2000, False, PARAMS)   # read hit
    assert math.isclose(total, first + second)


def test_distinct_prefixes_never_hit():
    order = [_r("a", "k1", 0.0), _r("b", "k2", 1.0)]
    total = simulate_order(order, PARAMS, service_time_s=1.0)
    each = billed_cost(order[0], 0, True, PARAMS)
    assert math.isclose(total, 2 * each)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# ccrp/simulate.py
from ccrp.cost_model import billed_cost
from ccrp.cache_sim import CacheSim


def simulate_order(order, params, service_time_s):
    if not order:
        return 0.0
    sim = CacheSim(params)
    now = order[0].arrival_s
    total = 0.0
    for req in order:
        now = max(now, req.arrival_s)
        res = sim.access(req.prefix_key, req.prefix_tokens, now)
        total += billed_cost(req, res.cached_prefix_tokens, res.wrote_prefix, params)
        now += service_time_s
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add ccrp/simulate.py tests/test_simulate.py
git commit -m "feat: price a fixed execution order against the cache simulator"
```

---

### Task 7: Greedy prefix-clustering scheduler

**Files:**
- Create: `ccrp/scheduler.py`
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: `Request` from `ccrp.cost_model`; `CacheSim` from `ccrp.cache_sim`.
- Produces: `greedy_schedule(requests: list[Request], params: CacheParams, service_time_s: float, max_slack_s: float) -> list[Request]` — an execution order. Greedy rule: at each step, among requests already arrived, prefer one whose prefix is currently warm; otherwise prefer the one whose prefix matches the most recently served (to start a warm streak); break ties by earliest arrival. A request must never be deferred past `arrival_s + max_slack_s` (deadline guard takes priority over the warm preference).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scheduler.py
from ccrp.cost_model import CacheParams, Request
from ccrp.scheduler import greedy_schedule
from ccrp.simulate import simulate_order

PARAMS = CacheParams(300.0, 1024, 0.1, 1.25, 3.0, 15.0)


def _r(rid, key, arrival):
    return Request(rid, key, 2000, 100, 50, arrival)


def test_interleaved_prefixes_get_grouped_when_slack_allows():
    # arrival order interleaves k1/k2; grouping yields more hits.
    # service_time_s (200) is chosen so that in the naive interleaved order the
    # gap between two same-key accesses (~400s) exceeds the 300s TTL, forcing a
    # cache miss; grouping keeps them inside the TTL window. max_slack_s is large
    # so the scheduler is free to reorder. Naive = 4 writes; grouped = 2 writes + 2 hits.
    reqs = [_r("a", "k1", 0.0), _r("b", "k2", 0.0), _r("c", "k1", 0.0), _r("d", "k2", 0.0)]
    order = greedy_schedule(reqs, PARAMS, service_time_s=200.0, max_slack_s=10000.0)
    keys = [r.prefix_key for r in order]
    # same-key requests end up adjacent
    assert keys in (["k1", "k1", "k2", "k2"], ["k2", "k2", "k1", "k1"])
    assert simulate_order(order, PARAMS, 200.0) < simulate_order(reqs, PARAMS, 200.0)


def test_deadline_guard_prevents_starvation():
    # b has zero slack and must be served before grouping a's together
    reqs = [_r("a1", "k1", 0.0), _r("b", "k2", 0.0), _r("a2", "k1", 0.0)]
    order = greedy_schedule(reqs, PARAMS, service_time_s=1.0, max_slack_s=0.0)
    # with zero slack nothing can be deferred, so order respects arrival/no reorder past deadline
    served_at = {r.id: i for i, r in enumerate(order)}
    assert served_at["b"] <= 1


def test_all_requests_scheduled_once():
    reqs = [_r("a", "k1", 0.0), _r("b", "k2", 1.0), _r("c", "k1", 2.0)]
    order = greedy_schedule(reqs, PARAMS, service_time_s=1.0, max_slack_s=10.0)
    assert sorted(r.id for r in order) == ["a", "b", "c"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scheduler.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# ccrp/scheduler.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scheduler.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add ccrp/scheduler.py tests/test_scheduler.py
git commit -m "feat: greedy prefix-clustering scheduler with deadline guard"
```

---

### Task 8: Workload generators

**Files:**
- Create: `ccrp/workloads.py`
- Test: `tests/test_workloads.py`

**Interfaces:**
- Consumes: `Request` from `ccrp.cost_model`.
- Produces:
  - `agentic_tooluse(n_runs: int, steps_per_run: int, schema_tokens: int, step_suffix_tokens: int, output_tokens: int, seed: int) -> list[Request]` — each run shares one large prefix key `run{r}` of `schema_tokens`; steps within a run reuse it. Arrivals are deterministic from `seed` (interleaved across runs to stress scheduling).
  - `multi_turn_chat(n_sessions: int, turns_per_session: int, system_tokens: int, turn_growth_tokens: int, think_time_s: float, seed: int) -> list[Request]` — each session shares `sess{s}` prefix; prefix grows per turn; arrivals spaced by `think_time_s` (stresses TTL).
- Determinism: use `random.Random(seed)`, no global RNG, no wall clock.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workloads.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workloads.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# ccrp/workloads.py
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
```

Note: in multi-turn chat the growing prefix means a real provider re-caches the longer prefix each turn; the shared `prefix_key` plus growing `prefix_tokens` models "the previous turn's prefix is a prefix of this one." The simulator treats each turn as the same key refreshing TTL, which is the intended approximation for v1 (documented as a limitation in the paper).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workloads.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add ccrp/workloads.py tests/test_workloads.py
git commit -m "feat: agentic tool-use and multi-turn chat workload generators"
```

---

### Task 9: Analytic baselines

**Files:**
- Create: `ccrp/baselines.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `Request`, `CacheParams`, `billed_cost` from `ccrp.cost_model`.
- Produces:
  - `compression_cost(requests, params, ratio: float, local_cost_per_request: float) -> tuple[float, float]` — returns `(server_cost, local_cost)`. Compression shrinks prefix+suffix input tokens by `ratio` (0.4 means 40% removed), no caching, plus a flat local cost per request. Output unchanged.
  - `semantic_cache_cost(requests, params, hit_rate: float, local_cost_per_request: float) -> tuple[float, float]` — on a hit the call is skipped entirely (server cost 0 for that request); on a miss full cost with no provider caching. Deterministic: first `round(hit_rate * n)` requests (by arrival) are treated as hits.

These are analytic (no real models) so the comparison plane is reproducible. Quality cost is reported as a fixed per-baseline constant in eval (Task 10), not computed here.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
import math
from ccrp.cost_model import CacheParams, Request, billed_cost
from ccrp.baselines import compression_cost, semantic_cache_cost

PARAMS = CacheParams(300.0, 1024, 0.1, 1.25, 3.0, 15.0)


def _r(rid, arrival):
    return Request(rid, "k", 2000, 200, 100, arrival)


def test_compression_shrinks_input_and_adds_local_cost():
    reqs = [_r("a", 0.0)]
    server, local = compression_cost(reqs, PARAMS, ratio=0.5, local_cost_per_request=0.01)
    # half the input tokens billed, no caching, output unchanged
    expected_server = (1000 + 100) * 3.0 / 1000 + 100 * 15.0 / 1000
    assert math.isclose(server, expected_server)
    assert math.isclose(local, 0.01)


def test_semantic_cache_skips_hits():
    reqs = [_r("a", 0.0), _r("b", 1.0)]
    server, local = semantic_cache_cost(reqs, PARAMS, hit_rate=0.5, local_cost_per_request=0.002)
    # 1 hit (skipped), 1 miss (full, no provider cache)
    miss = billed_cost(reqs[0], 0, False, PARAMS)
    assert math.isclose(server, miss)
    assert math.isclose(local, 2 * 0.002)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# ccrp/baselines.py
from ccrp.cost_model import Request, billed_cost


def compression_cost(requests, params, ratio, local_cost_per_request):
    server = 0.0
    for r in requests:
        shrunk = Request(
            r.id, r.prefix_key,
            prefix_tokens=int(round(r.prefix_tokens * (1.0 - ratio))),
            suffix_tokens=int(round(r.suffix_tokens * (1.0 - ratio))),
            output_tokens=r.output_tokens,
            arrival_s=r.arrival_s,
        )
        server += billed_cost(shrunk, 0, False, params)
    local = local_cost_per_request * len(requests)
    return server, local


def semantic_cache_cost(requests, params, hit_rate, local_cost_per_request):
    ordered = sorted(requests, key=lambda r: r.arrival_s)
    n_hits = round(hit_rate * len(ordered))
    server = 0.0
    for i, r in enumerate(ordered):
        if i < n_hits:
            continue  # hit: call skipped
        server += billed_cost(r, 0, False, params)
    local = local_cost_per_request * len(requests)
    return server, local
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_baselines.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add ccrp/baselines.py tests/test_baselines.py
git commit -m "feat: analytic compression and semantic-cache baselines"
```

---

### Task 10: Metrics and evaluation driver

**Files:**
- Create: `ccrp/metrics.py`
- Create: `ccrp/eval.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `metrics.reduction_pct(baseline: float, treated: float) -> float`
  - `@dataclass eval.Point(method: str, server_cost: float, local_cost: float, quality_cost: float)`
  - `eval.run_experiment(requests, params, service_time_s, max_slack_s, *, compression_ratio, compression_quality_cost, compression_local, semantic_hit_rate, semantic_quality_cost, semantic_local) -> list[Point]` — returns points for `naive`, `cache_shaping`, `compression`, `semantic_cache`. `naive` = `simulate_order` on arrival order. `cache_shaping` = `simulate_order` on `greedy_schedule` output, quality_cost 0.0, local 0.0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import math
from ccrp.metrics import reduction_pct


def test_reduction_pct():
    assert math.isclose(reduction_pct(1.0, 0.75), 25.0)


def test_reduction_pct_zero_baseline_is_zero():
    assert reduction_pct(0.0, 0.0) == 0.0
```

```python
# tests/test_eval.py
from ccrp.cost_model import CacheParams
from ccrp.workloads import agentic_tooluse
from ccrp.eval import run_experiment, Point


def test_run_experiment_returns_all_methods_and_shaping_beats_naive():
    params = CacheParams(300.0, 1024, 0.1, 1.25, 3.0, 15.0)
    reqs = agentic_tooluse(n_runs=4, steps_per_run=4, schema_tokens=3000,
                           step_suffix_tokens=150, output_tokens=80, seed=3)
    # service_time_s=100 makes each run's prefix re-access gap in the naive
    # interleaved order (~400s, every 4th request) exceed the 300s TTL, so naive
    # re-writes every step; grouping keeps each run warm. max_slack_s is large so
    # the scheduler may reorder freely. This is what makes shaping < naive hold.
    points = run_experiment(
        reqs, params, service_time_s=100.0, max_slack_s=100000.0,
        compression_ratio=0.4, compression_quality_cost=0.15, compression_local=0.01,
        semantic_hit_rate=0.3, semantic_quality_cost=0.1, semantic_local=0.002,
    )
    by = {p.method: p for p in points}
    assert set(by) == {"naive", "cache_shaping", "compression", "semantic_cache"}
    assert by["cache_shaping"].server_cost < by["naive"].server_cost
    assert by["cache_shaping"].quality_cost == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# ccrp/metrics.py
def reduction_pct(baseline, treated):
    if baseline == 0.0:
        return 0.0
    return (baseline - treated) / baseline * 100.0
```

```python
# ccrp/eval.py
from dataclasses import dataclass
from ccrp.simulate import simulate_order
from ccrp.scheduler import greedy_schedule
from ccrp.baselines import compression_cost, semantic_cache_cost


@dataclass
class Point:
    method: str
    server_cost: float
    local_cost: float
    quality_cost: float


def run_experiment(requests, params, service_time_s, max_slack_s, *,
                   compression_ratio, compression_quality_cost, compression_local,
                   semantic_hit_rate, semantic_quality_cost, semantic_local):
    naive_order = sorted(requests, key=lambda r: r.arrival_s)
    naive_cost = simulate_order(naive_order, params, service_time_s)

    shaped = greedy_schedule(requests, params, service_time_s, max_slack_s)
    shaped_cost = simulate_order(shaped, params, service_time_s)

    comp_server, comp_local = compression_cost(
        requests, params, compression_ratio, compression_local)
    sem_server, sem_local = semantic_cache_cost(
        requests, params, semantic_hit_rate, semantic_local)

    return [
        Point("naive", naive_cost, 0.0, 0.0),
        Point("cache_shaping", shaped_cost, 0.0, 0.0),
        Point("compression", comp_server, comp_local, compression_quality_cost),
        Point("semantic_cache", sem_server, sem_local, semantic_quality_cost),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py tests/test_eval.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add ccrp/metrics.py ccrp/eval.py tests/test_metrics.py tests/test_eval.py
git commit -m "feat: metrics and experiment driver across all methods"
```

---

### Task 11: Characterization harness (telemetry parsing)

**Files:**
- Create: `ccrp/characterize.py`
- Test: `tests/test_characterize.py`
- Create: `tests/fixtures/anthropic_usage.json`
- Create: `tests/fixtures/openai_usage.json`

**Interfaces:**
- Consumes: `CacheParams` from `ccrp.cost_model`.
- Produces:
  - `parse_anthropic_usage(usage: dict) -> dict` with keys `input_tokens`, `cache_read_tokens`, `cache_write_tokens`.
  - `parse_openai_usage(usage: dict) -> dict` with keys `input_tokens`, `cache_read_tokens`.
  - `infer_min_prefix_tokens(probe_results: list[tuple[int, bool]]) -> int` — given `(prefix_tokens, observed_hit_on_repeat)` pairs, return the smallest prefix length that ever produced a hit.
  - `probe_provider(send_fn, prefix_tokens_list) -> list[tuple[int, bool]]` — calls a provided `send_fn(prefix_tokens) -> usage_dict` twice per length and reports whether the second call showed a cache read. `send_fn` is injected, so no real network in tests.

The real-API wiring (constructing an `anthropic` / `openai` client and a `send_fn`) lives in a `__main__` block and is NOT imported by tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_characterize.py
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
```

- [ ] **Step 2: Create fixtures and run test to verify it fails**

```json
// tests/fixtures/anthropic_usage.json
{"input_tokens": 50, "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 0, "output_tokens": 30}
```

```json
// tests/fixtures/openai_usage.json
{"prompt_tokens": 2200, "prompt_tokens_details": {"cached_tokens": 2048}, "completion_tokens": 40}
```

Run: `pytest tests/test_characterize.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# ccrp/characterize.py
def parse_anthropic_usage(usage):
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_write_tokens": usage.get("cache_creation_input_tokens", 0),
    }


def parse_openai_usage(usage):
    details = usage.get("prompt_tokens_details", {})
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "cache_read_tokens": details.get("cached_tokens", 0),
    }


def infer_min_prefix_tokens(probe_results):
    hits = [n for n, hit in probe_results if hit]
    return min(hits) if hits else 0


def probe_provider(send_fn, prefix_tokens_list):
    results = []
    for n in prefix_tokens_list:
        send_fn(n)                      # warm
        usage = send_fn(n)              # repeat
        parsed = usage.get("cache_read_tokens", 0)
        results.append((n, parsed > 0))
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_characterize.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add ccrp/characterize.py tests/test_characterize.py tests/fixtures/
git commit -m "feat: provider cache characterization from usage telemetry"
```

---

### Task 12: Figures

**Files:**
- Create: `ccrp/figures.py`
- Test: `tests/test_figures.py`

**Interfaces:**
- Consumes: `Point` from `ccrp.eval`.
- Produces: `pareto_plane(points: list[Point], out_path: str) -> str` — writes a PNG plotting server_cost (x) vs quality_cost (y), marker size by local_cost, labelled by method; returns `out_path`. Imports matplotlib lazily inside the function so core stays import-light.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figures.py
import pathlib
import pytest
from ccrp.eval import Point

mpl = pytest.importorskip("matplotlib")
from ccrp.figures import pareto_plane


def test_pareto_plane_writes_png(tmp_path):
    points = [
        Point("naive", 1.0, 0.0, 0.0),
        Point("cache_shaping", 0.6, 0.0, 0.0),
        Point("compression", 0.5, 0.1, 0.15),
        Point("semantic_cache", 0.7, 0.2, 0.1),
    ]
    out = tmp_path / "pareto.png"
    result = pareto_plane(points, str(out))
    assert result == str(out)
    assert pathlib.Path(result).exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pip install -e ".[figures]" && pytest tests/test_figures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ccrp.figures'`

- [ ] **Step 3: Write minimal implementation**

```python
# ccrp/figures.py
def pareto_plane(points, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for p in points:
        size = 50 + p.local_cost * 5000
        ax.scatter(p.server_cost, p.quality_cost, s=size)
        ax.annotate(p.method, (p.server_cost, p.quality_cost))
    ax.set_xlabel("server cost (USD)")
    ax.set_ylabel("quality cost")
    ax.set_title("Cost vs quality (marker size = local compute)")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_figures.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add ccrp/figures.py tests/test_figures.py
git commit -m "feat: pareto-plane figure for the comparison plane"
```

---

## Self-Review

**Spec coverage:**
- Cost model (spec §Cost model) -> Task 2. Break-even -> `intervention_net_savings` (Task 2) + `Point` local/quality split (Task 10).
- Negative result / "ruled out and why" -> documented in spec; no code needed (it is prose). No task required.
- Characterization / telemetry (spec §Contributions.2, §Evaluation) -> Task 11.
- Canonicalization (spec §Centerpiece step 1) -> Task 3.
- Cache simulator (spec §Evaluation methodology) -> Task 4.
- Clustering (spec §Centerpiece step 2) -> Task 5.
- Greedy scheduler + temporal/TTL + deadline slack (spec §Centerpiece step 3) -> Tasks 6 (pricing) + 7 (scheduler).
- Baselines compression + semantic caching (spec §Baselines) -> Task 9.
- Workloads agentic + multi-turn (spec §Evaluation workloads) -> Task 8.
- Metrics + comparison plane (spec §Evaluation metrics, headline figure) -> Tasks 10 + 12.
- Provider pick Anthropic + OpenAI (spec §Evaluation) -> Task 11 fixtures cover both telemetry shapes.
- Gemini deferred, privacy out of scope, optimal scheduler out of scope -> respected (no tasks).

**Placeholder scan:** No TBD/TODO. Every code step has complete code. No "add error handling" style steps.

**Type consistency:** `Request(id, prefix_key, prefix_tokens, suffix_tokens, output_tokens, arrival_s)` used identically in Tasks 2/5/6/7/8/9. `CacheParams` 6-field signature identical in Tasks 2/4/6/7/9/10. `billed_cost(req, cached_prefix_tokens, wrote_prefix, params)` identical in Tasks 2/6/9. `AccessResult(hit, cached_prefix_tokens, wrote_prefix)` consistent Tasks 4/6. `Point(method, server_cost, local_cost, quality_cost)` consistent Tasks 10/12. `greedy_schedule(...)` and `simulate_order(...)` signatures consistent Tasks 6/7/10.

**Out-of-plan (paper prose):** §Thesis, §Taxonomy "ruled out" table, §Positioning vs FrugalGPT/RadixAttention, §Open risks narrative are written in the paper, not this artifact. Tracked separately.
