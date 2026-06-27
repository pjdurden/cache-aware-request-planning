# Cache-Aware Client-Side Request Planning for Black-Box LLM APIs

**Date:** 2026-06-27
**Status:** Design approved, pending implementation plan
**Type:** Research paper (systems-for-ML), with open-source artifact

## Problem

Application developers consume large language models through black-box, per-token-billed APIs (Anthropic, OpenAI, Gemini). They do not own the model weights, cannot change server kernels, and receive only text (plus limited telemetry). A pervasive folk belief is that token cost can be reduced by relocating work to the client (for example tokenizing on-device and shipping token IDs, or prompting in "denser" languages). Most of these moves save nothing.

The governing invariant: **you pay for the tokens the server processes. Relocating computation to the client changes the bill only if it reduces billable tokens or the number of calls, at fixed task quality.** The space of interventions that actually satisfy this is small, under-formalized, and distinct from the large server-side / model-owner literature (split inference, speculative decoding, RadixAttention, Mooncake, vLLM prefix caching), all of which assume control of the GPU.

## Thesis

For a consumer of a black-box per-token API, **cache-aware request shaping is the only intervention that is simultaneously lossless (zero task-quality cost) and local-compute-free.** It exploits the provider's own prompt cache, which the consumer cannot build but can align to. The paper formalizes when this dominates and when the consumer must fall back to lossy or compute-costly alternatives.

## Contributions

1. **Model.** A formal cost model for the API consumer that expresses billed cost as a function of (full vs cache-discounted input tokens, output tokens, call count) and treats each client-side intervention as a transform on that function with an associated local-compute price. Yields a per-intervention break-even condition. Includes the negative result: an explicit account of which moves are illegal or null in this setting (client tokenization, speculative decoding, early-layer / split execution, server-side constrained decoding).
2. **Measurement.** An empirical characterization of real provider prompt caches (TTL, minimum cacheable prefix length, prefix granularity, discount rate, eviction behavior) recovered from usage telemetry (`cache_read_input_tokens` / `cache_creation_input_tokens` for Anthropic, `cached_tokens` for OpenAI).
3. **Mechanism.** A greedy prefix-clustering scheduler: client-side middleware that canonicalizes, clusters, and time-orders a request stream to maximize provider-cache hit rate under TTL constraints. Lossless and near-zero local compute. The claim is that a deliberately simple greedy scheduler captures most of the available win.

## Scope

**Primary optimization target:** dollar cost (billable input + output tokens, call count) at fixed task quality.
**Secondary axis (measured):** the local-compute price of each client-side intervention, and the break-even between local cost and server-token savings.
**Out of scope (cited as motivation, not evaluated):** privacy / data-egress threat model; energy modeling beyond a coarse local-compute proxy.

## Design

### Cost model (spine)

Per-request billed cost as a function of: full-price input tokens, cache-discounted input tokens, output tokens, and number of calls. Each intervention is a transform on this function plus a local-compute term (latency / quality / device cost). The break-even condition tells you when an intervention is net-positive.

Cache-shaping is the unique intervention with a zero quality term and a near-zero local-compute term, so it dominates by construction up to the **cacheable fraction** of the prompt. Quantifying that fraction across workloads is the empirical core.

### Taxonomy and positioning

Full enumeration of consumer-available moves, with FrugalGPT (model cascade) slotted as one cell, not the whole space. An explicit "ruled out and why" table fences off the server-side literature:

| Move | Status for black-box consumer | Reason |
|------|-------------------------------|--------|
| Client-side tokenization, ship token IDs | Null | Billing and forward-pass compute are invariant to where tokenization runs |
| Speculative decoding / draft on client | Illegal | No server verification API; no weights |
| Early-layer / split execution | Illegal | No model weights |
| Output constrained decoding | Server-side | Decoding happens on the provider |
| Context compression (local SLM) | Legal, lossy, local cost | Baseline |
| Semantic caching (local) | Legal, skip-call on hit | Baseline |
| Cache-aware request shaping | Legal, lossless, free | **Centerpiece** |

### Centerpiece: greedy prefix-clustering scheduler

Client-side middleware between the application and the provider SDK. Maintains a model of the provider cache (prefix to estimated warm-until time). For an incoming request stream:

1. **Canonicalize** prefixes: sort JSON keys, normalize whitespace, stabilize tool-schema serialization so byte-identical prefixes survive across requests.
2. **Cluster** requests by longest shared prefix.
3. **Greedily order / batch** within a cluster to land inside the TTL window, deferring or promoting requests where latency slack allows.

Static layout discipline (invariant content at the front, variable content at the tail, deliberate breakpoint placement for explicit-cache providers) is the precondition the scheduler assumes. The scheduler is intentionally greedy and simple; the contribution is that a trivial scheduler captures most of the win, not an optimal solver.

### Baselines (comparison plane)

Plotted on three axes (dollars saved, quality cost, local compute):

- **Context compression**, LLMLingua-style local small model. Lossy, real local cost.
- **Semantic caching**, local embedding plus response reuse. Skips calls on a hit, requires a hit.
- **Cache-shaping (ours)**, lossless and free, bounded by cacheable fraction.

Headline figure: a Pareto plane showing ours dominates in the high-cacheable-prefix regime (agentic tool-use) and where it stops dominating (low prefix reuse, forcing a fall back to compression or caching).

### Evaluation

- **Workloads:**
  - Primary: agentic tool-use loops (large shared system prompt plus tool schemas reused across every step of every run).
  - Secondary: multi-turn chat sessions (shared system prompt plus growing history; user think-time vs short cache TTL exercises the temporal scheduling).
- **Methodology (hybrid):**
  1. Small real-API probe on **Anthropic** (explicit `cache_control`) and **OpenAI** (automatic caching) to calibrate cache parameters from usage telemetry. Chosen for the explicit-vs-implicit contrast. Gemini deferred to future work.
  2. A **calibrated simulator** for the scheduler sweep (thousands of orderings / configs), avoiding a four-figure live-API bill. Calibration shown for credibility.
- **Metrics:** billed-token reduction %, cache hit rate, dollars saved; measured against naive ordering and against the two baselines.

## Artifact

- A small Python client-side middleware (drop-in wrapper around the provider SDKs) implementing canonicalization, clustering, and the greedy scheduler.
- The characterization harness (probes real providers, recovers cache parameters from telemetry).
- The calibrated simulator.

Open-sourced. Doubles as an AI-infra credibility piece.

## Venue

arXiv first; target a systems-for-ML workshop (MLSys or NeurIPS ML-for-Systems). Workshop-scoped, not a full-conference push, to fit the time budget.

## Non-goals / explicit cuts

- No privacy / leakage evaluation (cited as motivation only).
- No Gemini in v1 (future work).
- No optimal scheduler; greedy only.
- Baselines (compression, caching) are implemented only to the depth needed for the comparison plane, not as standalone contributions.
- Taxonomy moves 2/4/5/7 (history summarization, client-side RAG, local cascade, output shaping) are discussed and left as "composes with ours / future work," not built.

## Open risks

- **Provider cache opacity / drift.** Providers may change cache behavior between characterization and publication. Mitigation: report characterization date and method so results are reproducible against the telemetry, not against undocumented internals.
- **Novelty challenge.** Reviewers may equate this with server-side prefix caching (RadixAttention). Mitigation: the "ruled out and why" table and the black-box-consumer framing make the distinction explicit and central.
- **Workload realism.** The scheduler shows a flat line on streams without shared-prefix structure. Mitigation: workloads chosen specifically for prefix structure; the negative regime is reported, not hidden.
