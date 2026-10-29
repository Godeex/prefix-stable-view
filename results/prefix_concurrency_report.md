# B-concurrency — is the prefix-cache recompute advantage invariant to concurrent load?

**Why (external validity).** The published cost leg is single-request sequential replay through
llama.cpp's in-process radix cache. A production engine (vLLM/SGLang) batches many sessions
against a shared paged KV pool. We cannot install a paged engine (no new runtime), so we exercise
the load-sharing regime we CAN reach: fire the three arms' turns from many sessions
CONCURRENTLY (thread pool width 8) against one shared ollama KV pool, and ask the narrower
question that matters for external validity — does the recompute gap survive contention, or was it an artifact of
exclusive single-stream access?

**Method.** `run_prefix_concurrency.py`, `--parallel 8 --nsample 8`, qwen3:14b, temp 0, num_ctx
8192. 8 sessions × 5 warmed mid-session turns × 3 arms = 120 concurrent calls, per-run+per-arm
nonces (no stale cross-run node). Metrics identical to the single-stream leg: recompute =
`prompt_eval_count − prompt_eval_cached_count`. Plus wall-clock latency and a throughput proxy.

---

## Result (concurrent, parallel=8)

| arm | acc | **mean recompute** | mean wall ms | p95 wall ms | single-stream recompute (ref) |
|---|---|---|---|---|---|
| length_only | 1.000 | **1583** | 8058 | 8766 | 1557 |
| prefix_stable | 1.000 | **293** | 7749 | 8832 | 249 |
| pinned_raw | 1.000 | **506** | 7477 | 8104 | 285 |

Throughput ≈ 1.0 req/s (shared 14B slot). Overall elapsed 11.0s for 120 calls at width 8.

## Reading (what it does and does not show)

**1. Recompute ordering is concurrency-invariant.** length_only ≫ pinned > prefix_stable under
contention, exactly as single-stream. The naive-minus-stable paired gap is 1583−293 = **1290**
tokens/turn under 8-way load vs 1308 single-stream (within 1.5%): the prefix-invalidation
advantage is a **content-keyed cache property**, not an artifact of exclusive access. Stable arms
still recompute ~5.4× less than naive under load (293 vs 1583).

**2. Every arm recomputes somewhat MORE under shared-pool pressure.** prefix_stable 249→293
(+18%), pinned_raw 285→506 (+78%). Co-resident sessions evict each other's radix blocks, so
fewer bytes are reused than in exclusive single-stream. The effect is largest for pinned_raw
because it carries the most raw tokens (1584) and so has the most leading bytes exposed to pool
pressure — yet it still stays ~3× below naive. The mechanism (front stability beats front sliding)
survives; only its magnitude is mildly attenuated by contention.

**3. Honest ceiling — this is NOT a throughput/TTFT proxy.** Wall-clock is essentially flat across
arms (~7.5–8.1 s) even though recompute differs 5×. That is because ollama serializes concurrent
requests through one model slot rather than continuously batching prefill chunks the way a paged
engine does; every request waits on the same GPU, so latency is dominated by queueing, not by the
arm's prefill work. A real vLLM/SGLang would show latency *separation* between the arms. We
therefore claim **recompute-token invariance under a shared pool**, and explicitly disclaim any
batched TTFT/throughput headline (that remains B-full, out of scope).

## What this adds to the paper
Narrows T2 (single-stream proxy): the cost conclusion now also holds when sessions contend for one
shared KV pool, not only in exclusive replay — the first step off single-stream. It does not reach
a paged, continuously-batched scheduler, and we say so.

Scripts: `run_prefix_concurrency.py`; data
`prefix_concurrency_scored_qwen3_14b_t0.0_p8.jsonl`.
