# B-lite — prefix-invalidation → recompute/prefill cost axis (M-A mechanism evidence)

**Purpose.** The quality experiments (A2/B*/H1) established M-A's *answer-fidelity* case but
left the third leg of the project's stated triple (信息保真 × 上下文膨胀 × **KV Cache 前缀保护**)
with **zero measured cost evidence**. B-lite is the directional mechanism probe for that leg:
does M-A's prefix-stable maintenance actually reduce serving recompute vs naive length-based
truncation, at matched quality?

**This is mechanism-level evidence, not an end-to-end throughput headline.** See §Ceiling.

---

## Design (three arms, matched quality, one isolated variable = prefix stability)

Multi-turn incremental-history sessions (`gen_prefix_cost.py`), 24 sessions × 30 turns, real
prose record bodies drawn from LongBench hotpotqa; the recall slot is a controlled `OBSk=val`
binding so correctness is machine-checkable. At every measured turn the agent must recall the
value carried by the **newest** observation — which is present verbatim in all three arms, so
**all three answer correctly (matched quality)**. The arms differ only in how older history is
placed relative to the KV prefix cache:

| arm | policy | older history | front bytes |
|---|---|---|---|
| `length_only` | naive recency window | last W raw records | **slide every turn** |
| `prefix_stable` (M-A) | append-only compact STATE head + volatile newest raw | one stable `key=val` digest line/turn, append-only | **stable prefix** |
| `pinned_raw` (CONTROL) | same raw window **size** as length_only, but oldest records pinned (never slide) | fixed first W-1 raw records + newest | **stable prefix** |

`pinned_raw` exists to separate the two ways M-A could look cheap: (a) it *sends fewer tokens*
(compaction) vs (b) its window *doesn't invalidate the prefix* (stability). pinned_raw holds
size constant and varies only stability.

**Cost metric = recompute tokens = `prompt_eval_count − prompt_eval_cached_count`** — the tokens
the GPU must actually prefill; cached tokens are free. Corroborated by `prompt_eval_duration`
(prefill ms). Sequential single-request replay; each arm is its own ascending-turn cache stream
so the prefix warms exactly as in a real conversation. Per-run + per-arm nonces defeat stale
whole-prompt radix nodes from prior invocations (a cross-run contamination bug caught and fixed
during the smoke phase — without it the sliding window faked a cache hit).

---

## Result (qwen3:14b, temp 0, 552 paired turns / 24 sessions)

| arm | acc | mean prompt tok | mean cached | **mean recompute** | cache hit | prefill ms |
|---|---|---|---|---|---|---|
| length_only | 1.000 | 1575 | 18 | **1557** | 1.1% | 1056 |
| prefix_stable (M-A) | 1.000 | 445 | 197 | **249** | 44.2% | 205 |
| pinned_raw (control) | 1.000 | 1584 | 1299 | **285** | 82.0% | 230 |

Matched-quality assertion: accuracy 1.000 in all three arms. Matched-size control: length_only
1575 tok vs pinned_raw 1584 tok (≈equal); prefix_stable 445 (compact).

**Session-cluster bootstrap of the mean paired recompute difference (tokens):**

| contrast | D | 95% CI | P(≤0) | verdict |
|---|---|---|---|---|
| length_only − M-A | **1308.6** | [1280.1, 1331.0] | 0 | SIGNIFICANT |
| length_only − pinned (pure SLIDE, matched size) | **1272.1** | [1243.9, 1294.2] | 0 | SIGNIFICANT |
| M-A − pinned (compaction beyond stability) | −36.5 | [−37.8, −35.2] | 1 | M-A recomputes slightly MORE |

**Prefill latency (ms):** LO − M-A = 850.8 [826.9, 868.2]; LO − pinned = 825.8 [801.1, 843.4];
both SIGNIFICANT. **Recompute reduction:** M-A vs naive = **6.3×**; same-size stable pin vs naive
= **5.5×**.

**Cross-model confirmation (qwen2.5:14b, temp 0, same 552 pairs):** recompute 1549→241 (M-A) /
277 (pinned); LO − M-A = **1308.7** [1280.2, 1331.1], LO − pinned = 1272.2, prefill 856.4 ms
[831.9, 874.2]; reduction **6.4×**. Matched-quality holds (acc length_only 0.987, prefix_stable
1.000, pinned_raw 0.998, all ≥0.95). The paired-recompute D matches qwen3:14b to within 0.1
token and the pinned control reproduces the gap again — the cost mechanism is a serving-layer
property keyed on token prefixes, **not a model-specific artifact**.

---

## What this does and does not establish

**Established (within scope):** At matched answer-fidelity, M-A's prefix-stable maintenance
recomputes ~6× fewer tokens and prefills ~5× faster than naive recency-window truncation. The
gain is **attributable to prefix invalidation, not to sending less**: the `pinned_raw` control
carries the *same* total tokens as length_only yet reproduces **97% of the gap** (1272/1309) at
matched size. The prefix-invalidation → recompute → prefill-latency axis is real and large for
these two policies.

**Honest residual (not hidden):** M-A recomputes ~37 tok/turn *more* than the frozen-pin control,
because its append-only head still writes a new digest line each turn while pinned_raw's set is
frozen. This is tiny and slightly unfavorable to M-A on recompute — but M-A buys **bounded memory**
(445 vs 1584 tokens carried) that pinned_raw does not. Different axes; reported both ways.

**Ceiling / limitations:**
1. **llama.cpp in-process radix cache, single-request sequential replay** — no concurrency,
   batching, or paged-attention scheduler. A production vLLM/LMCache TTFT/throughput number is
   **B-full** (requires the vLLM install we deliberately excluded). This is *directional mechanism*
   evidence that the axis exists and is large, not an end-to-end serving benchmark.
2. **Controlled synthetic sessions** with a trivially-verifiable recall query, chosen so all arms
   are correct (to isolate cost at matched quality). Not a realistic multi-step agent task; the
   distractor records are not adversarial. It answers "does stability cut recompute," not "does
   M-A win a real workload end-to-end."
3. **num_ctx fixed 8192; 30-turn sessions.** M-A's append-only head grows with session length; a
   long-running agent must fold/evict *within* the head (what the walking-skeleton budget governs).
   Bounded here; not exercised past 30 turns.
4. **Model coverage:** qwen3:14b + qwen2.5:14b, both reproduce the effect to within 0.1 recompute
   token. Prefix caching is a serving-layer property keyed on token prefixes, so the mechanism is
   largely model-agnostic; a 2nd *family* (llama3/mistral) is cheap optional reinforcement.
