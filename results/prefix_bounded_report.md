# B-bounded — budgeted (folding) head: does the prefix-cache advantage survive a finite KV budget?

**Why.** The published cost leg (`prefix_cost_report.md`) uses an *unbounded* append-only STATE
head for the M-A arm: one stable `key=val` digest line per prior turn, never rewritten. That
maximizes prefix hits but its head grows linearly with session length, so a long-lived agent
eventually hits the KV/context budget. The question is: "if you never evict, the head grows
without bound — what then?" This probe measures the honest answer: fold within the head under a
cap and re-measure the cost. It pairs with the closed-form head-growth bound in the paper
(Eq. `eq:bounded`): $H_t=H_0+t\Delta$ hits budget $C$ at $t^*=(C-H_0)/\Delta$; a capped head of
$H_{max}$ lines that folds the oldest $H_{max}-F$ into one summary on overflow gives a sawtooth
whose amortized recompute is $\Theta(V)+\Theta(H_{max}\Delta)/m$ ($m$ = turns between folds),
constant in $t$.

**Design.** 12 sessions × up to turn 40, WINDOW 8, head cap `H_MAX=10` digest lines, keep `F=4`
after a fold (so period $m=7$). Three arms at the SAME turns/records/newest-value recall slot,
all answerable (matched quality):
- `length_only` — naive recency window, front slides every turn (recompute every turn).
- `prefix_stable` — the paper's UNBOUNDED append-only head (idealized ceiling).
- `bounded_stable` — head capped at H_MAX, folds oldest→summary on overflow; a fold turn
  rewrites the head's leading bytes → prefix-cache invalidation → recompute spike (sawtooth).

Cost metric identical to the published leg: recompute = `prompt_eval_count −
prompt_eval_cached_count`. Sequential single-request replay per arm; per-run + per-arm nonces.

---

## Result (qwen3:14b, temp 0, 396 paired turns / 12 sessions, 48 fold events)

| arm | acc | mean prompt tok | **mean recompute** | prefill ms |
|---|---|---|---|---|
| length_only | 0.997 | 1589 | **1565** | 1058.7 |
| prefix_stable (unbounded) | 1.000 | 498 | **248** | 200.0 |
| bounded_stable (folds) | 1.000 | 454 | **272** | 212.1 |

**Sawtooth, measured directly (the two terms of Eq. eq:bounded):**
- append turns (n=348): mean recompute **247** — the $\Theta(V)$ floor, matches the unbounded
  arm's 248 (head bytes unchanged, cache stays warm).
- fold turns (n=48): mean recompute **456** — the $\Theta(H_{max}\Delta)$ spike when the head's
  leading bytes are rewritten. Folds are 12.1% of turns, so the amortized mean 272 ≈
  247 + 0.121×(456−247) — the bootstrapped theory, reproduced on real prefill counts.

**Head size stays flat under the budget** (the whole point): bounded_stable mean prompt **454**
tokens, while the unbounded head is already **498** at these 40 turns and *keeps growing* with t
(it would hit num_ctx=8192 at $t^*\approx$(8192−H₀)/Δ ≈ a few hundred turns). The bounded arm
buys a **flat, budget-safe** head for a small surcharge.

**Session-cluster bootstrap of mean paired recompute (tokens):**

| contrast | D | 95% CI | P(≤0) | verdict |
|---|---|---|---|---|
| length_only − bounded_stable | **1292.6** | [1271.1, 1315.5] | 0 | SIGNIFICANT |
| bounded_stable − prefix_stable (unbounded) | +23.8 | [23.3, 24.4] | 0 | fold surcharge, small + |
| length_only − prefix_stable (unbounded) | 1316.4 | [1294.8, 1339.6] | 0 | SIGNIFICANT |

**Recompute reduction under budget: bounded vs naive = 5.8×** (unbounded vs naive = 6.3×). The
prefix-invalidation advantage is essentially **budget-invariant**: forcing the head to fold costs
≈ +24 recompute tokens/turn (6.3×→5.8×) and still leaves naive truncation ~6× more expensive.

---

## What this establishes (and its ceiling)

**Established.** The no-eviction limitation (T1) is no longer a bare assumption. A finite head
budget that folds old content is measured to (a) keep the head size flat (454 tok, not growing to
the KV budget), and (b) preserve ~93% of the recompute reduction (5.8× vs 6.3×), paying only a
small, predictable fold surcharge that matches the closed-form amortization. The cost conclusion
does not depend on an unbounded head.

**Honest residuals.** The fold surcharge is real and slightly unfavorable to a budgeted M-A on
mean recompute (+24 tok/turn); we report it, not hide it. Folds here are a compact digest-line
rollup, not a learned summary; a real operator's fold content differs but the *mechanism* (head
front rewritten → one bounded recompute spike) is what we isolate. num_ctx fixed 8192; still
llama.cpp single-stream radix cache, no batching / paged engine (that axis is the separate
concurrency probe and the B-full ceiling).

Scripts: `gen_prefix_bounded.py`, `run_prefix_bounded.py`, `analyze_prefix_bounded.py`;
data `prefix_bounded_trials.jsonl`, `prefix_bounded_scored_qwen3_14b_t0.0.jsonl`.
