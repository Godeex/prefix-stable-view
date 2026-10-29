# Option B — 2×2 factorial: DISTRACTION × PREFIXDEPTH at matched budget (I-03 kill test)

**Question (I-03 kill criterion).** Does a *joint* admission/eviction mechanism — semantic-distraction
admission (drop competing near-twin values) **combined with** prefix-depth-aware stable-head layout —
beat the best *decoupled* single mechanism at a **matched token budget**? I-03 (the joint
"utility − prefix-depth" scorer) requires the joint arm to be meaningfully better than *either* lever
alone; if one lever already suffices, the joint scorer buys nothing.

**Design (pre-registered, then gated before any model call).** 2×2 on the M-A maintenance policy:
- **d = DISTRACTION**: 1 → fold a distractor's competing `TARGET=` value out (`[dropped]`); 0 → carry it.
- **p = PREFIXDEPTH**: 1 → append-only *stable head* (older records' full bodies never rewritten) +
  volatile recent tail; 0 → single recent-first sliding window (front shifts every turn).
- Configs `c00` naive / `c10` semantic-only / `c01` prefix-only / `c11` joint.
- 24 sessions × 20 records; each carries the gold `TARGET` value in **every** config and turn
  (fidelity floor), so any substitution is about *carrying a competing value*, never gold loss.
- Distractors = rank-selected most-confusable sentences (bge-m3, gold-adjacent); competing-vs-gold cos
  median 0.63 (max 0.79). Bodies are real LongBench passages.
- Outcomes: **contamination** = substitution rate (emitted a WRONG value present only in distractors);
  **cost** = recompute tokens = `prompt_eval_count − prompt_eval_cached_count` (ollama radix prefix
  cache; validated end-to-end in B-lite). Each config is its own ascending-turn cache stream;
  per-config nonce defeats stale/cross-arm cache contamination; temp 0, seed fixed.

**Deterministic layout gate (`check_joint_layout.py`, exit 0 before running).** Confirms the axes are
actually manipulated and matched: (A) competing values 1.06 (d=0) vs 0.00 (d=1); (B) leading-byte stable
prefix fraction 0.018 (slide) vs 0.640 (mature stable head); (C) prompt size matched across p (≤1.4%).

**Model:** qwen3:14b, 24 sessions, 408 turn-cells/config (1632 calls). Primary endpoint unchanged.

## Results (session-cluster bootstrap, 4000 resamples, 95% CI)

| cfg | (d,p) | substitution | recompute tok | correct (construct-limited) |
|----|----|----|----|----|
| c00 | naive (0,0)     | **0.392** | 538.1 | 0.118 |
| c10 | semantic (1,0)  | 0.000     | 537.5 | 0.728 |
| c01 | prefix (0,1)    | 0.002     | 275.4 | 0.696 |
| c11 | joint (1,1)     | 0.000     | 275.2 | 0.512 |

**Main effects (both large, significant):**
- d → substitution: `c10−c00` Δ = **−0.392** CI[−0.495,−0.294] SIG.
- p → recompute:   `c01−c00` Δ = **−262.8** CI[−276,−249] SIG (~1.95× reduction).

**Interaction (tested, not forced):**
- substitution: `+0.390` CI[0.299,0.488] — **sub-additive**: the distractor-dropping benefit is large
  under a sliding head but ~0 under a stable head (the stable head already prevents substitution).
  The two levers **remove each other's marginal value** — redundancy, not synergy.
- recompute: `+0.47` tokens — practically zero.

**Joint vs each single (Pareto with SESOI: substitution ≥0.03, recompute ≥20 tok):**
- c11 vs c10: dominates (cost −262 ≫ SESOI; quality no-harm). Expected — prefix lever adds the cost win.
- c11 vs **c01**: **TIE.** substitution 0.000 vs 0.002 (Δ−0.0025, within SESOI); recompute 275.2 vs
  275.4 (Δ−0.22 tok, 0.08%, within SESOI). Joint does **not** dominate prefix-only by any meaningful margin.

## Verdict

**I-03 (joint semantic × prefix-depth admission/eviction beats decoupled) — KILLED.**
At matched budget, **prefix-stability alone (`c01`) already captures both benefits**: it halves recompute
(the cost lever) *and* drives contamination to ~0 (because the append-only head keeps the authoritative
gold value prominent so the model reads it instead of substituting a distractor). Adding the semantic-
distraction admission lever (`c11`) buys nothing — the levers are **substitutes, not complements**, and the
interaction is sub-additive, the opposite of the non-additive/synergy premise I-03 needs.

## Honest caveats / limits

1. **Fidelity `correct` is construct-limited here**: the gold is a synthetic code (`R7`) competing with
   meaningful passage bodies, so the model often answers with salient body text rather than the code.
   The metric under-counts correct recall uniformly; it is **not** used to gate the verdict. It also
   cannot rescue I-03: joint `c11`=0.512 is not better than prefix-only `c01`=0.696.
2. This 2×2 tests **"both mechanisms > one mechanism"**; it cannot separate a *unified* joint score from
   *independently-applied* levers. Under either reading there is no joint benefit, so I-03's motivation
   fails regardless.
3. Single model family (qwen3:14b); single budget (tail V=5, ~538-token prompts). Contamination-
   suppression by layout could weaken for distractors that are *not* separable by position (e.g. a
   competing value appended in the volatile tail at the newest turn). Not tested.
4. Cost is a llama.cpp single-stream radix-cache proxy (as in B-lite), not a vLLM/SGLM batched prefix cache.

---

# Rework (authorized): tail-distractor stress — the regime built to break `c01`

Caveat #3 above named the one place prefix-only could fail: a competing value that **no layout can move
away from the query**. This rework constructs exactly that and re-runs the full 2×2.

**Design change (`TAILMODE=1`).** Gold stays deep (idx 2 → inside the stable head for `p=1`). **Both**
competing near-twin distractors are placed **inside the volatile tail band** (idx 15 and 17, tail = last
`V=5`). We probe only the 5 mature tail turns (15–19) where a tail distractor is *always* in the recent
window. The layout axis can no longer separate gold from distractor by position — the distractor sits at
the front, nearest the query. If the semantic value-folding lever (`d`) ever earns its keep, it is here.

**Deterministic gate re-passed on tail sessions (`check_joint_layout.py`, exit 0).** (A) competing values
**1.60 (d=0) vs 0.00 (d=1)** — the hazard is *live* in `c00`/`c01` even with a stable head, because the
distractors are in the tail. (B) prefix fraction 0.011 (slide) vs 0.653 (stable head). (C) size matched
≤1.1%. So both axes still separate and the substitution hazard is genuinely present in the prefix-only arm.

**Model:** qwen3:14b, 24 sessions × 5 probe turns × 4 configs = **480 calls, 0 errors.**

## Results (session-cluster bootstrap, 4000 resamples, 95% CI)

| cfg | (d,p) | substitution | recompute tok | correct (construct-limited) |
|----|----|----|----|----|
| c00 | naive (0,0)     | **0.808** | 790.7 | 0.058 |
| c10 | semantic (1,0)  | 0.000     | 789.6 | 0.683 |
| c01 | prefix (0,1)    | **0.000** | 377.8 | 0.842 |
| c11 | joint (1,1)     | 0.000     | 376.8 | 0.750 |

**Main effects (both large, significant):**
- d → substitution: `c10−c00` Δ = **−0.808** CI[−0.883,−0.725] SIG — naive recent-first *does* collapse
  (0.808) when a near-twin sits in the tail, so the hazard is real.
- p → recompute:   `c01−c00` Δ = **−412.9** CI[−429,−396] SIG (~2.1× reduction).

**Interaction (tested):** substitution `+0.808` CI[0.733,0.883] — again **sub-additive**: `d`'s benefit is
large under the sliding view but **exactly 0 under the stable head**. recompute interaction +0.14 tok (noise).

**Joint vs each single (Pareto with SESOI: substitution ≥0.03, recompute ≥20 tok):**
- c11 vs c10 (semantic-only): dominates on cost (−413 tok), tie on quality — the prefix lever still carries the cost win.
- c11 vs **c01 (prefix-only)**: **TIE.** substitution 0.000 vs 0.000 (Δ = 0, CI[0,0]); recompute 376.8 vs
  377.8 (Δ −1.0 tok, 0.3%, far below SESOI 20). Even with the competing value parked in the volatile tail,
  **the semantic-admission lever adds nothing on top of prefix-stability.**

## Why `c01` holds here (the honest mechanism)

`c01` still renders the distractor values in-context (gate confirms competing = 1.60) yet substitution is
0. The stable-head framing — gold in the labeled `STATE (accumulated)` block, distractors in
`RECENT TOOL RESULTS` — gives the model a **structural authority cue**: it reads the accumulated gold
rather than the freshly-slid competing value. That is a property of the *layout* arm, and it already closes
the gap the semantic arm was meant to fill. The joint scorer's premise (two mechanisms are needed, and
together super-additive) does not survive this direct test.

## Verdict after rework

**I-03 — KILLED (rework confirms it).** The tail-distractor stress was the most favorable regime for
I-03's claim, engineered so position/layout could not rescue the naive arm — and `c01` (prefix-only) still
matches `c11` (joint) on contamination *and* cost within SESOI. The semantic-distraction admission lever
is a **substitute** for prefix-stability, not a complement. There is no joint > decoupled effect to build a
"semantic utility − prefix-depth" scorer around.

## Rework caveats

1. Same construct-limited `correct` caveat as above (synthetic code gold); it is not used to gate. Here
   `c01`=0.842 ≥ `c11`=0.750 — the joint arm is, if anything, *worse* on fidelity, again not rescuing I-03.
2. Still one model family (qwen3:14b), one budget (V=5). A cross-family confirmation is not run.
3. The stress places distractors in the tail; it does not model adversarial paraphrase of the *gold*
   value itself, nor multi-hop aggregation across turns — different failure modes, out of scope here.


## Implication for the project (user decision, not ours)

The prefix-cost mechanism and the naive-truncation fragility (c00 collapse: 39% substitution, 12%
correct) stand as **positive results supporting I-01's framing** (history as a prefix-stability-
constrained streaming view; the cost leg). But **I-03 as a distinct "joint > decoupled" contribution is
not supported by its own kill experiment.** The tail-distractor rework below was the authorized second
chance for I-03; it failed to revive the joint claim. What remains to decide (user's call): drop I-03 to a
secondary/ablation finding, or carry only I-01 forward.

---

# DECISION (2026-09-24, user): A — I-03 archived as negative result, carry I-01 only

User elected option **A**: treat "prefix-stability already absorbs the admission benefit in the fidelity
regime" as the conclusion, archive **I-03** as a negative/ablation result, and carry only **I-01**
forward. The bounded-eviction reframe (option B) was **knowingly not run**.

## Known limitations this hands to I-01 (must appear in the paper's scope/limitations)

1. **No-eviction head is a design choice, not a deployed budget.** Both `p=1` arms keep an
   append-only head that grows across the session and never drops a record, so they never face the
   admission/eviction trade-off a finite KV budget imposes. The prefix-cost win is real for the
   **cacheable-prefix** mechanism (B-lite-validated radix hit), but it is **not** evidence that a
   bounded M-A view matches an eviction policy — that comparison is out of scope and was not tested.
2. **The fidelity regime is where the substitution claim lives.** c01/c11 suppress contamination when
   the gold is *present* and the stable-head `STATE` vs `RECENT` labeling acts as an authority cue. If
   gold is at risk of eviction under budget pressure, a utility-based admission rule could matter — the
   exact question option B would have tested. **Do not claim admission control is subsumed in general**;
   claim only that, at this no-eviction operating point, the semantic-distraction lever is a substitute
   for prefix-stability, not a complement.
3. Single model family (qwen3:14b); llama.cpp single-stream radix proxy, not vLLM/SGLM batched cache;
   construct-limited `correct` (synthetic gold code) — carried unchanged from the base caveats.
   *(Hardened by the 2nd-family + forced-answer generality run below; see "Generality" section.)*

Net: I-01's contribution is **the prefix-constrained streaming-view *formulation* + the *cost* leg +
the *naive-collapse fragility* finding**, all positive and falsified-surviving. I-03's *joint>decoupled*
synergy is a documented negative result, not a claim.

---

# Generality (authorized extra run): 2nd model family + forced-commit answer protocol

Caveat #3 flagged two weaknesses: single family, and the `correct` metric was construct-limited because
models could answer with body text instead of committing to the gold code. Both are addressed here.

**Protocol tweak (`RJ_FORCE=1`).** One llama3 smoke call revealed a *answer-style confound*: qwen commits
to a single value but llama3 **listed every `TARGET=`** it saw, which the lenient substitution rule
(gold-present cancels the flag) scored as ~0 for all arms — uninformative, not a null mechanism. Fix:
append an explicit *"Answer with ONLY the single recorded value"* instruction so every model must commit.
To avoid a prompt-change confound, the tweak is applied to **both** families as a matched pair, and the
frozen non-forced qwen results above are left intact. Tail-distractor sessions; 480 calls each, 0 errors.

**Forced-commit also repairs `correct`:** with a single value demanded, the fidelity metric becomes
valid — it is 1.000 for every arm that keeps the gold in-context and collapses only for naive.

### Substitution / correct / recompute (mean), forced protocol, tail stress

| cfg | (d,p) | subst (qwen3) | subst (llama3) | correct (qwen3/llama3) | recompute (qwen3/llama3) |
|----|----|----|----|----|----|
| c00 naive | (0,0) | **0.617** | **0.550** | 0.383 / 0.450 | 810.7 / 810.7 |
| c10 semantic | (1,0) | 0.000 | 0.000 | 1.000 / 1.000 | 809.6 / 809.6 |
| c01 prefix | (0,1) | **0.000** | **0.000** | 1.000 / 1.000 | 397.8 / 397.8 |
| c11 joint | (1,1) | 0.000 | 0.000 | 1.000 / 1.000 | 396.8 / 396.8 |

**Every conclusion replicates across both families, with the same signs and significance:**
- Main effect d→substitution: `c10−c00` Δ = −0.617 (qwen) / −0.550 (llama3), both SIG.
- Main effect p→recompute: `c01−c00` Δ ≈ −412.9 tok, **identical across families** (it is a
  cache-mechanism effect, not a model-behavior effect) — the prefix-cost leg is family-invariant.
- Substitution interaction: **sub-additive** in both (qwen +0.617, llama +0.550) — each lever removes
  the other's marginal value; no synergy for I-03 to build on.
- **I-03 kill test — c11 vs c01: TIE in BOTH families.** substitution 0.000 vs 0.000; recompute −1.0 tok
  (0.3%, far below the 20-token SESOI). Joint does **not** dominate prefix-only by any meaningful margin.

**Robustness note (honest):** the naive-collapse magnitude shifts with protocol/answer style (non-forced
0.808 vs forced 0.55–0.62) — that is answer-format sensitivity in the *naive* arm, but the *decisive*
comparison (`c01` = `c11` = 0.000; prefix-only already at floor) is stable across both prompt styles and
both families. Verdict unchanged: **I-03 KILLED**, now on 2 model families + 2 answer protocols.

Updated caveat #3 → *single family* and *construct-limited `correct`* are retired: two families
(qwen3:14b, llama3:8b) and a forced-commit protocol reproduce all effects, and the `correct` metric is
valid under forced commitment. What remains: llama.cpp single-stream radix proxy (not vLLM/SGLM batched
cache), and the no-eviction head (limitation #1) still stand.
