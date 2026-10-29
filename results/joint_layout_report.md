# Layout-only control: does the fidelity gain come from prefix stability or from the block labels?

Why this control: the main 2×2 ablates *content* (distractor folding `d`) against *layout*
(`p` = sliding window vs append-only head), and the manuscript explained the stable arm's zero
substitution by pointing at the labelled `STATE (accumulated, append-only)` block as an authority
cue. That explanation is not implied by the 2×2: the stable arm changed the labels **and** the
stability at once. The fix is a second factorial that holds content fixed and crosses the two
layout factors.

## Design

`run_joint_layout.py`, one epoch, same sessions/turns/model/seed as the cost study
(`qwen3:14b`, `temperature=0`, `seed=1234`, `num_ctx=8192`, `num_predict=40`, per-cell nonce so a
cell cannot read another cell's radix nodes):

| cell | labels | prefix | construction |
|---|---|---|---|
| `s0l0` | no | sliding | recent-first window, generic `CONTEXT:` header (= naive arm) |
| `s0l1` | yes | sliding | same `STATE`/`RECENT` headers, but the older block is re-sorted newest→oldest so its leading bytes change every turn |
| `s1l0` | no | stable | older-first append-only order, neutral headers (`CONTEXT:` / `MORE:` / `Answer from the lines above.`) |
| `s1l1` | yes | stable | labelled append-only `STATE` + volatile `RECENT` (= the paper's stable arm) |

Content is identical in all four cells: `d=0`, gold verbatim, competing near-twin values verbatim,
only ordering and section headers differ. 12 sessions × 34 mature turns × 4 cells = 1632 calls, 0
server errors, 408 (session,turn) blocks with all four arms.

## Controls that had to hold first (`check_layout_controls.py`, no model calls)

* front-reuse (longest common character prefix between consecutive-turn prompts): `s0l0` 2.0 %,
  `s0l1` 7.1 %, `s1l0` 51.2 %, `s1l1` 51.3 % — the stability factor manipulates stability and the
  label factor does not touch it;
* competing values in context: 1.06 in every cell (matched by construction, measured per turn);
* prompt-token spread across the four cells 1.97 % (only the header wording differs in length,
  ~46 chars / ~12 tokens; the gate is a declared 5 % bound, not a silent tolerance).

## Results (408 paired blocks)

| cell | labels | stable | substitution (95 % Wilson) | correct | other | recompute tok | prompt tok |
|---|---|---|---|---|---|---|---|
| `s0l0` | 0 | 0 | 0.473 [0.425, 0.522] | 0.113 | 0.414 | 538.2 | 566.7 |
| `s0l1` | 1 | 0 | 0.387 [0.341, 0.435] | 0.191 | 0.422 | 535.8 | 574.8 |
| `s1l0` | 0 | 1 | 0.000 [0.000, 0.009] | 0.828 | 0.172 | 269.3 | 563.5 |
| `s1l1` | 1 | 1 | 0.000 [0.000, 0.009] | 0.608 | 0.392 | 275.3 | 574.8 |

Paired contrasts on the same (session, turn) cells; CIs are session-cluster bootstrap (6000
resamples), `b`/`c`/`p` is exact McNemar:

| contrast | Δ substitution [95 % CI] | McNemar | Δ recompute tok [95 % CI] |
|---|---|---|---|
| label effect, stable layout (`s1l1`−`s1l0`) | +0.000 [0.000, 0.000] | b=0 c=0 p=1 | +6.1 [6.1, 6.1] |
| label effect, sliding layout (`s0l1`−`s0l0`) | −0.086 [−0.181, 0.000] | b=59 c=94 p=0.0058 | −2.5 [−3.2, −1.7] |
| stability effect, labels on (`s1l1`−`s0l1`) | −0.387 [−0.476, −0.297] | b=0 c=158 p=5.5e−48 | −260.4 [−273.0, −247.7] |
| stability effect, labels off (`s1l0`−`s0l0`) | −0.473 [−0.578, −0.370] | b=0 c=193 p=1.6e−58 | −268.9 [−282.2, −255.6] |

Factorial model, cluster-robust linear probability (the logit is not estimable: both stable cells
are 0/408 events, i.e. complete separation, and the IRLS Hessian is singular):

```
substitution ~ const + labels + stable + labels:stable     (SEs clustered on session, 12 clusters)
  const              +0.473  se 0.055  p=1.3e-08
  labels             -0.086  se 0.047  p=0.081
  stable             -0.473  se 0.055  p=1.3e-08
  labels:stable      +0.086  se 0.047  p=0.081
```

## Verdict

1. **The cost axis is prefix stability, and only prefix stability.** Removing the labels changes
   per-turn recompute by 6.1 tokens (the header-length asymmetry) while keeping stability changes
   it by 261–269 tokens. A 1.97 % prompt-size spread cannot host a 2.0× cost gap.
2. **The fidelity axis is the age-ordered, stable placement of the authoritative value — not the
   section labels.** Stripping the labels from the stable arm leaves substitution at 0/408
   (Wilson upper 0.009). The pre-declared SESOI for a substitution effect that would matter is
   0.03; the largest label effect anywhere in the factorial is 0.086, and it appears **only** in
   the sliding layout, where it moves substitution from 0.473 to 0.387 and is still three times
   the naive-arm floor. Labels therefore help a little when the front slides and do nothing once
   it does not: the two factors are substitutes on this axis too, not complements.
3. **The manuscript's "structural authority cue" attribution is not supported and must go.** The
   explanation that survives the factorial is placement: in the stable layout the accumulated block
   is read as an ordered, older-first record whose leading bytes never move, and the freshly-slid
   competitor sits in a separate volatile region; the wording of the region headers is immaterial.
4. Honest loose ends. `correct` is *lower* in the labelled stable cell (0.608) than in the
   unlabelled one (0.828), entirely in the `other` bucket (0.392 vs 0.172): the labelled arm
   produces more non-committal or restating answers that neither name the gold alone nor name a
   competitor. We do not claim labels are neutral for answer *style*; we claim they carry neither
   the cost nor the substitution effect. Second, this cell is the paper's stable arm re-run in a
   second epoch, and only 56 % of its 408 answer strings are byte-identical to the main run's
   `c01` arm at `temperature=0, seed=1234` (substitution agrees on 407/408; `correct` differs by
   0.088, again all in `other`). Server-side batching, not sampling, drives that: single-token
   endpoint answers are reproducible, hedging length is not. Cost claims are therefore anchored on
   the recompute counters, and the quality claims on substitution, which replicated.

## Reproduce

```bash
python run_joint_layout.py                 # 1632 calls against a local ollama, ~21 min
python check_layout_controls.py            # deterministic controls, no model calls
python analyze_layout.py                   # tables above -> layout_analysis.json
```
