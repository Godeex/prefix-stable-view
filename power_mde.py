#!/usr/bin/env python
"""Minimum detectable effect for the pooled danger-slice risk difference.

The primary read is a question-cluster bootstrap over 47 questions, so the design's precision is
set by the between-QUESTION spread of the paired difference (LO - MA) averaged over a question's
config rows -- not by the 652 rows. We estimate that spread from the frozen run files and report
(a) the SE and 80%-power MDE at n=47, (b) the n needed to push the 95% CI lower bound past the
pre-declared SESOI of 0.03 at the observed D, and (c) the n needed for the MDE itself to fall below
0.03. Analytic (normal-approximation) numbers, cross-checked against the bootstrap CI width.
"""
import json, glob, os, math, collections, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
THR = 0.60

rows = []
for fn in sorted(glob.glob(os.path.join(HERE, "lb_danger_scored_*.jsonl"))):
    for l in open(fn, encoding="utf-8"):
        rows.append(json.loads(l))

by = collections.defaultdict(dict)
for r in rows:
    by[(f"{r['model']}_t{r['temp']}", r["base_id"])][r["arm"]] = r
per_q = collections.defaultdict(list)
for (cfg, q), g in by.items():
    if "length_only" in g and "m_a" in g and g["length_only"]["distractor_max_sim"] >= THR:
        per_q[q].append(g["length_only"]["sub_from_distractor"] - g["m_a"]["sub_from_distractor"])

qs = sorted(per_q)
qmean = [statistics.fmean(per_q[q]) for q in qs]
n = len(qs)
rows_tot = sum(len(v) for v in per_q.values())
D = sum(sum(v) for v in per_q.values()) / rows_tot
sd = statistics.stdev(qmean)
se = sd / math.sqrt(n)
print(f"questions n={n}  paired rows={rows_tot}  D_hat={D:.4f}")
print(f"per-question mean-difference SD={sd:.4f}  SE={se:.4f}")
print(f"bootstrap-equivalent half-width used by the paper: 0.0800 "
      f"(CI [0.0031,0.1631]) vs normal 1.96*SE={1.96*se:.4f}")
mde80 = (1.96 + 0.8416) * se
print(f"80%-power MDE at n={n}: {mde80:.4f}  (two-sided alpha=0.05)")
print(f"50%-power MDE (= 1.96*SE): {1.96*se:.4f}")
# n needed so the 95% lower bound clears the SESOI at the OBSERVED D (sd/sqrt(n) scaling)
need_lb = (1.96 * sd / (D - 0.03)) ** 2 if D > 0.03 else float("inf")
# n needed so the 80%-power MDE itself is <= 0.03
need_mde = ((1.96 + 0.8416) * sd / 0.03) ** 2
print(f"n to push the 95% lower bound past 0.03 at D={D:.4f}: {need_mde if False else need_lb:.0f} questions")
print(f"n for the 80%-power MDE to fall to 0.03:               {need_mde:.0f} questions")
