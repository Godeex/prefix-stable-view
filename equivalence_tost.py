#!/usr/bin/env python
"""M-minor-3: turn the "below SESOI" reading into an actual equivalence test.

Section VI.E currently says joint (c11) and prefix-only (c01) differ by less than the
pre-declared smallest effect size of interest, and then calls the two levers substitutes. A
difference that misses a SESOI is *undetermined*, not equivalent, so
the claim needs a two one-sided tests (TOST) result with the margins stated.

TOST here is cluster-based, because the unit of randomization is the session, not the turn:
resample sessions with replacement, take the paired mean difference inside each resample, and
read off the bootstrap distribution of the contrast. The TOST p-value is
  p = max( Pr(D >= +margin), Pr(D <= -margin) )
so p <= alpha means the whole one-sided tail is inside the margin, i.e. equivalence at level
alpha. Margins are the pre-declared SESOIs (0.03 substitution, 20 recompute tokens), not
post-hoc widths, and both contrasts (joint vs prefix-only, joint vs semantic-only) are tested
so the substitution of the two levers is shown on each axis separately.

Env: RJ_FILE (default joint_scored_qwen3_14b.jsonl).
"""
import json, os, statistics as st, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FN = os.environ.get("RJ_FILE") or os.path.join(HERE, "joint_scored_qwen3_14b.jsonl")
M_SUBST, M_RECOMP = 0.03, 20.0
B = 6000


def paired(rows, a, b, field):
    d = defaultdict(dict)
    for r in rows:
        d[(r["session_id"], r["turn"])][r["cfg"]] = r[field]
    out = defaultdict(list)
    for (sid, _t), g in d.items():
        if a in g and b in g:
            out[sid].append(g[a] - g[b])
    keys = list(out)
    point = st.mean([x for v in out.values() for x in v])
    rng = np.random.default_rng(1234)
    boot = []
    for _ in range(B):
        pick = [out[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
        boot.append(st.mean([x for g in pick for x in g]))
    return point, boot


def tost(point, boot, margin):
    p_hi = sum(1 for x in boot if x >= margin) / len(boot)
    p_lo = sum(1 for x in boot if x <= -margin) / len(boot)
    p = max(p_hi, p_lo)
    lo, hi = [float(q) for q in np.percentile(boot, [2.5, 97.5])]
    inside = lo > -margin and hi < margin
    return {"D": point, "ci95": (lo, hi), "p_hi": p_hi, "p_lo": p_lo, "p": p,
            "equiv_05": p <= 0.05, "ci_inside": inside}


def main():
    rows = [json.loads(l) for l in open(FN, encoding="utf-8")]
    print(f"== TOST equivalence, {os.path.basename(FN)}, {len(rows)} rows, "
          f"margins: substitution +/-{M_SUBST}, recompute +/-{M_RECOMP} tok, "
          f"{B} session-cluster resamples ==")
    for a, b, lbl in [("c11", "c01", "joint vs prefix-only"),
                      ("c11", "c10", "joint vs semantic-only")]:
        for field, margin, flbl in [("substitution", M_SUBST, "subst"),
                                    ("recompute", M_RECOMP, "recompute")]:
            pt, boot = paired(rows, a, b, field)
            r = tost(pt, boot, margin)
            print(f"  {lbl:22s} {flbl:9s} D {r['D']:+.4f} "
                  f"95%CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
                  f"vs +/-{margin}: p(>=+m)={r['p_hi']:.4f} p(<=-m)={r['p_lo']:.4f} "
                  f"TOST p={r['p']:.4f} -> "
                  f"{'equivalent at 0.05' if r['equiv_05'] else 'NOT established'}"
                  f"{' (CI inside margin)' if r['ci_inside'] else ''}")


if __name__ == "__main__":
    main()
