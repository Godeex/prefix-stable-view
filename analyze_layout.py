#!/usr/bin/env python
"""Analysis: does the fidelity gain come from prefix stability or from the block labels?

The main 2x2 shows the stable arm reaching substitution ~0 while competing near-twin values are
still in context, and the manuscript attributes that to a labelled STATE block acting as an
authority cue. That attribution is testable: the layout run measures all four cells of the
(labels x stability) factorial on identical content, so each axis has a main effect and they have an
interaction.

Reported:
  cell table        substitution / correct / other with Wilson CIs, mean recompute, mean prompt
                    tokens, mean competing values in context (matched by construction);
  axis contrasts    paired McNemar + cluster-bootstrap CI for the label effect at stable layout
                    (s1l1-s1l0), at sliding layout (s0l1-s0l0), and for the stability effect with
                    labels (s1l1-s0l1) and without (s1l0-s0l0);
  factorial model   binomial GLM substitution ~ labels*stable with session-clustered SEs, so the
                    interaction is estimated rather than eyeballed;
  cost attribution  the same contrasts on recompute tokens, which should load on stability only;
  SESOI verdict     the pre-declared 0.03 substitution margin applied to the label effect, so a null
                    is read as "below the threshold that would matter", not as "equal".
"""
import json, os, statistics as st, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

FN = os.environ.get("RJ_LAYOUT_OUT") or os.path.join(HERE, "joint_layout_scored_qwen3_14b.jsonl")
CELLS = ("s0l0", "s0l1", "s1l0", "s1l1")
SESOI_SUBST = 0.03


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot(vals_by_sess, B=6000, seed=1234):
    keys = list(vals_by_sess)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = [vals_by_sess[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
        flat = [x for g in pick for x in g]
        out.append(st.mean(flat))
    return [float(np.percentile(out, q)) for q in (2.5, 97.5)]


def paired_contrast(rows, a, b, field):
    """rows keyed by (session,turn) -> cell -> record."""
    d = defaultdict(list)
    for (sid, t), g in rows.items():
        if a in g and b in g:
            d[sid].append(g[a][field] - g[b][field])
    flat = [x for v in d.values() for x in v]
    if not flat:
        return None
    return {"D": st.mean(flat), "ci": boot(d), "n": len(flat),
            "sign": "a>b" if st.mean(flat) > 0 else "a<b"}


def mcnemar(rows, a, b, field):
    bb = cc = 0
    for g in rows.values():
        if a in g and b in g:
            x, y = bool(g[a][field]), bool(g[b][field])
            if x and not y:
                bb += 1
            elif y and not x:
                cc += 1
    p = float(stats.binomtest(bb, bb + cc, 0.5).pvalue) if bb + cc else 1.0
    return {"b": bb, "c": cc, "p": p}


def _cell_subst(rows, cell):
    v = [g[cell]["substitution"] for g in rows.values() if cell in g]
    return st.mean(v) if v else float("nan")


def main():
    R = [json.loads(l) for l in open(FN, encoding="utf-8")]
    # cell name is s{stable}l{labels}; re-derive the flags from it so a row-level recording bug in an
    # earlier runner version cannot silently enter the model.
    for r in R:
        r["stable"], r["labels"] = int(r["cell"][1] == "1"), int(r["cell"][3] == "1")
    idx = defaultdict(dict)
    for r in R:
        idx[(r["session_id"], r["turn"])][r["cell"]] = r
    complete = {k: v for k, v in idx.items() if len(v) == len(CELLS)}
    print(f"== layout 2x2 (labels x stability), {len(R)} rows, "
          f"{len(complete)} cells with all four arms, model {R[0]['model']} ==")
    print(f"{'cell':6s} {'labels':>6s} {'stable':>6s} {'n':>5s} {'subst':>7s} {'95%CI':>16s} "
          f"{'correct':>8s} {'other':>6s} {'recomp':>7s} {'prompt':>7s} {'competing':>9s}")
    for c in CELLS:
        s = [r for r in R if r["cell"] == c]
        n = len(s)
        k = sum(r["substitution"] for r in s)
        co = sum(r["correct"] for r in s)
        lo, hi = wilson(k, n)
        print(f"{c:6s} {s[0]['labels']:6d} {s[0]['stable']:6d} {n:5d} {k/n:7.3f} "
              f"[{lo:.3f},{hi:.3f}] {co/n:8.3f} {1-k/n-co/n:6.3f} "
              f"{st.mean(r['recompute'] for r in s):7.1f} "
              f"{st.mean(r['prompt_eval_count'] for r in s):7.1f} "
              f"{st.mean(r['competing_in_ctx'] for r in s):9.2f}")

    print("\n== paired contrasts on the same (session,turn) cells ==")
    pairs = [("s1l1", "s1l0", "label effect under stable layout"),
             ("s0l1", "s0l0", "label effect under sliding layout"),
             ("s1l1", "s0l1", "stability effect with labels"),
             ("s1l0", "s0l0", "stability effect without labels")]
    for a, b, lbl in pairs:
        rs = {k: v for k, v in complete.items()}
        sr = paired_contrast(rs, a, b, "substitution")
        rc = paired_contrast(rs, a, b, "recompute")
        mn = mcnemar(rs, a, b, "substitution")
        print(f"  {lbl:34s} subst {sr['D']:+.3f} CI[{sr['ci'][0]:+.3f},{sr['ci'][1]:+.3f}] "
              f"McNemar b={mn['b']} c={mn['c']} p={mn['p']:.3g} | "
              f"recompute {rc['D']:+7.1f} tok CI[{rc['ci'][0]:+.1f},{rc['ci'][1]:+.1f}]")

    # factorial model on the two axes with session-clustered SEs.
    # Both stable cells are 0/408 events, so the logit MLE does not exist (complete separation) and
    # statsmodels raises a singular Hessian; we therefore report a cluster-robust linear probability
    # model, whose coefficients are risk differences and stay identified at 0/1 cells.
    frame = pd.DataFrame([[r["labels"], r["stable"], r["labels"] * r["stable"]] for r in R],
                         columns=["labels", "stable", "labels_x_stable"], dtype=float)
    X = sm.add_constant(frame)
    y = np.array([r["substitution"] for r in R], dtype=float)
    grp = np.array([r["session_id"] for r in R])
    separated = any(len({r["substitution"] for r in R if r["cell"] == c}) < 2 for c in CELLS)
    lpm = sm.OLS(y, X.to_numpy()).fit(cov_type="cluster",
                                      cov_kwds={"groups": grp}, use_t=True)
    print("\n== substitution ~ labels*stable, cluster-robust linear probability model "
          f"(risk differences; logit not estimable: complete separation={separated}) ==")
    for name, b, se, p in zip(X.columns, lpm.params, lpm.bse, lpm.pvalues):
        print(f"  {name:16s} RD {b:+.3f} se {se:.3f} 95%CI[{b-1.96*se:+.3f},{b+1.96*se:+.3f}] p={p:.3g}")

    # cost axis sanity: prompt size matched across cells
    sizes = {c: st.mean(r["prompt_eval_count"] for r in R if r["cell"] == c) for c in CELLS}
    spread = (max(sizes.values()) - min(sizes.values())) / max(sizes.values())
    print(f"\n  prompt-token spread across the four cells: {100*spread:.2f}% "
          f"(so a cost gap is not a size gap)")

    # marginal effects: each axis averaged over the other, so the split reads off the table
    marg = {}
    for axis in ("labels", "stable"):
        for lev in (0, 1):
            s = [r for r in R if r[axis] == lev]
            marg[f"{axis}={lev}"] = {
                "n": len(s),
                "subst": st.mean(r["substitution"] for r in s),
                "correct": st.mean(r["correct"] for r in s),
                "recompute": st.mean(r["recompute"] for r in s),
                "competing": st.mean(r["competing_in_ctx"] for r in s),
            }
        a, b = marg[f"{axis}=0"], marg[f"{axis}=1"]
        print(f"  marginal {axis}: subst {a['subst']:.3f} -> {b['subst']:.3f} "
              f"({b['subst']-a['subst']:+.3f}) | recompute {a['recompute']:.1f} -> "
              f"{b['recompute']:.1f} ({b['recompute']-a['recompute']:+.1f} tok) | "
              f"competing {a['competing']:.2f} vs {b['competing']:.2f}")

    label_max = max(abs(marg["labels=1"]["subst"] - marg["labels=0"]["subst"]),
                    abs(_cell_subst(complete, "s1l1") - _cell_subst(complete, "s1l0")),
                    abs(_cell_subst(complete, "s0l1") - _cell_subst(complete, "s0l0")))
    print(f"\n  largest label effect on substitution across the factorial: {label_max:.3f} "
          f"vs pre-declared SESOI {SESOI_SUBST} -> "
          f"{'labels matter' if label_max >= SESOI_SUBST else 'below SESOI: labels do not carry the fidelity gain'}")

    out = {"cells": {c: {"subst": st.mean(r["substitution"] for r in R if r["cell"] == c),
                         "recompute": st.mean(r["recompute"] for r in R if r["cell"] == c)}
                     for c in CELLS},
           "marginals": marg,
           "label_max_subst_effect": label_max,
           "sesoi_subst": SESOI_SUBST,
           "prompt_spread": spread,
           "lpm": {str(n): [float(b), float(se), float(p)]
                   for n, b, se, p in zip(X.columns, lpm.params, lpm.bse, lpm.pvalues)},
           "logit_estimable": not separated,
           "contrasts": {f"{a}-{b}": {"subst": paired_contrast(complete, a, b, "substitution"),
                                      "recompute": paired_contrast(complete, a, b, "recompute"),
                                      "mcnemar": mcnemar(complete, a, b, "substitution")}
                         for a, b, _ in pairs},
           "n_rows": len(R), "n_complete": len(complete)}
    json.dump(out, open(os.path.join(HERE, "layout_analysis.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
