#!/usr/bin/env python
"""Bounded-fold analysis: does a budgeted, folding head keep the prefix-cache advantage?

Reads prefix_bounded_scored_*.jsonl (arms: length_only, prefix_stable [UNBOUNDED head],
bounded_stable [head capped at H_MAX, folds oldest->summary on overflow, carries fold_event]).

Reports, at matched quality (all arms recall the newest OBSk=val):
  1. per-arm mean recompute / prefill (does bounded_stable stay far under naive length_only?)
  2. the SAWTOOTH: mean recompute of bounded_stable on APPEND turns vs FOLD turns -- the fold
     spike is the direct measurement of the Theta(H_max*Delta) term in Eq (eq:bounded), the
     append floor is Theta(V).
  3. session-cluster bootstrap of paired recompute diffs LO-bounded and bounded-unbounded.
"""
import json, os, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(pattern):
    rows = []
    for fn in sorted(glob.glob(os.path.join(HERE, pattern))):
        for l in open(fn, encoding="utf-8"):
            rows.append(json.loads(l))
    return rows


def mean(rows, field):
    return float(np.mean([r[field] for r in rows])) if rows else 0.0


def cluster_boot(pairs, a, b, field, B=6000, seed=1234):
    dvals = [p[a][field] - p[b][field] for p in pairs]
    if not dvals:
        return None
    D = float(np.mean(dvals))
    by_sess = {}
    for p, dv in zip(pairs, dvals):
        sid = next(iter(p.values()))["session_id"]
        by_sess.setdefault(sid, []).append(dv)
    sess = list(by_sess)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        pick = [by_sess[sess[i]] for i in rng.integers(0, len(sess), len(sess))]
        flat = [x for grp in pick for x in grp]
        boots.append(float(np.mean(flat)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(D=D, ci=(float(lo), float(hi)), p=float(np.mean([bb <= 0 for bb in boots])),
                n=len(pairs), nsess=len(sess))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="prefix_bounded_scored_*.jsonl")
    args = ap.parse_args()
    rows = load(args.glob)
    if not rows:
        print("no scored rows"); return
    # paired by (session,turn) across all three arms
    d = {}
    for r in rows:
        d.setdefault((r["session_id"], r["turn"]), {})[r["arm"]] = r
    pairs = [v for v in d.values() if len(v) == 3]
    print(f"== bounded-fold  model={rows[0]['model']} temp={rows[0]['temp']} "
          f"pairs={len(pairs)} sessions={len(set(r['session_id'] for r in rows))} ==")
    print(f"{'arm':16s} {'acc':>6s} {'mean_prompt':>11s} {'mean_recomp':>11s} {'prefill_ms':>10s}")
    for arm in ("length_only", "prefix_stable", "bounded_stable"):
        sub = [p[arm] for p in pairs]
        acc = np.mean([r["correct"] for r in sub])
        print(f"{arm:16s} {acc:6.3f} {mean(sub,'prompt_eval_count'):11.0f} "
              f"{mean(sub,'recompute'):11.0f} {mean(sub,'prompt_eval_duration_ms'):10.1f}")

    bs = [p["bounded_stable"] for p in pairs]
    app = [r for r in bs if r["fold_event"] == 0]
    fold = [r for r in bs if r["fold_event"] == 1]
    print(f"\nSAWTOOTH bounded_stable:  append-turn n={len(app)} mean_recomp={mean(app,'recompute'):.0f}"
          f"   fold-turn n={len(fold)} mean_recomp={mean(fold,'recompute'):.0f}")
    print(f"  head-size check: bounded mean prompt={mean(bs,'prompt_eval_count'):.0f} vs "
          f"unbounded prefix_stable mean prompt={mean([p['prefix_stable'] for p in pairs],'prompt_eval_count'):.0f}"
          f"  (bounded stays flat, unbounded grows)")

    print("\nsession-cluster bootstrap of mean PAIRED recompute (tokens):")
    for (a, b, lbl) in [("length_only", "bounded_stable", "LO - bounded (does budget keep the win?)"),
                        ("bounded_stable", "prefix_stable", "bounded - unbounded (fold surcharge)"),
                        ("length_only", "prefix_stable", "LO - unbounded M-A")]:
        r = cluster_boot(pairs, a, b, "recompute")
        if r:
            sig = "SIGNIFICANT" if r["ci"][0] > 0 else ("(reverse)" if r["ci"][1] < 0 else "n.s.")
            print(f"  D({lbl:44s}) = {r['D']:8.1f} CI[{r['ci'][0]:.1f},{r['ci'][1]:.1f}] "
                  f"P(<=0)={r['p']:.2g} -> {sig}")
    lo = mean([p["length_only"] for p in pairs], "recompute")
    bo = mean(bs, "recompute")
    ma = mean([p["prefix_stable"] for p in pairs], "recompute")
    print(f"\nrecompute reduction under budget: bounded vs naive = {lo/max(bo,1):.1f}x ; "
          f"unbounded vs naive = {lo/max(ma,1):.1f}x")


if __name__ == "__main__":
    main()
