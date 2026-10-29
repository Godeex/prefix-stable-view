#!/usr/bin/env python
"""Cost-leg analysis: what does a rewritable current-binding INDEX cost per turn?

Cost driver = recompute = prompt_eval_count - prompt_eval_cached_count; prefill duration is the
corroborating real-clock measure. Same protocol and same session-cluster bootstrap as the published
cost leg (analyze_prefix_cost.py), on the published sessions with the index arms inserted
(gen_index_cost_trials.py asserts the baseline arms are byte-identical to the published trials).

Tests, fixed before the run:
  (i)   matched quality: every arm answers the recall query, so a recompute gap is placement only;
  (ii)  the index premium: mean paired recompute (index_I* - prefix_stable), which the model
        predicts to be Theta(I) -- the index block plus nothing else, because the head stays a
        shared byte prefix (asserted per row at generation time);
  (iii) the slope: regress the per-turn paired premium on I = {2, 8, 24} within session, cluster-
        robust on sessions, and report tokens per index row. Theta(I) predicts an approximately
        linear relation with intercept near zero;
  (iv)  the counterfactual the index replaces: moving the value into the sliding window
        (length_only) is what the controller does today when it wants the answer near the end,
        and its recompute is the alternative cost, so report index vs length_only as well.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("prefix_stable", "index_I2", "index_I8", "index_I24", "length_only")


def load(pattern):
    rows = []
    for fn in sorted(glob.glob(os.path.join(HERE, pattern))):
        for l in open(fn, encoding="utf-8"):
            rows.append(json.loads(l))
    return rows


def by_arm(rows):
    d = {}
    for r in rows:
        d.setdefault((r["session_id"], r["turn"]), {})[r["arm"]] = r
    return {k: v for k, v in d.items() if all(a in v for a in ARMS)}


def mean(rows, field):
    return float(np.mean([r[field] for r in rows])) if rows else 0.0


def cluster_boot(pairs, field, a, b, B=6000, seed=1234):
    """Session-cluster bootstrap CI for the mean paired difference a - b of `field`."""
    vals = [(p[a][field] - p[b][field], next(iter(p.values()))["session_id"]) for p in pairs]
    if not vals:
        return None
    by_sess = {}
    for v, sid in vals:
        by_sess.setdefault(sid, []).append(v)
    sess = list(by_sess)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        pick = [by_sess[sess[i]] for i in rng.integers(0, len(sess), len(sess))]
        boots.append(float(np.mean([x for g in pick for x in g])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(D=float(np.mean([v for v, _ in vals])), ci=(float(lo), float(hi)),
                p=float(np.mean([bb <= 0 for bb in boots])), n_pairs=len(vals), n_sess=len(sess))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="index_cost_scored_*.jsonl")
    args = ap.parse_args()
    rows = load(args.glob)
    if not rows:
        print("no scored rows")
        return
    pairs = list(by_arm(rows).values())
    print(f"== index cost  model={rows[0]['model']} temp={rows[0]['temp']} pairs={len(pairs)} "
          f"sessions={len(set(r['session_id'] for r in rows))} "
          f"turns={sorted({r['turn'] for r in rows})[0]}..{sorted({r['turn'] for r in rows})[-1]} ==")
    print(f"{'arm':14s} {'acc':>6s} {'mean_prompt':>11s} {'mean_cached':>11s} {'mean_recomp':>11s} "
          f"{'hit%':>6s} {'prefill_ms':>10s}")
    for arm in ARMS:
        sub = [p[arm] for p in pairs]
        mp = mean(sub, "prompt_eval_count"); mc = mean(sub, "prompt_eval_cached_count")
        print(f"{arm:14s} {np.mean([r['correct'] for r in sub]):6.3f} {mp:11.0f} {mc:11.0f} "
              f"{mean(sub, 'recompute'):11.1f} {100 * mc / max(mp, 1):5.1f}% "
              f"{mean(sub, 'prompt_eval_duration_ms'):10.1f}")
    accs = {a: round(float(np.mean([p[a]["correct"] for p in pairs])), 4) for a in ARMS}
    print(f"\n(i) matched-quality premise: {accs} -> "
          f"{'OK' if min(accs.values()) >= 0.95 else 'QUALITY NOT MATCHED'}")

    print("\n== (ii) session-cluster bootstrap of mean PAIRED recompute difference (tokens) ==")
    for b in ("prefix_stable", "length_only"):
        for a in ("index_I2", "index_I8", "index_I24"):
            if a == b:
                continue
            r = cluster_boot(pairs, "recompute", a, b)
            sig = "SIGNIFICANT" if r["ci"][0] > 0 else ("(reverse)" if r["ci"][1] < 0 else "n.s.")
            print(f"  D({a:9s} - {b:13s}) = {r['D']:8.1f} 95%CI[{r['ci'][0]:.1f},{r['ci'][1]:.1f}] "
                  f"P(<=0)={r['p']:.2g} -> {sig}   [n={r['n_pairs']} pairs, {r['n_sess']} sessions]")

    print("\n== (iii) slope of the premium on I (tokens per index row) ==")
    df = pd.DataFrame([{"session_id": p["index_I2"]["session_id"], "turn": p["index_I2"]["turn"],
                        "I": I,
                        "prem": p[f"index_I{I}"]["recompute"] - p["prefix_stable"]["recompute"]}
                       for p in pairs for I in (2, 8, 24)])
    X = sm.add_constant(df[["I"]].astype(float))
    m = sm.OLS(df["prem"].astype(float), X.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": df["session_id"].to_numpy()}, use_t=True)
    for n, b, se, p in zip(["intercept", "slope_I"], m.params, m.bse, m.pvalues):
        print(f"  {n:10s} {b:+8.2f} se {se:5.2f} [{b - 1.96 * se:+.2f},{b + 1.96 * se:+.2f}] p={p:.3g}")
    print(f"  rows={len(df)} (3 index sizes x {len(pairs)} session-turns), "
          f"clustered on {df['session_id'].nunique()} sessions")

    print("\n== (iv) corroborating prefill latency (ms), premium vs baseline ==")
    for a in ("index_I2", "index_I8", "index_I24"):
        r = cluster_boot(pairs, "prompt_eval_duration_ms", a, "prefix_stable")
        print(f"  D({a:9s} - prefix_stable) = {r['D']:7.1f} ms 95%CI[{r['ci'][0]:.1f},{r['ci'][1]:.1f}] "
              f"P(<=0)={r['p']:.2g}")
    lo = mean([p["length_only"] for p in pairs], "recompute")
    st_ = mean([p["prefix_stable"] for p in pairs], "recompute")
    for a in ("index_I2", "index_I8", "index_I24"):
        v = mean([p[a] for p in pairs], "recompute")
        print(f"\nrecompute: {a} = {v:.1f} tok/turn vs prefix_stable {st_:.1f} "
              f"(x{v / max(st_, 1):.2f}) and length_only {lo:.1f} "
              f"({lo / max(v, 1):.2f}x cheaper than naive)")


if __name__ == "__main__":
    main()
