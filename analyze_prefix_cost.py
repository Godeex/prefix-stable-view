#!/usr/bin/env python
"""B-lite analysis: paired per-turn RECOMPUTE (prefix-cache cost) between maintenance policies.

Cost driver = recompute tokens = prompt_eval_count - prompt_eval_cached_count (the tokens the
GPU must actually prefill; cached tokens are free). Primary outcome is recompute; prefill latency
(prompt_eval_duration) is the corroborating real-clock measure.

Three arms at matched quality (all must answer the recall query; accuracy parity asserted):
  length_only    naive recency window, front slides            -> proposal's competitor
  prefix_stable  M-A append-only compact STATE + volatile tail -> the proposal
  pinned_raw     SAME raw size as length_only but stable front -> CONTROL isolating 'slide'
                 from 'sends less': if pinned_raw recomputes ~like prefix_stable and << length_only,
                 the cost is prefix invalidation, not reduced content.

Paired per (session,turn). Turn observations within a session are correlated -> SESSION-cluster
bootstrap of the mean paired difference (resample sessions, carry all their turns together).
"""
import json, os, glob, math, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


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
    return {k: v for k, v in d.items() if len(v) == 3}


def paired(pairs, a, b, field):
    """list of (a-b) per pair for given field."""
    return [p[a][field] - p[b][field] for p in pairs]


def cluster_boot(pairs, keys, field, a, b, B=6000, seed=1234):
    """D_hat and session-cluster bootstrap CI for mean(a-b) of `field`."""
    dvals = paired(pairs, a, b, field)
    if not dvals:
        return None
    D = float(np.mean(dvals))
    by_sess = {}
    for p, dv in zip(pairs, dvals):
        sid = next(iter(p.values()))["session_id"]   # pair dict is keyed by arm, not session
        by_sess.setdefault(sid, []).append(dv)
    sess = list(by_sess)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        pick = [by_sess[sess[i]] for i in rng.integers(0, len(sess), len(sess))]
        flat = [x for grp in pick for x in grp]
        boots.append(float(np.mean(flat)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_le = float(np.mean([bb <= 0 for bb in boots]))
    return dict(D=D, ci=(float(lo), float(hi)), p=p_le, n_pairs=len(pairs), n_sess=len(sess))


def mean(rows, field):
    return float(np.mean([r[field] for r in rows])) if rows else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="prefix_cost_scored_*.jsonl")
    args = ap.parse_args()
    rows = load(args.glob)
    if not rows:
        print("no scored rows"); return
    pairs = list(by_arm(rows).values())

    print(f"== B-lite prefix-cache cost  model={rows[0]['model']} temp={rows[0]['temp']} "
          f"pairs={len(pairs)} sessions={len(set(r['session_id'] for r in rows))} ==")
    print(f"{'arm':16s} {'acc':>6s} {'mean_prompt':>11s} {'mean_cached':>11s} "
          f"{'mean_recomp':>11s} {'hit%':>6s} {'prefill_ms':>10s}")
    for arm in ("length_only", "prefix_stable", "pinned_raw"):
        sub = [p[arm] for p in pairs]
        acc = np.mean([r["correct"] for r in sub])
        mp = mean(sub, "prompt_eval_count"); mc = mean(sub, "prompt_eval_cached_count")
        mr = mean(sub, "recompute"); ms = mean(sub, "prompt_eval_duration_ms")
        print(f"{arm:16s} {acc:6.3f} {mp:11.0f} {mc:11.0f} {mr:11.0f} {100*mc/max(mp,1):5.1f}% {ms:10.1f}")

    # matched-quality assertion
    accs = {a: round(float(np.mean([p[a]["correct"] for p in pairs])), 4)
            for a in ("length_only", "prefix_stable", "pinned_raw")}
    print(f"\nmatched-quality (accuracy all arms ~1): {accs}  "
          f"-> {'OK' if min(accs.values()) >= 0.95 else 'QUALITY NOT MATCHED'}")
    mp = {a: round(mean([p[a] for p in pairs], "prompt_eval_count")) for a in
          ("length_only", "prefix_stable", "pinned_raw")}
    print(f"matched-size control: length_only={mp['length_only']} tok vs pinned_raw={mp['pinned_raw']} tok "
          f"(should be ~equal) ; prefix_stable={mp['prefix_stable']} (compact)")

    print("\n== session-cluster bootstrap of mean PAIRED recompute difference (tokens) ==")
    for (a, b, label) in [("length_only", "prefix_stable", "LO - M-A"),
                          ("length_only", "pinned_raw", "LO - pinned (pure SLIDE effect, matched size)"),
                          ("prefix_stable", "pinned_raw", "M-A - pinned (compaction beyond stability)")]:
        r = cluster_boot(pairs, "session_id", "recompute", a, b)
        sig = "SIGNIFICANT" if r and r["ci"][0] > 0 else ("(reverse)" if r and r["ci"][1] < 0 else "n.s.")
        print(f"  D({label:38s}) = {r['D']:8.1f}  95%CI[{r['ci'][0]:.1f},{r['ci'][1]:.1f}]  "
              f"P(<=0)={r['p']:.2g}  -> {sig}")

    print("\n== corroborating prefill latency (ms) ==")
    for (a, b, label) in [("length_only", "prefix_stable", "LO - M-A"),
                          ("length_only", "pinned_raw", "LO - pinned")]:
        r = cluster_boot(pairs, "session_id", "prompt_eval_duration_ms", a, b)
        print(f"  D({label:12s}) = {r['D']:7.1f} ms  95%CI[{r['ci'][0]:.1f},{r['ci'][1]:.1f}]  "
              f"P(<=0)={r['p']:.2g}  -> {'SIGNIFICANT' if r['ci'][0] > 0 else 'n.s.'}")

    # headline ratios
    lo = mean([p["length_only"] for p in pairs], "recompute")
    ma = mean([p["prefix_stable"] for p in pairs], "recompute")
    pin = mean([p["pinned_raw"] for p in pairs], "recompute")
    print(f"\nrecompute reduction: M-A vs naive = {lo/max(ma,1):.1f}x ; "
          f"pinned(stable same-size) vs naive = {lo/max(pin,1):.1f}x")


if __name__ == "__main__":
    main()
