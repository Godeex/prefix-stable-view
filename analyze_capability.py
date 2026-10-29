#!/usr/bin/env python
"""Is the danger-slice benefit really "capability-tracking"?

The manuscript used to say the benefit SIZE tracks model capability.  A reader
can produce a counter-example straight out of our own table (mistral-7B
delta=0.149 > qwen2.5-14B delta=0.128 > qwen3-14B delta=0.106 > qwen3-8B
delta=0.000), so the claim needs either an externally checkable capability
anchor or a downgrade.  This script does the honest version of both:

  (1) per-model risk difference D = P(subst | length_only) - P(subst | m_a) on the
      a-priori danger slice, one row per model (temperature-0 stratum, so the seven
      models are mutually independent), with a question-cluster bootstrap CI;
  (2) meta-analytic heterogeneity of those seven D's (Cochran Q, I^2) -- how much
      of the spread is real between-model variation rather than sampling noise;
  (3) Spearman rank correlation of D against the EXTERNAL anchor we can actually
      verify offline: the checkpoint size class reported by the model tag / model
      card (2B, 7B, 8B, 8B, 14B, 14B, 27B).  Exact permutation p over all 7! =
      5040 rank assignments, plus a leave-one-model-out jackknife range.  We do
      NOT use a cross-era leaderboard aggregate: the seven checkpoints were
      scored under different public protocols, so any single number we would
      quote would be an apples-to-oranges anchor chosen by us.

If (3) is not significant, or (2) says the spread is unstructured, the defensible
statement is "direction consistent, magnitude heterogeneous" rather than
"capability-tracking", and the manuscript says so.
"""
import collections
import itertools
import json
import math
import os
import random

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
THR = 0.60

# External anchor: checkpoint size class as reported by the ollama tag / public
# model card (billions of parameters).  Not measured by us, not chosen to fit.
SIZE_B = {
    "gemma2:2b": 2.0,
    "mistral:latest": 7.0,        # Mistral 7B
    "qwen3:8b": 8.0,
    "llama3:latest": 8.0,         # Llama-3 8B
    "qwen2.5:14b": 14.0,
    "qwen3:14b": 14.0,
    "qwen3.8:27b": 27.0,
}


def load_all():
    rows = []
    for fn in sorted(os.listdir(HERE)):
        if fn.startswith("lb_danger_scored_") and fn.endswith(".jsonl"):
            with open(os.path.join(HERE, fn), encoding="utf-8") as fh:
                rows += [json.loads(l) for l in fh]
    return rows


def paired_temp0(rows):
    """model -> {base_id: (lo_sub, ma_sub)} on the danger slice, temperature-0 only."""
    by = collections.defaultdict(dict)
    for r in rows:
        if r["temp"] != 0.0:
            continue
        by[r["model"]].setdefault(r["base_id"], {})[r["arm"]] = r
    out = {}
    for model, qmap in by.items():
        keep = {q: (g["length_only"]["sub_from_distractor"], g["m_a"]["sub_from_distractor"])
                for q, g in qmap.items()
                if "length_only" in g and "m_a" in g
                and g["length_only"]["distractor_max_sim"] >= THR}
        if keep:
            out[model] = keep
    return out


def percentile(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] * (c - k) + s[c] * (k - f)


def boot_ci(pairs, reps=4000, seed=1234):
    """Question-cluster bootstrap of the mean of (lo - ma)."""
    qs = list(pairs)
    random.seed(seed)
    out = []
    for _ in range(reps):
        samp = [pairs[random.choice(qs)] for _ in qs]
        out.append(sum(lo - ma for lo, ma in samp) / len(samp))
    return percentile(out, 2.5), percentile(out, 97.5)


def spearman_exact(xs, ys):
    """Spearman rho plus an exact two-sided permutation p over all n! rankings."""
    rho = stats.spearmanr(xs, ys).statistic
    obs = abs(rho)
    n = len(xs)
    ge = 0
    ys0 = list(ys)
    for perm in itertools.permutations(range(n)):
        r = stats.spearmanr(xs, [ys0[i] for i in perm]).statistic
        if abs(r) >= obs - 1e-12:
            ge += 1
    return rho, ge / math.factorial(n)


def paired_all(rows):
    """(model, temp) -> {base_id: (lo_sub, ma_sub)} on the danger slice."""
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["model"], r["temp"], r["arm"])][r["base_id"]] = r
    out = {}
    for (model, temp, arm), qmap in by.items():
        if arm not in ("length_only", "m_a"):
            continue
        out.setdefault((model, temp), {})[arm] = qmap
    kept = {}
    for key, arms in out.items():
        if len(arms) < 2:
            continue
        a, b = arms["length_only"], arms["m_a"]
        kept[key] = {q: (a[q]["sub_from_distractor"], b[q]["sub_from_distractor"])
                     for q in set(a) & set(b) if a[q]["distractor_max_sim"] >= THR}
    return kept


def main():
    rows = load_all()
    p0 = paired_temp0(rows)
    print("== per-model danger-slice benefit, temperature-0 strata only "
          f"(independent models; slice = LO kept a near-twin, max_sim>={THR:.2f}) ==")
    models = sorted(p0, key=lambda m: SIZE_B[m])
    ds, ses, ns = [], [], []
    for m in models:
        pairs = p0[m]
        n = len(pairs)
        d = sum(lo - ma for lo, ma in pairs.values()) / n
        lo_ci, hi_ci = boot_ci(pairs)
        se = (hi_ci - lo_ci) / (2 * 1.96)
        b = sum(1 for lo, ma in pairs.values() if lo and not ma)
        c = sum(1 for lo, ma in pairs.values() if ma and not lo)
        ds.append(d); ses.append(se); ns.append(n)
        print(f"  {m:15s} {SIZE_B[m]:5.0f}B n={n:3d} b/c={b}/{c}  "
              f"D={d:+.4f}  95%CI [{lo_ci:+.4f},{hi_ci:+.4f}]  SE={se:.4f}")

    # (2) heterogeneity across models: is the spread bigger than sampling noise?
    w = [1.0 / (s * s) for s in ses]
    sw = sum(w)
    dbar = sum(wi * di for wi, di in zip(w, ds)) / sw
    Q = sum(wi * (di - dbar) ** 2 for wi, di in zip(w, ds))
    dfree = len(ds) - 1
    I2 = max(0.0, (Q - dfree) / Q) * 100 if Q > 0 else 0.0
    p_q = float(stats.chi2.sf(Q, dfree))
    print(f"\n(2) heterogeneity of the {len(ds)} model-level D's: pooled D={dbar:+.4f}, "
          f"Cochran Q={Q:.2f} on {dfree} df, p={p_q:.4g}, I^2={I2:.1f}%")
    print(f"    spread of D: min={min(ds):+.4f} max={max(ds):+.4f} "
          f"range={max(ds) - min(ds):.4f}")

    # (3) Spearman against the external size anchor (exact permutation p, n=7)
    sizes = [SIZE_B[m] for m in models]
    rho, p_exact = spearman_exact(sizes, ds)
    print(f"\n(3) Spearman rho(D, checkpoint size class), n={len(ds)} models: "
          f"rho={rho:+.3f}, exact two-sided permutation p={p_exact:.3f} "
          f"(all {math.factorial(len(ds))} rank assignments; smallest attainable "
          f"two-sided p = {2 / math.factorial(len(ds)):.5f})")

    # leave-one-out range: how stable is the rank correlation in a 7-point sample?
    jack = []
    for drop in range(len(ds)):
        keep = [i for i in range(len(ds)) if i != drop]
        r = stats.spearmanr([sizes[i] for i in keep], [ds[i] for i in keep]).statistic
        jack.append((models[drop], r))
    vals = [r for _, r in jack]
    print(f"    leave-one-model-out rho range: [{min(vals):+.3f},{max(vals):+.3f}] "
          f"(drop -> rho: " + ", ".join(f"{m.replace(':', '_')}:{r:+.2f}"
                                        for m, r in jack) + ")")

    # monotonicity check: is D non-decreasing in size?
    order = sorted(zip(sizes, ds, models))
    viol = [(a, b) for (_, a, _), (_, b, _) in zip(order, order[1:]) if b < a - 1e-12]
    print(f"    size-monotonicity violations: {len(viol)} of {len(order) - 1} adjacent "
          f"pairs; e.g. " + "; ".join(f"{x:.3f} -> {y:.3f}" for x, y in viol[:3]))

    # (4) are the two temperature strata of one model separate calls?
    print("\n(4) same-model temperature pairs: shared question set but separate calls "
          "(identical outcome flags counted over the common slice)")
    allp = paired_all(rows)
    temps = collections.defaultdict(list)
    for (model, temp) in allp:
        temps[model].append(temp)
    for model in sorted(temps, key=lambda m: SIZE_B[m]):
        ts = sorted(temps[model])
        if len(ts) < 2:
            continue
        f0, f1 = allp[(model, ts[0])], allp[(model, ts[1])]
        common = set(f0) & set(f1)
        same = sum(1 for q in common if f0[q] == f1[q])
        seeds = collections.Counter()
        for r in rows:
            if r["model"] == model:
                seeds[(r["temp"], r["seed"])] += 1
        sd = ", ".join(f"t{t}/s{s}" for (t, s) in sorted(seeds))
        print(f"    {model:15s} t{ts[0]} vs t{ts[1]}: n={len(f0)}/{len(f1)} "
              f"common={len(common)} identical_pairs={same}/{len(common)}  runs[{sd}]")



if __name__ == "__main__":
    main()
