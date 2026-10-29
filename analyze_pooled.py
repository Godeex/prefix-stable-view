#!/usr/bin/env python
"""Pooled paired significance for M-A's passage danger-slice benefit.

The per-config table showed b>c in every CAPABLE model (both 14B, both 27B configs) but only one
cell individually significant -> that reads as cherry-picking unless we pool the
discordant evidence CORRECTLY. Naive pooling over all (config,question) pairs is anti-conservative:
the same question recurs across configs and 27B t0 / t0.5 are near-duplicate observations. So we
report three, from weakest to most defensible:
  (1) naive pooled McNemar binomtest(sum b, sum c)  [flagged anti-conservative]
  (2) Mantel-Haenszel over INDEPENDENT-model strata (one row per model, using the temp=0 config to
      avoid the 27B double-count): T = (sum(b-c))^2 / sum(b+c) ~ chi2_1
  (3) QUESTION-cluster bootstrap over the paired difference (LO - MA), pooling capable configs and
      carrying all of a question's config-rows together -> percentile CI + P(D<=0). This is the
      primary pooled test; it respects within-question correlation across models.
Danger slice = the length_only (naive truncation) arm kept a near-twin (distractor_max_sim >= THR).
"""
import json, os, glob, math, random, collections, statistics
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
THR = 0.60
# CAPABLE = independent temp=0 models that showed b>c (excludes qwen3:8b which ties 2/2 = genuine wash).
# Now spans 3 model FAMILIES: qwen3 (14b/27b), qwen2.5 (14b), llama3, mistral -> cross-family generality.
CAPABLE = ("qwen3:14b", "qwen2.5:14b", "qwen3.8:27b", "llama3:latest", "mistral:latest")
# The pooling must be declared a priori rather than chosen after seeing
# which layers look favourable, so the PRIMARY stratum set is every served model (no
# filter) and SUBGROUP is retained only as the pre-declared sensitivity analysis.
# SUBGROUP = the five checkpoints of >=7B whose temperature-0 stratum showed b>c
# (i.e. dropping gemma2:2b and the tied qwen3:8b), a criterion read off the same
# table the result appears in, hence labelled a subgroup and never the headline.
# Caution: that filter uses the OUTCOME (b>c), which mechanically inflates the
# effect. The SIZE-ONLY pool keeps every checkpoint of >=7B params (all served models except
# gemma2:2b -> 6 checkpoints x 2 temperatures = 12 strata) with NO outcome filter; it is the
# defensible sensitivity analysis and is what the manuscript now quotes.
# Sizes (params): gemma2:2b<7B; qwen3:8b, llama3:latest(8b), mistral:latest(7b), qwen2.5:14b,
# qwen3:14b, qwen3.8:27b are all >=7B.
# Select: python analyze_pooled.py [all|subgroup|size]
import sys
_MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
ONLY_CAPABLE = _MODE in ("subgroup", "capable")
ONLY_SIZE = _MODE == "size"
SMALL = ("gemma2:2b",)  # the only <7B served checkpoint
if ONLY_SIZE:
    STRATA = "sensitivity: all >=7B checkpoints (size-only criterion, no outcome filter)"
elif ONLY_CAPABLE:
    STRATA = "subgroup: >=7B checkpoints with b>c at t0 (outcome-filtered, legacy)"
else:
    STRATA = "all served models (primary)"


def load_all():
    rows = []
    for fn in sorted(glob.glob(os.path.join(HERE, "lb_danger_scored_*.jsonl"))):
        for l in open(fn, encoding="utf-8"):
            rows.append(json.loads(l))
    return rows


def cfg_of(r):
    return f"{r['model']}_t{r['temp']}"


def paired_by_config(rows, only_capable=None, only_size=None):
    """(cfg, base_id) -> {'length_only': row, 'm_a': row} restricted to the danger slice."""
    only_capable = ONLY_CAPABLE if only_capable is None else only_capable
    only_size = ONLY_SIZE if only_size is None else only_size
    by = collections.defaultdict(dict)
    for r in rows:
        if only_capable and r["model"] not in CAPABLE:
            continue
        if only_size and r["model"] in SMALL:   # size-only: drop every <7B checkpoint
            continue
        by[(cfg_of(r), r["base_id"])][r["arm"]] = r
    return {k: v for k, v in by.items()
            if "length_only" in v and "m_a" in v
            and v["length_only"]["distractor_max_sim"] >= THR}


def discordant(paired):
    """cfg -> [b, c];  b = LO dirty & MA clean,  c = MA dirty & LO clean."""
    disc = collections.defaultdict(lambda: [0, 0])
    for (cfg, _q), g in paired.items():
        lo, ma = g["length_only"]["sub_from_distractor"], g["m_a"]["sub_from_distractor"]
        if lo and not ma:
            disc[cfg][0] += 1
        elif ma and not lo:
            disc[cfg][1] += 1
    return disc


def percentile(xs, p):
    s = sorted(xs); k = (len(s) - 1) * p / 100; f = math.floor(k); c = math.ceil(k)
    return s[f] if f == c else s[f] * (c - k) + s[c] * (k - f)


def mh(disc):
    """Mantel-Haenszel chi2_1 over strata of discordant pairs: T = (sum(b-c))^2 / sum(b+c)."""
    sb = sum(b - c for b, c in disc.values())
    s_bc = sum(b + c for b, c in disc.values())
    T = (sb ** 2) / s_bc if s_bc else 0.0
    return sb, s_bc, T, float(stats.chi2.sf(T, 1))


def pooled_stats(mode="all"):
    """All the C4 numbers the manuscript quotes, for one pooling rule.

    mode: 'all' = pre-specified primary (every served (model,temp) cell);
          'size' = size-only sensitivity (every checkpoint >=7B, no outcome filter);
          'subgroup' = superseded outcome-filtered pool, kept for provenance only.
    Returns MH statistics (all cells and independent temp=0 strata), the question-cluster
    bootstrap risk difference with its percentile CI and P(D<=0), and the design's precision
    (per-question SD, SE, 80%-power MDE, n needed to clear the 0.03 SESOI floor).
    """
    only_capable = mode in ("subgroup", "capable")
    only_size = mode == "size"
    rows = load_all()
    paired = paired_by_config(rows, only_capable=only_capable, only_size=only_size)
    disc = discordant(paired)
    out = {"mode": mode, "strata": len(disc),
           "cells": {cfg: [b, c] for cfg, (b, c) in sorted(disc.items())}}
    Bn = sum(b for b, _ in disc.values()); Cn = sum(c for _, c in disc.values())
    out["naive_mcnemar"] = {"B": Bn, "C": Cn,
                            "p": float(stats.binomtest(Bn, Bn + Cn, .5).pvalue) if Bn + Cn else 1.0}
    sb, s_bc, T, p_mh = mh(disc)
    out["mh"] = {"strata": len(disc), "sum_diff": sb, "sum_disc": s_bc, "T": T, "p": p_mh}
    t0 = {cfg: v for cfg, v in disc.items() if cfg.endswith("_t0.0")}
    sb0, s_bc0, T0, p0 = mh(t0)
    out["mh_indep_t0"] = {"strata": len(t0), "sum_diff": sb0, "sum_disc": s_bc0, "T": T0, "p": p0}

    per_q = collections.defaultdict(list)   # base_id -> [(lo, ma), ...] across configs
    for (_cfg, q), g in paired.items():
        per_q[q].append((g["length_only"]["sub_from_distractor"], g["m_a"]["sub_from_distractor"]))
    qs = list(per_q)
    n_pairs = sum(len(v) for v in per_q.values())
    Dhat = sum(lo - ma for pairs in per_q.values() for (lo, ma) in pairs) / n_pairs
    random.seed(1234)
    boots = []
    for _ in range(4000):
        samp = [per_q[random.choice(qs)] for _ in qs]
        num = sum(lo - ma for pairs in samp for (lo, ma) in pairs)
        den = sum(len(pairs) for pairs in samp)
        boots.append(num / den)
    lo95, hi95 = percentile(boots, 2.5), percentile(boots, 97.5)
    out["bootstrap"] = {"questions": len(qs), "rows": n_pairs, "D": Dhat,
                        "ci": [lo95, hi95], "p_le_0": sum(1 for b in boots if b <= 0) / len(boots)}

    # Design precision: the cluster is the question, so the SE is set by the
    # between-question spread of the paired difference averaged within a question over configs.
    qmean = [statistics.fmean(lo - ma for (lo, ma) in per_q[q]) for q in qs]
    sd = statistics.stdev(qmean); n = len(qmean); se = sd / math.sqrt(n)
    SESOI, Z1, Z2 = 0.03, 1.96, 0.8416          # two-sided alpha=.05, 80% power
    out["precision"] = {"question_sd": sd, "se": se, "half_width_normal": Z1 * se,
                        "mde80": (Z1 + Z2) * se,
                        "n_for_lb_clearing_sesoi": (Z1 * sd / (Dhat - SESOI)) ** 2 if Dhat > SESOI else None,
                        "n_for_mde80_at_sesoi": ((Z1 + Z2) * sd / SESOI) ** 2}
    return out


def main():
    s = pooled_stats(_MODE)
    if ONLY_SIZE:                      # guard: a silently ignored mode argument would reuse the
        bad = [cfg for cfg in s["cells"] if any(m in cfg for m in SMALL)]
        assert not bad, f"size filter not applied: {bad}"     # full pool under a sensitivity label
    if ONLY_CAPABLE:
        bad = [cfg for cfg in s["cells"] if not any(m in cfg for m in CAPABLE)]
        assert not bad, f"outcome filter not applied: {bad}"
    print(f"== mode={_MODE}; stratum set: {STRATA}; "
          f"danger slice: LO kept a near-twin (max_sim>={THR:.2f}) ==")
    print("\n== per-stratum slice size and discordants ==")
    disc = s["cells"]
    for cfg in sorted(disc):
        print(f"  {cfg:26s} b={disc[cfg][0]} c={disc[cfg][1]}")
    nm = s["naive_mcnemar"]
    print(f"\n(1) naive pooled McNemar: B={nm['B']} C={nm['C']} p={nm['p']:.4g}  "
          f"[ANTI-CONSERVATIVE: ignores within-question + within-model correlation]")
    for key, label in (("mh", "all (model,temp) cells"), ("mh_indep_t0", "independent temp0 model strata")):
        d = s[key]
        print(f"(2{'b' if key == 'mh_indep_t0' else ''}) Mantel-Haenszel over {d['strata']} {label}: "
              f"sum(b-c)={d['sum_diff']} sum(b+c)={d['sum_disc']} T={d['T']:.2f} chi2_1 p={d['p']:.4g}")
    b = s["bootstrap"]; pr = s["precision"]
    print(f"(3) QUESTION-cluster bootstrap ({b['questions']} questions, {b['rows']} paired rows, {STRATA}):")
    print(f"    D_hat (LO-MA substitution diff) = {b['D']:.4f}  95% CI [{b['ci'][0]:.4f},{b['ci'][1]:.4f}]  "
          f"P(D<=0)={b['p_le_0']:.4g}  -> {'SIGNIFICANT (CI>0)' if b['ci'][0] > 0 else 'NOT significant (CI spans 0)'}")
    print(f"(4) design precision: per-question SD={pr['question_sd']:.4f} SE={pr['se']:.4f} "
          f"1.96SE={pr['half_width_normal']:.4f} 80%-power MDE={pr['mde80']:.4f} "
          f"n to put the lower bound over 0.03={pr['n_for_lb_clearing_sesoi']:.0f} "
          f"n for MDE80=0.03={pr['n_for_mde80_at_sesoi']:.0f}")


if __name__ == "__main__":
    main()


