#!/usr/bin/env python
"""(1-hardening) THR sensitivity sweep for M-A passage danger-slice benefit.

Reuses analyze_pooled's load/pair/discordant/bootstraps but sweeps the danger threshold and, for the
expanded low-THR pool, keeps only FULL-COVERAGE models (>=100 length_only questions) so 27B (danger-
slice-only, 44 q) doesn't silently drop questions. Reports MH (over independent temp=0 model strata)
and the question-cluster bootstrap CI of D=mean(LO-MA) at each threshold.
"""
import os, sys, collections
import numpy as np
from scipy import stats
import analyze_pooled as P

FULL_MIN_Q = 100   # a model must cover >= this many length_only questions to join the expanded pool


def full_coverage_models(rows):
    cnt = collections.defaultdict(set)
    for r in rows:
        if r["arm"] == "length_only":
            cnt[r["model"]].add(r["base_id"])
    return {m for m, s in cnt.items() if len(s) >= FULL_MIN_Q}


def pair_all(rows):
    """(cfg, base_id) -> {arm: row} for every question BOTH arms of a capable model cover.
    No THR filter here (analyze_pooled.paired_by_config hard-filters at module THR=0.6, which
    would hide the 0.55-0.60 band); the caller applies the threshold."""
    by = collections.defaultdict(dict)
    for r in rows:
        by[(P.cfg_of(r), r["base_id"])][r["arm"]] = r
    return {k: v for k, v in by.items() if "length_only" in v and "m_a" in v}


def pooled_at(thr, paired, cap_models, boot=6000):
    paired = {k: v for k, v in paired.items()
              if v["length_only"]["distractor_max_sim"] >= thr
              and v["length_only"]["model"] in cap_models}
    # per-config discordants
    disc = collections.defaultdict(lambda: [0, 0])
    per_q = collections.defaultdict(list)
    for (cfg, q), g in paired.items():
        lo, ma = g["length_only"]["sub_from_distractor"], g["m_a"]["sub_from_distractor"]
        if lo and not ma:
            disc[cfg][0] += 1
        elif ma and not lo:
            disc[cfg][1] += 1
        per_q[q].append((lo, ma))
    t0 = {c: (b, cc) for c, (b, cc) in disc.items() if c.endswith("_t0.0")}
    sb = sum(b - cc for b, cc in t0.values()); s_bc = sum(b + cc for b, cc in t0.values())
    T = (sb ** 2) / s_bc if s_bc else 0.0
    p_mh = float(stats.chi2.sf(T, 1))
    # question-cluster bootstrap of D
    qs = list(per_q); npr = sum(len(v) for v in per_q.values())
    Dhat = sum(lo - ma for q in qs for lo, ma in per_q[q]) / max(npr, 1)
    rng = np.random.default_rng(1234)
    idx = np.arange(len(qs))
    boots = []
    for _ in range(boot):
        pick = rng.choice(idx, size=len(qs), replace=True)
        num = den = 0
        for i in pick:
            for lo, ma in per_q[qs[i]]:
                num += lo - ma; den += 1
        boots.append(num / den if den else 0.0)
    lo95, hi95 = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    p_boot = float(np.mean(np.array(boots) <= 0))
    return dict(n_models=len(t0), n_q=len(qs), n_pairs=npr, sum_bc=sb, T=T, p_mh=p_mh,
                Dhat=Dhat, lo95=lo95, hi95=hi95, p_boot=p_boot,
                per_cfg={c: tuple(v) for c, v in sorted(t0.items())})


def main():
    rows = P.load_all()
    full = full_coverage_models(rows)
    print("full-coverage models (>=100 lo q):", sorted(full))
    print("capable (b>c) subset used:", sorted(full & set(P.CAPABLE) - {"qwen3:8b"}),
          " (qwen3:8b is the tie/wash -> excluded from primary)\n")
    print(f"{'THR':>5} {'mod':>4} {'q':>4} {'pairs':>6}  MH-T   MH-p     bootCI(D=LO-MA)        boot-p")
    paired = pair_all(rows)
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
        cap = full & set(P.CAPABLE) - {"qwen3:8b"}
        # 27B already dropped by full-coverage; qwen3:14b/qwen2.5:14b/llama3/mistral remain
        r = pooled_at(thr, paired, cap)
        flag = "SIG" if (r["p_mh"] < 0.05 and r["lo95"] > 0) else "    "
        print(f"{thr:>5.2f} {r['n_models']:>4} {r['n_q']:>4} {r['n_pairs']:>6}  "
              f"{r['T']:>5.2f} {r['p_mh']:.2e}  [{r['lo95']:+.3f},{r['hi95']:+.3f}] {r['p_boot']:.3e}  {flag}")
        for c, (b, cc) in r["per_cfg"].items():
            print(f"        {c:26s} b={b} c={cc}")


if __name__ == "__main__":
    main()
