#!/usr/bin/env python
"""(1-hardening) multi-seed robustness for M-A passage danger-slice benefit.

Two independent-ish strata per capable model: (temp=0, seed=1234) [main] and (temp=0.6, seed=777)
[resample]. danger slice THR>=0.55 (n=74). Checks: (a) does b>c hold in the seed777 resample for every
model; (b) MH pooled over all 8 (model x config) strata; (c) question-cluster bootstrap of D=LO-MA
pooled over both seeds (clusters by question so within-question seed/model rows travel together).
"""
import os, collections, json
import numpy as np
from scipy import stats
import analyze_pooled as P
import analyze_sweep as S

THR = 0.55
MODELS = ("qwen3:14b", "qwen2.5:14b", "llama3:latest", "mistral:latest")
XA = ("gemma2:2b",)   # cross-architecture probe (different attention design + vendor)
CFGS = {  # label -> cfg string; P.cfg_of builds f"{model}_t{temp}" from the raw model field (colon kept)
    "t0/s1234": lambda m: f"{m}_t0.0",
    "t0.6/s777": lambda m: f"{m}_t0.6",
}


def main():
    rows = P.load_all()
    paired = {k: v for k, v in S.pair_all(rows).items()
              if v["length_only"]["distractor_max_sim"] >= THR
              and v["length_only"]["model"] in MODELS}
    per_q = collections.defaultdict(list)
    disc = collections.defaultdict(lambda: [0, 0])
    for (cfg, q), g in paired.items():
        lo, ma = g["length_only"]["sub_from_distractor"], g["m_a"]["sub_from_distractor"]
        per_q[q].append((lo, ma))
        if lo and not ma: disc[cfg][0] += 1
        elif ma and not lo: disc[cfg][1] += 1

    print(f"danger slice THR>=%.2f (n=%d questions), per-model discordants:" % (THR, len(per_q)))
    for m in MODELS:
        line = [f"  {m:14s}"]
        for lab, fn in CFGS.items():
            b, c = disc.get(fn(m), [0, 0])
            pv = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
            line.append(f"{lab}: b={b}/c={c} p={pv:.3f} {'b>c' if b>c else 'X'}")
        print("  ".join(line))

    # (b) MH over all 8 (model x config) strata
    strata = [(cfg, tuple(v)) for cfg, v in disc.items() if any(cfg == fn(m) for m in MODELS for fn in CFGS.values())]
    sb = sum(b - c for _, (b, c) in strata); s_bc = sum(b + c for _, (b, c) in strata)
    T = (sb ** 2) / s_bc if s_bc else 0.0
    p_mh = float(stats.chi2.sf(T, 1))
    print(f"\nMH over {len(strata)} strata (4 models x 2 configs): sum(b-c)={sb} sum(b+c)={s_bc} T={T:.2f} p={p_mh:.2e}")

    # --- sensitivity: add the cross-architecture gemma2:2b strata (still b>c but small effect) ---
    paired_x = {k: v for k, v in S.pair_all(rows).items()
                if v["length_only"]["distractor_max_sim"] >= THR
                and v["length_only"]["model"] in MODELS + XA}
    disc_x = collections.defaultdict(lambda: [0, 0])
    for (cfg, q), g in paired_x.items():
        lo, ma = g["length_only"]["sub_from_distractor"], g["m_a"]["sub_from_distractor"]
        if lo and not ma: disc_x[cfg][0] += 1
        elif ma and not lo: disc_x[cfg][1] += 1
    strata_x = [(c, tuple(v)) for c, v in disc_x.items()
                if any(c == fn(m) for m in MODELS + XA for fn in CFGS.values())]
    sbx = sum(b - c for _, (b, c) in strata_x); sbcx = sum(b + c for _, (b, c) in strata_x)
    Tx = (sbx ** 2) / sbcx if sbcx else 0.0
    print(f"MH + cross-arch gemma2 (10 strata): sum(b-c)={sbx} sum(b+c)={sbcx} T={Tx:.2f} "
          f"p={float(stats.chi2.sf(Tx,1)):.2e}  (gemma2 strata: "
          + ", ".join(f"{c}={tuple(disc_x[c])}" for c in sorted(disc_x) if any(c == fn(m) for m in XA for fn in CFGS.values())) + ")")

    # (c) question-cluster bootstrap of D over both seeds
    qs = list(per_q); npr = sum(len(v) for v in per_q.values())
    Dhat = sum(lo - ma for q in qs for lo, ma in per_q[q]) / npr
    rng = np.random.default_rng(99); idx = np.arange(len(qs)); boots = []
    for _ in range(6000):
        pick = rng.choice(idx, size=len(qs), replace=True)
        num = den = 0
        for i in pick:
            for lo, ma in per_q[qs[i]]: num += lo - ma; den += 1
        boots.append(num / den if den else 0.0)
    lo95, hi95 = np.percentile(boots, [2.5, 97.5]); p_boot = np.mean(np.array(boots) <= 0)
    print(f"cluster bootstrap ({len(qs)} q x {npr} rows both seeds): D={Dhat:.4f} "
          f"95%CI[{lo95:+.3f},{hi95:+.3f}] P(D<=0)={p_boot:.2e} -> {'SIG' if lo95>0 else 'ns'}")


if __name__ == "__main__":
    main()
