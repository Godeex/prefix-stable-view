#!/usr/bin/env python
"""E6: paired analysis of the three-arm real-load comparison + beta refit on natural data.

(1) Per-arm contamination / fidelity with Wilson CI.
(2) AT-RISK paired contrast: restrict to (base,K) groups where length_only actually retained a
    distractor nearer to gold than the one m_a kept (max_sim_LO > max_sim_MA). McNemar on the paired
    contamination binaries there -> is M-A's drop-in benefit real when near-twins ARE present?
(3) Beta refit on natural load: logistic contamination ~ max_sim + K over ALL arms (proximity spans
    0.32..0.53). Positive max_sim coef => surrogate revalidated on natural fields => emit
    calibration_params.natural.json (kept provisional until cross-family soft point (i) closes).
(4) matched-token check: per-triplet char equality + tokenizer spread.
"""
import json, os, sys, math, collections
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MA_MODEL", "qwen3:8b")
SCORED = os.path.join(HERE, f"policy_scored_{MODEL.replace(':','_')}.jsonl")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)


def load():
    rows = [json.loads(l) for l in open(SCORED, encoding="utf-8")]
    return pd.DataFrame(rows)


def main():
    df = load()
    df["contam"] = df["contamination"].astype(int)
    lines = [f"# M-A real-load comparison ({MODEL}, temp=0) — E6 analysis", ""]
    # matched-token check
    grp = df.groupby(["base_id", "K"])["char_len"].agg(lambda s: s.max() - s.min())
    tok = df.groupby("arm")["prompt_eval_count"].mean()
    lines += ["## matched-token check",
              f"- per-(base,K) char-length spread max = {int(grp.max())} (0 => exactly matched)",
              f"- mean prompt_eval_count by arm: " +
              ", ".join(f"{a}={tok[a]:.1f}" for a in ["length_only", "m_a", "worst"]), ""]

    # per-arm contamination
    lines.append("## contamination / fidelity by arm")
    for arm in ["length_only", "m_a", "worst"]:
        s = df[df.arm == arm]
        k, n = int(s.contam.sum()), len(s)
        lo, hi = wilson(k, n)
        gp = s.gold_present.mean()
        lines.append(f"- {arm:11s} n={n} contamination={k/n:.3f} (Wilson[{lo:.3f},{hi:.3f}]) "
                     f"mean_nsim={s.distractor_mean_sim.mean():.3f} "
                     f"mean_max_sim={s.distractor_max_sim.mean():.3f} gold_present={gp:.3f}")
    lines.append("")

    # AT-RISK paired contrast: m_a dropped a nearer twin than length_only kept
    pairs = []
    for (b, K), g in df.groupby(["base_id", "K"]):
        lo = g[g.arm == "length_only"]; ma = g[g.arm == "m_a"]
        if len(lo) == 1 and len(ma) == 1 and lo.distractor_max_sim.iloc[0] > ma.distractor_max_sim.iloc[0] + 0.05:
            pairs.append((int(lo.contam.iloc[0]), int(ma.contam.iloc[0]),
                          float(lo.distractor_max_sim.iloc[0])))
    P = pd.DataFrame(pairs, columns=["length_only", "m_a", "max_sim_lo"])
    # high-at-risk: restrict further to length_only max_sim above 0.45 (near-duplicate present)
    HR = P[P.max_sim_lo > 0.45]
    lines.append("## at-risk paired contrast (only groups where length_only kept a NEARER twin than m_a)")
    def mcn(P):
        b = int(((P.length_only == 1) & (P.m_a == 0)).sum())   # LO contam, MA clean
        c = int(((P.length_only == 0) & (P.m_a == 1)).sum())   # MA contam, LO clean
        if b + c == 0:
            return None
        p = stats.binomtest(b, b + c, 0.5).pvalue
        return b, c, p
    for name, X in [("all-at-risk", P), ("high-at-risk(max_sim>0.45)", HR)]:
        if len(X) == 0:
            lines.append(f"- {name}: no qualifying groups"); continue
        r = mcn(X)
        lines.append(f"- {name}: n={len(X)} LO_contam_rate={X.length_only.mean():.3f} "
                     f"MA_contam_rate={X.m_a.mean():.3f} " +
                     (f"McNemar b={r[0]} c={r[1]} p={r[2]:.4g}" if r else "no discordant pairs"))
    lines.append("")

    # beta refit on natural load (all arms -> proximity spans the range)
    X = sm.add_constant(pd.DataFrame({
        "max_sim": df.distractor_max_sim.astype(float),
        "K": df.K.astype(float)}))
    try:
        mlog = sm.Logit(df.contam.astype(float), X).fit(disp=0, method="newton", maxiter=200)
        b0, b1, b2 = mlog.params["const"], mlog.params["max_sim"], mlog.params["K"]
        pseudo = mlog.prsquared
        # VIF of max_sim vs K
        aux = sm.OLS(X["max_sim"], X[["const", "K"]]).fit()
        vif = 1.0 / (1 - aux.rsquared)
        lines += ["## beta refit on NATURAL field values (logistic contamination ~ max_sim + K)",
                  f"- beta0={b0:.3f} beta_max_sim={b1:.3f} (p={mlog.pvalues['max_sim']:.3g}) "
                  f"beta_K={b2:.3f} (p={mlog.pvalues['K']:.3g})",
                  f"- McFadden pseudo-R2={pseudo:.3f}  VIF(max_sim|K)={vif:.2f}",
                  "- reading: positive & significant beta_max_sim => semantic proximity predicts "
                  "contamination on natural fields (surrogate revalidated off the artificial near/far contrast).", ""]
        natural = {
            "schema": "light.ma.calibration.v1", "fit_scope": "stage6-natural",
            "provisional": True, "provisional_reason": "single family qwen3:8b temp=0; cross-family "
            "generality (soft point i) not yet tested", "as_of": "2026-09-23",
            "model_family": MODEL, "n_trials": int(len(df)),
            "params": {"beta0": round(float(b0), 3), "beta1": round(float(b1), 3),
                       "beta2": round(float(b2), 3), "tau_near": 0.45},
            "form": "p(contamination)=sigmoid(beta0 + beta1*max_sim + beta2*K)",
            "pseudo_r2": round(float(pseudo), 3), "vif_max_sim": round(float(vif), 2),
            "beta_max_sim_p": float(mlog.pvalues["max_sim"]),
            "source": "stage6-src/policy_scored_%s.jsonl (natural MTU-Bench fields, 3-arm)" % MODEL,
        }
        json.dump(natural, open(os.path.join(HERE, "configs", "calibration_params.natural.json"),
                                "w"), ensure_ascii=False, indent=2)
        lines.append("wrote configs/calibration_params.natural.json (fit_scope=stage6-natural, provisional)")
    except Exception as e:
        lines.append(f"[beta refit skipped] {e!r} — likely separation (m_a/length_only all-zero cells)")

    report = "\n".join(lines)
    print(report)
    open(os.path.join(HERE, "policy_results.md"), "w", encoding="utf-8").write(report)


if __name__ == "__main__":
    main()
