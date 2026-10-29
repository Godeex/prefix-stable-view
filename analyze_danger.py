#!/usr/bin/env python
"""A3: confirmatory analysis of the conditional-benefit test on the DANGER slice.

Question: at matched token budget, on the (base,K) groups where naive length_only truncation
actually retained a semantic near-twin of gold (lo_max_sim >= threshold), does M-A's distance-aware
eviction lower contamination? This is the regime where M-A has something to prevent; the (1) whole-set
result averaged over many groups where length_only happened to keep only far distractors (no risk to
reduce), which washed the effect out.

Per model-config (family/temp): pair LO vs MA on the same group, McNemar exact test on the
discordant pairs (b = LO dirty & MA clean, c = MA dirty & LO clean). Report Wilson CI of the
contamination difference, fidelity floor (gold_present), and the sharper >=0.55 subset. Then check
cross-family / non-zero-temp sign consistency (soft point i).
"""
import json, os, glob, math, collections
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson_diff(b, c, n_pairs, z=1.96):
    # CI for p(b) - p(c) via Wilson on each then difference approximation
    def w(k, n):
        if n == 0:
            return (0.0, 0.0)
        p = k / n
        d = 1 + z * z / n
        cc = p + z * z / (2 * n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return ((cc - h) / d, (cc + h) / d)
    return w(b, n_pairs), w(c, n_pairs)


def paired(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["base_id"], r["K"])][r["arm"]] = r
    pairs = []
    for k, g in by.items():
        if "length_only" in g and "m_a" in g:
            pairs.append(g)
    return pairs


def analyze(rows, label, thr_filter=None):
    ps = paired(rows)
    if thr_filter is not None:
        ps = [p for p in ps if p["length_only"]["distractor_max_sim"] >= thr_filter]
    n = len(ps)
    lo = sum(int(p["length_only"]["contamination"]) for p in ps)
    ma = sum(int(p["m_a"]["contamination"]) for p in ps)
    b = sum(1 for p in ps if p["length_only"]["contamination"] and not p["m_a"]["contamination"])
    c = sum(1 for p in ps if not p["length_only"]["contamination"] and p["m_a"]["contamination"])
    gp_lo = np.mean([p["length_only"]["gold_present"] for p in ps]) if ps else float("nan")
    gp_ma = np.mean([p["m_a"]["gold_present"] for p in ps]) if ps else float("nan")
    pval = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    (lo_lo, lo_hi), (ma_lo, ma_hi) = wilson_diff(lo, ma, n)
    lines = [f"### {label}  (n_paired={n})",
             f"- contamination: length_only={lo/n if n else 0:.3f}[{lo_lo:.3f},{lo_hi:.3f}]  "
             f"m_a={ma/n if n else 0:.3f}[{ma_lo:.3f},{ma_hi:.3f}]",
             f"- McNemar discordant: b(LO dirty,MA clean)={b}  c(MA dirty,LO clean)={c}  exact p={pval:.4g}",
             f"- fidelity floor gold_present: LO={gp_lo:.3f}  m_a={gp_ma:.3f}",
             f"- one-sided (M-A better) signal: {'YES b>c' if b > c else ('tie' if b == c else 'NO c>b')}"]
    return lines, (b, c, pval, lo, ma, n)


def load(fn):
    return [json.loads(l) for l in open(fn, encoding="utf-8")]


def main():
    files = sorted(glob.glob(os.path.join(HERE, "danger_scored_*.jsonl")))
    if not files:
        print("no danger_scored_*.jsonl yet"); return
    scan = json.load(open(os.path.join(HERE, "danger_prevalence.json")))
    lines = ["# M-A conditional-benefit test on the DANGER slice (A)", "",
             f"danger prevalence (naive length_only retains near-twin, lo_max_sim>=t): "
             f"{scan['prevalence_lo_max_sim_ge']} of {scan['n_groups']} groups",
             f"threshold used={scan['danger_threshold_used']}; matched-token char spread reported by generator.", ""]
    sign = []
    for fn in files:
        rows = load(fn)
        r0 = rows[0]
        cfg = f"{r0['model']} temp={r0['temp']} seed={r0['seed']}"
        lines.append(f"## {cfg}")
        full, stat_full = analyze(rows, "all danger groups (>=0.52)")
        lines += full
        sharp, stat_sharp = analyze(rows, "high-danger subset (>=0.55)", thr_filter=0.55)
        lines += sharp + [""]
        b, c, pval, lo, ma, n = stat_sharp if stat_sharp[5] >= 20 else stat_full
        sign.append((cfg, stat_full[0] > stat_full[1], stat_full[0], stat_full[1], stat_full[2]))
    # cross-family / temp consistency (full slice)
    consistent = all(s[1] for s in sign)
    sig_any = any(s[4] < 0.05 for s in sign)
    lines += ["## cross-config verdict",
              f"- M-A direction (b>c) in all configs: {consistent}",
              "- per-config b>c (full danger slice): " +
              "; ".join(f"{cfg}: b={b}/c={c} p={p:.3g}" for cfg, _, b, c, p in sign),
              f"- at least one config individually significant (p<0.05): {sig_any}"]
    rep = "\n".join(lines)
    print(rep)
    open(os.path.join(HERE, "danger_report.md"), "w", encoding="utf-8").write(rep)


if __name__ == "__main__":
    main()
