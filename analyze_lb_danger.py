#!/usr/bin/env python
"""B3: confirmatory analysis of M-A vs naive truncation on the passage DANGER slice.

Primary outcome on this load = substitution (answer wrong AND the answer span appears only in a
shown distractor passage = the model grabbed a competing entity from an adjacent-but-wrong passage).
Paired per question: length_only (kept a near-adversary by position) vs m_a (evicted it).
McNemar exact on discordant pairs (b = LO substitutes, MA clean; c = MA substitutes, LO clean).
Report the >=threshold danger slice and a sharper subset, Wilson CI, fidelity (correct/gold rate),
then cross-family + non-zero-temp sign consistency. This is the make-or-break for the M-A systems
claim: if even here M-A shows no benefit over naive truncation, the claim should be honestly downgraded.
"""
import json, os, glob, math, collections
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def paired(rows, key="sub_from_distractor"):
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["base_id"]][r["arm"]] = r
    return [g for g in by.values() if "length_only" in g and "m_a" in g]


def analyze(rows, thr, label):
    ps = paired(rows)
    if thr is not None:
        ps = [p for p in ps if p["length_only"]["distractor_max_sim"] >= thr]
    n = len(ps)
    lo = sum(int(p["length_only"]["sub_from_distractor"]) for p in ps)
    ma = sum(int(p["m_a"]["sub_from_distractor"]) for p in ps)
    b = sum(1 for p in ps if p["length_only"]["sub_from_distractor"] and not p["m_a"]["sub_from_distractor"])
    c = sum(1 for p in ps if not p["length_only"]["sub_from_distractor"] and p["m_a"]["sub_from_distractor"])
    cor_lo = np.mean([p["length_only"]["correct"] for p in ps]) if ps else float("nan")
    cor_ma = np.mean([p["m_a"]["correct"] for p in ps]) if ps else float("nan")
    pv = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    lolo, lohi = wilson(lo, n)
    mal, mah = wilson(ma, n)
    lines = [f"### {label} (n_paired={n})",
             f"- substitution: length_only={lo/max(n,1):.3f}[{lolo:.3f},{lohi:.3f}]  "
             f"m_a={ma/max(n,1):.3f}[{mal:.3f},{mah:.3f}]",
             f"- McNemar: b(LO subst,MA clean)={b}  c(MA subst,LO clean)={c}  exact p={pv:.4g}",
             f"- correct rate: LO={cor_lo:.3f}  m_a={cor_ma:.3f}",
             f"- M-A better signal: {'YES b>c' if b > c else ('tie' if b == c else 'NO c>b')}"]
    return lines, (b, c, pv, n)


def main():
    files = sorted(glob.glob(os.path.join(HERE, "lb_danger_scored_*.jsonl")))
    if not files:
        print("no lb_danger_scored_*.jsonl yet"); return
    scan = json.load(open(os.path.join(HERE, "lb_danger_prevalence.json")))
    thr = scan["danger_threshold_used"]
    lines = ["# M-A vs naive truncation on the PASSAGE danger slice (①)", "",
             f"usable questions={scan['n_questions']} K={scan['K']} trunc={scan['trunc']}",
             f"naive positional truncation retains near-adversary passage (lo_max>=t): "
             f"{scan['prevalence_lo_max_ge']}", f"threshold used={thr}", ""]
    sign = []
    for fn in files:
        rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
        r0 = rows[0]
        cfg = f"{r0.get('model','?')} temp={r0.get('temp','?')} seed={r0.get('seed','?')}"
        lines.append(f"## {cfg}")
        full, sf = analyze(rows, thr, f"danger slice (lo_max>={thr})")
        lines += full
        sharp, ss = analyze(rows, 0.70, "high-danger subset (lo_max>=0.70)")
        lines += sharp + [""]
        sign.append((cfg, sf[0] > sf[1], sf[0], sf[1], sf[2], sf[3]))
    consistent = all(s[1] for s in sign)
    any_sig = any(s[4] < 0.05 for s in sign)
    lines += ["## cross-config verdict",
              f"- M-A direction (b>c) in all configs: {consistent}",
              "- per-config (danger slice): " + "; ".join(
                  f"{cfg}: b={b}/c={c} p={p:.3g} n={n}" for cfg, _, b, c, p, n in sign),
              f"- at least one config individually significant (p<0.05): {any_sig}"]
    rep = "\n".join(lines)
    print(rep)
    open(os.path.join(HERE, "lb_danger_report.md"), "w", encoding="utf-8").write(rep)


if __name__ == "__main__":
    main()
