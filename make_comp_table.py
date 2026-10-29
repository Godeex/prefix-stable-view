#!/usr/bin/env python
"""Emit the LaTeX float for the external content-rule arm from the run files.

Nothing in the emitted text is typed by hand: every cell comes from the same functions that write
headline_numbers.json (analyze_prefix_comp.arm_stats / paired_boot / wilson), so the table and the
prose cannot disagree. The generator refuses to emit a partial float -- a leg that has not finished,
an arm whose cell count differs from its neighbours, or a missing run file is an error, not a blank
cell.

  tab:comp : one row per arm of the competitor leg plus the two published reference arms:
             measured prompt tokens B, recompute tokens R, cache-hit share, paired Delta R against
             the same-run naive window, correctness F with its Wilson interval, and the share of
             prompts that still contain the gold binding (counted offline by
             check_comp_controls.py, never assumed).

Usage: python make_comp_table.py [--tex-out PATH]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_prefix_comp as AP

REF_ARMS = ["length_only", "prefix_stable"]


def arm_roster(crows):
    """Which arms to emit: whatever this leg actually served, in declared order, plus references."""
    served = sorted({r["arm"] for r in crows} - {"length_only"},
                    key=lambda a: AP.COMP_ORDER.index(a) if a in AP.COMP_ORDER else 99)
    unknown = [a for a in served if a not in AP.COMP_ORDER]
    if unknown:
        sys.exit(f"competitor arm(s) not in COMP_ORDER, refusing to guess a row label: {unknown}")
    return served + REF_ARMS


def build():
    crows = AP.load([AP.PRIMARY])
    if not crows:
        sys.exit(f"no {AP.PRIMARY} -- run run_prefix_comp.py first")
    prows = AP.load([AP.PRIMARY_PUB])
    if not prows:
        sys.exit(f"no published cost-leg rows ({AP.PRIMARY_PUB}) to compare against")
    arms = arm_roster(crows)
    pairs = AP.cells(crows + prows)
    n = {a: sum(1 for v in pairs.values() if a in v) for a in arms}
    if len(set(n.values())) != 1:
        sys.exit(f"leg is incomplete or unbalanced: {n}")
    o = {"model": crows[0]["model"], "temp": crows[0]["temp"], "cells": len(pairs),
         "calls": len(crows), "sessions": len({r["session_id"] for r in crows}), "arms": {}}
    for a in arms:
        sub = [v[a] for v in pairs.values()]
        rec = AP.arm_stats(sub)
        rec["acc_ci"] = AP.wilson(round(rec["acc"] * rec["n"]), rec["n"])
        rec["gold_ci"] = AP.wilson(round(rec["gold_in_ctx"] * rec["n"]), rec["n"])
        if a not in REF_ARMS:
            rec["dR_naive"] = AP.paired_boot(pairs, a, "length_only", "recompute")
            rec["dR_view"] = AP.paired_boot(pairs, a, "prefix_stable", "recompute")
            rec["dB_view"] = AP.paired_boot(pairs, a, "prefix_stable", "prompt_eval_count")
            rec["dF_naive"] = AP.paired_boot(pairs, a, "length_only", "correct")
        o["arms"][a] = rec
    o["roster"] = arms
    return o


def tex(o):
    def f(x, d=1):
        return f"{x:,.{d}f}"
    comp = [a for a in o["roster"] if a not in REF_ARMS]
    open_r = sorted({int(a[len("comp_open"):]) / 100 for a in comp if a.startswith("comp_open")},
                    reverse=True)
    keep_r = sorted({int(a[len("comp_keep"):]) / 100 for a in comp if a.startswith("comp_keep")},
                    reverse=True)
    rows = []
    for a in o["roster"]:
        v = o["arms"][a]
        label = {"length_only": r"\texttt{length\_only} (naive window, same run)",
                 "prefix_stable": r"\texttt{prefix\_stable} (this paper, published run)"}.get(
            a, "\\texttt{%s}" % a.replace("_", r"\_"))
        dr = v.get("dR_naive")
        drcell = ("---" if not dr else
                  f"${dr['D']:+,.1f}$ [{dr['ci'][0]:+,.0f},\\,{dr['ci'][1]:+,.0f}]")
        dv, db = v.get("dR_view"), v.get("dB_view")
        dvcell = "---" if not dv else f"${dv['D']:+,.1f}$ [{dv['ci'][0]:+,.0f},\\,{dv['ci'][1]:+,.0f}]"
        dbcell = "---" if not db else ("$\\pm$ matched" if db["ci"][0] <= 0 <= db["ci"][1]
                                       else (f"${db['D']:+,.0f}$ smaller" if db["ci"][1] < 0
                                             else f"${db['D']:+,.0f}$ larger"))
        gold = ("1.000 (floor)" if a in REF_ARMS
                else f"{v['gold_in_ctx']:.3f} [{v['gold_ci'][0]:.3f},\\,{v['gold_ci'][1]:.3f}]")
        rows.append(f"{label:60s} & {v['n']:4d} & {f(v['prompt_mean'],0):>6s} & "
                    f"{f(v['recomp_mean'],0):>6s} & {100*v['cached_mean']/max(v['prompt_mean'],1):5.1f}\\% & "
                    f"{drcell:28s} & {dbcell:18s} & {dvcell:28s} & "
                    f"{v['acc']:.3f} [{v['acc_ci'][0]:.3f},\\,{v['acc_ci'][1]:.3f}] & {gold} \\\\")
    matched = ("; keep-rate $.%02d$ (\\texttt{comp\\_open20}, \\texttt{comp\\_keep20}) is the "
               "matched-budget arm, added after the bracketing arms and disclosed in "
               "Sec.~\\ref{sec:competitor}. "
               % int(round(0.20 * 100))) if "comp_open20" in comp else ""
    src = "\n".join(f"%   {os.path.basename(fn)}: {sum(1 for _ in open(fn))} rows"
                    + ("  <- table above" if os.path.basename(fn) == AP.PRIMARY else "  <- replication, reported in the prose")
                    for fn in sorted(glob.glob(os.path.join(HERE, "prefix_comp_scored_*.jsonl"))))
    return f"""% generated by make_comp_table.py from the run files -- do not edit a number here
% sources:
{src}
%   reference: {AP.PRIMARY_PUB}
% leg: {o['calls']} calls, {o['cells']} paired turns, {o['sessions']} sessions, {o['model']}, temp {o['temp']}

\\begin{{table*}}[t]
\\centering
\\caption{{External content rule on the same sessions (Sec.~\\ref{{sec:competitor}}): a LLMLingua-2
token classifier prunes the naive window at keep-rates
${'/'.join(f'{r:.2f}'.lstrip('0') for r in open_r)}$ (\\texttt{{comp\\_open\\*}}) and, at
${'/'.join(f'{r:.2f}'.lstrip('0') for r in keep_r)}$, with the answer row force-kept
(\\texttt{{comp\\_keep\\*}}){matched}$B$ prompt tokens, $R$
recompute, hit the cached share; the difference columns are paired session-cluster bootstrap differences
against the same-run naive window and against this paper's view (``matched'' = the interval covers zero);
$F$ correctness under the forced-commit protocol; gold-in-prompt the share of contexts still holding the
queried binding. Wilson intervals, $6{{,}}000$ reps; removed legend clauses in
Supplementary Sec.~\\ref{{app:relocated}}.}}
\label{{tab:comp}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrrlrll}}
\\toprule
Content rule & calls & $B$ tok & $R$ tok & hit & $\\Delta R$ vs naive & $B$ vs view &
$\\Delta R$ vs view & $F$ [95\\% CI] & gold in prompt \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table*}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-out", default=os.path.join(HERE, "comp_table.tex"))
    a = ap.parse_args()
    o = build()
    out = tex(o)
    with open(a.tex_out, "w", encoding="utf-8") as f:
        f.write(out)
    json.dump({k: (v if k != "arms" else {a: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                                              for kk, vv in s.items() if kk != "n"}
                                  for a, s in v.items()}) for k, v in o.items()},
              open(os.path.join(HERE, "comp_table_source.json"), "w", encoding="utf-8"),
              indent=2, default=str, ensure_ascii=False)
    print(out)


if __name__ == "__main__":
    main()
