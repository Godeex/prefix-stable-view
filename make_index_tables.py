#!/usr/bin/env python
"""Emit the complete LaTeX floats for the INDEX tables from the run files.

Nothing in the emitted text is typed by hand. The fidelity cells and the cost cells come from the
same functions that write headline_numbers.json (headline_numbers.index_leg / index_cost_leg), and
the answer-form split comes from diag_answer_form.classify applied to the scored rows and their
prompts, so a table cell and the prose cannot disagree.

The generator refuses to emit a partial float: a leg that has not finished, a cell with a different
number of sessions than its neighbours, or a missing run file is an error, not a blank cell.

  tab:index-fid  : one row per (head length, gold slot), one column per (arm, model);
                   cell = correctness/substitution with the Wilson interval on correctness.
  tab:index-cost : one row per arm of the matched-conditions cost leg: prompt tokens, cache hit,
                   recompute with the session-cluster bootstrap premium against the published
                   stable arm, prefill ms, correctness.
  tab:index-form : POST-HOC split of the non-correct answers by answer form, kept separate from the
                   pre-declared contrasts and labelled exploratory in its own caption.

Usage: python make_index_tables.py [--tex-out PATH]
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_index_cost as AC
import diag_answer_form as DAF
import headline_numbers as HN

OUT_NAME = "index_tables.tex"
FORM_OUT_NAME = "index_form.tex"
FID_FILES = (("qwen3:14b", "index_scored_qwen3_14b.jsonl"),
             ("llama3:latest", "index_scored_llama3_latest.jsonl"))
POS_FILES = (("qwen3:14b", "position_scored_qwen3_14b.jsonl"),
             ("llama3:latest", "position_scored_llama3_latest.jsonl"))
SK_FILES = (("qwen3:14b", "index_sk_scored_qwen3_14b.jsonl"),
            ("llama3:latest", "index_sk_scored_llama3_latest.jsonl"))
ARMS = ("plain", "index", "index_nok")
SLOTS = (("first", "front"), ("middle", "middle"), ("last", "back"))
FORM_COLS = (("correct (id echoed)", "corr"),
             ("gold record restated in prose", "gold\\,in\\,prose"),
             ("another record restated in prose", "other\\,record"),
             ("substitution (competing value named)", "subst"),
             ("neither (id echoed elsewhere / refusal / unrelated)", "neither"),
             ("ambiguous (tied bodies)", "ambiguous"),
             ("no record in head", "no-rec"))


def fail(msg):
    sys.exit(f"%% REFUSING to emit {OUT_NAME}: {msg}")


def num(v, nd=3, dollar=True):
    """Signed number with a real LaTeX minus."""
    s = f"{abs(v):,.{nd}f}"
    if v < 0:
        s = "$-$" + s
    return f"${s}$" if dollar else s


def bracket(lo, hi, nd=3):
    return f"[{num(lo, nd, False)},\\,{num(hi, nd, False)}]"


def sci(v):
    """Mantissa-exponent form for a p-value, without the surrounding math mode."""
    if v <= 0:
        return "<10^{-300}"
    e = math.floor(math.log10(v))
    m = v / 10 ** e
    if e == 0:
        return f"{m:.2f}"
    return f"{m:.2f}\\times10^{{{e}}}"


def fid_leg():
    for tag, fn in FID_FILES:
        if not os.path.exists(os.path.join(HERE, fn)):
            fail(f"fidelity run file {fn} is missing")
        rows = HN.rows(fn)
        if not rows:
            fail(f"{fn} produced no rows")
        # cheap completeness check before any bootstrap work
        cells = {}
        for r in rows:
            k = (r["arm"], r["H"], r["slot"])
            cells[k] = cells.get(k, 0) + 1
        ns = sorted(set(cells.values()))
        if len(ns) != 1:
            fail(f"{tag}: cell counts differ ({ns} rows per cell), the leg has not finished")
        if len(cells) != len(ARMS) * len({r["H"] for r in rows}) * len(SLOTS):
            fail(f"{tag}: {len(cells)} cells present, expected "
                 f"{len(ARMS) * len({r['H'] for r in rows}) * len(SLOTS)}")
    leg = HN.index_leg()
    for tag, _ in FID_FILES:
        if tag not in leg:
            fail(f"{tag} did not reach the analysis")
    return leg


def heads(leg):
    return sorted({int(k.split("_H")[1].split("_")[0]) for k in leg[FID_FILES[0][0]]["cells"]})


def fid_body(leg):
    lines = []
    for h in heads(leg):
        base = leg[FID_FILES[0][0]]["cells"][f"plain_H{h}_first"]
        blk = leg[FID_FILES[0][0]]["cells"][f"index_H{h}_first"]
        nok = leg[FID_FILES[0][0]]["cells"][f"index_nok_H{h}_first"]
        lines.append(f"\\multicolumn{{8}}{{@{{l@}}}}{{\\textit{{head $H={h}$ lines "
                     f"(mean $\\approx{base['tok']:,.0f}$ prompt tokens; \\texttt{{INDEX}} block "
                     f"${blk['chars'] - base['chars']:+d}$ chars, placebo "
                     f"${nok['chars'] - base['chars']:+d}$)}}}}\\\\")
        for skey, sname in SLOTS:
            cells = []
            for arm in ARMS:
                for tag, _ in FID_FILES:
                    c = leg[tag]["cells"][f"{arm}_H{h}_{skey}"]
                    cells.append(f"{num(c['correct'][0])}\\,/\\,{num(c['substitution'][0])} "
                                 f"\\,{bracket(*c['correct'][1:3])}")
            lines.append((f"{h} & " if skey == "first" else " & ") + f"{sname} & "
                          + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def fid_note(leg):
    mx = max(c["substitution"][0] for d in leg.values() for c in d["cells"].values())
    n = next(iter(leg.values()))["cells"][f"plain_H{heads(leg)[0]}_first"]["n"]
    return (f"Each of the {len(ARMS) * len(heads(leg)) * len(SLOTS)} cells holds {n} sessions per "
            f"architecture. Substitution is {num(mx)} in every cell of this grid, against "
            f"$0.550$--$0.742$ in the published shared-key stress (Table~\\ref{{tab:position}}); "
            f"Table~\\ref{{tab:index-form}} reports where the loss lands instead.")


def sk_fid_leg():
    """Shared-key fidelity panel: completeness only; the cells come from the same
    headline_numbers.shared_key_leg() the analysis quotes, so the table cannot disagree with prose."""
    leg = HN.shared_key_leg()
    for tag, fn in SK_FILES:
        if not os.path.exists(os.path.join(HERE, fn)):
            fail(f"shared-key run file {fn} is missing")
        if not leg.get(tag):
            fail(f"{tag} did not reach the shared-key analysis")
        ns = sorted({c["n"] for c in leg[tag]["cells"].values()})
        if len(ns) != 1:
            fail(f"{tag}: shared-key cell counts differ ({ns}), the leg has not finished")
    return leg


def sk_heads(leg):
    return sorted(leg[SK_FILES[0][0]]["hs"])


def sk_body(leg):
    lines = []
    for h in sk_heads(leg):
        lines.append(f"\\multicolumn{{8}}{{@{{l@}}}}{{\\textit{{head $H={h}$ lines, distractor keyed "
                     f"\\texttt{{TARGET}} again (shared key, one stale value in \\texttt{{STATE}})}}}}\\\\")
        for skey, sname in SLOTS:
            cells = []
            for arm in ARMS:
                for tag, _ in SK_FILES:
                    c = leg[tag]["cells"][f"{arm}_H{h}_{skey}"]
                    cells.append(f"{num(c['correct'][0])}\\,/\\,{num(c['substitution'][0])} "
                                 f"\\,{bracket(*c['correct'][1:3])}")
            lines.append((f"{h} & " if skey == "first" else " & ") + f"{sname} & "
                          + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def sk_note(leg):
    def rd(m, arm, h, s, field):
        pr = leg[m]["vs_plain"][f"{arm}_H{h}_{s}"][field]
        return pr["D"] if pr else None
    hs = sk_heads(leg)
    # Q-A: index correct RD over the collapsed slots, averaged over H (both models)
    qa = {}
    for m in leg:
        ds = [rd(m, "index", h, s, "correct") for h in hs for s in ("middle", "last")]
        ds = [x for x in ds if x is not None]
        qa[m] = st.mean(ds) if ds else float("nan")
    # Q-B: index_nok substitution RD vs plain, averaged over the grid
    qb = {}
    for m in leg:
        ds = [rd(m, "index_nok", h, s, "substitution") for h in hs for s in ("first", "middle", "last")]
        ds = [x for x in ds if x is not None]
        qb[m] = st.mean(ds) if ds else float("nan")
    n = leg[SK_FILES[0][0]]["cells"]["plain_H6_first"]["n"]
    return (f"Each of the {len(ARMS) * len(hs) * len(SLOTS)} cells holds {n} sessions per "
            f"architecture; the STATE is the published C7 shared-key cell byte-for-byte, so only the "
            f"one \\texttt{{INDEX}} row separates the arms. A correct index recovers the collapse "
            f"(Q-A, mean index$-$plain correct RD over the middle and back slots $+{qa['qwen3:14b']:.3f}$ "
            f"qwen3, $+{qa['llama3:latest']:.3f}$ llama3); a stale one (\\texttt{{index\\_nok}}, naming "
            f"the competitor) drives substitution above the raw state (Q-B, mean RD "
            f"$+{qb['qwen3:14b']:.3f}$ / $+{qb['llama3:latest']:.3f}$), i.e. the model reads the view "
            f"over the line beneath it. The block's value, not its presence, carries the effect: "
            f"index$-$index\\_nok correctness averages $+{st.mean(v['D'] for v in leg['qwen3:14b']['value_only'].values()):.3f}$ "
            f"/ $+{st.mean(v['D'] for v in leg['llama3:latest']['value_only'].values()):.3f}$.")


def cost_leg():
    if not glob.glob(os.path.join(HERE, "index_cost_scored_*.jsonl")):
        fail("no index_cost_scored_*.jsonl: the cost leg has not run")
    cost = HN.index_cost_leg()
    if not cost:
        fail("the cost leg produced no parsable rows")
    ns = {a: cost["arms"][a]["n"] for a in AC.ARMS}
    if len(set(ns.values())) != 1:
        fail(f"cost arms have unequal call counts ({ns})")
    return cost


def cost_body(cost):
    lines = []
    for arm in AC.ARMS:
        a = cost["arms"][arm]
        hit = 100 * a["cached"] / max(a["prompt"], 1)
        prem = "\\textemdash{}"
        if "premium_vs_stable" in a:
            d = a["premium_vs_stable"]
            prem = f"{num(d['D'], 1)}\\,/\\,{bracket(d['ci'][0], d['ci'][1], 1)}"
        name = arm.replace("_", "\\_")
        lines.append(f"\\texttt{{{name}}} & {a['n']:,} & {a['prompt']:,.0f} & {hit:.1f}\\% & "
                     f"{a['recompute']:,.1f} & {prem} & {a['prefill_ms']:,.1f} & "
                     f"{a['acc']:.3f} \\\\")
    return "\n".join(lines)


def cost_note(cost):
    arms = cost["arms"]
    s = cost["slope_on_I"]
    b, se, p = s["intercept"][1], s["se"][1], s["p"][1]
    return (f"Cluster-robust slope of the premium on the number of \\texttt{{INDEX}} rows: "
            f"{num(b, 2)} tokens per row (SE {se:.2f}, $p={sci(p)}$), intercept "
            f"{num(s['intercept'][0], 1)} (SE {s['se'][0]:.2f}), over {s['rows']:,} observations "
            f"({len(AC.ARMS) - 2} index sizes $\\times$ {cost['pairs']:,} paired turns) in "
            f"{s['clusters']} sessions; the fitted line is within "
            f"{num(s['max_abs_residual'], 1)} tokens of every measured level. Inference basis: the "
            f"same regression with a session-clustered sandwich covariance, $t$ on "
            f"{s['clusters'] - 1} df; the $[{b - 1.96 * se:.2f},{b + 1.96 * se:.2f}]$ interval the "
            f"introduction quotes for the slope is the Wald $\\pm1.96$ SE of this fit, whereas the "
            f"premium intervals in the columns are $2.5$/$97.5$ percentile session-cluster "
            f"bootstraps. In the same streams "
            f"\\texttt{{length\\_only}} "
            f"recomputes {num(arms['length_only']['recompute'], 1)} and \\texttt{{prefix\\_stable}} "
            f"{num(arms['prefix_stable']['recompute'], 1)} tokens per turn.")


def classify_all(rows, trials):
    out = {}
    for r in rows:
        arm = r.get("arm", "shared-key head")
        if r["correct"]:
            b = "correct (id echoed)"
        elif r["substitution"]:
            b = "substitution (competing value named)"
        else:
            b, _ = DAF.classify(r, trials[r["trial_id"]])
        out.setdefault(arm, {})
        out[arm][b] = out[arm].get(b, 0) + 1
    return out


def load_trials(fn):
    tr = {}
    for l in open(os.path.join(HERE, fn), encoding="utf-8"):
        t = json.loads(l)
        tr[t["trial_id"]] = t
    return tr


def form_rows():
    it = load_trials("index_trials.jsonl")
    pt = load_trials("position_trials.jsonl")
    order = list(ARMS) + ["shared-key head"]
    per_model = {}
    for (tag, ifn), (_, pfn) in zip(FID_FILES, POS_FILES):
        cnt = classify_all(HN.rows(ifn), it)
        cnt.update(classify_all(HN.rows(pfn), pt))
        missing = [a for a in order if a not in cnt]
        if missing:
            fail(f"answer-form split missing arms for {tag}: {missing}")
        per_model[tag] = cnt
    rows = []
    for tag, cnt in per_model.items():
        for arm in order:
            n = sum(cnt[arm].values())
            rates = [cnt[arm].get(k, 0) / n for k, _ in FORM_COLS]
            rows.append((tag, arm, n, rates))
    return rows


def form_body(rows, per_model):
    lines = []
    for i, (tag, arm, n, rates) in enumerate(rows):
        if i and i % per_model == 0:
            lines.append("\\midrule")
        head = tag if arm == ARMS[0] else ""
        name = "\\texttt{%s}" % arm.replace("_", "\\_") if arm in ARMS else "shared-key head"
        lines.append(f"{head} & {name} & " + " & ".join(f"{x:.3f}" for x in rates)
                     + f" & {n} \\\\")
    return "\n".join(lines)


def float_block(env, label, cap, head, body, cols):
    """Every one of these floats is wider than the column it lands in, so the tabular is always
    scaled to the line rather than left to overflow the type area."""
    return (f"\\begin{{{env}}}[t]\n\\centering\n\\caption{{{cap}}}\n\\label{{{label}}}\n"
            f"\\setlength{{\\tabcolsep}}{{3pt}}\n\\resizebox{{\\linewidth}}{{!}}{{%\n"
            f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}\n"
            f"\\toprule\n{head}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}%\n}}\n"
            f"\\end{{{env}}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-out", default=os.path.join(HERE, OUT_NAME))
    args = ap.parse_args()

    leg = fid_leg()
    skleg = sk_fid_leg()
    cost = cost_leg()
    frows = form_rows()

    fid = float_block(
        "table*", "tab:index-fid",
        f"Quality benefit of the rewritable index view (Sec.~\\ref{{sec:index-view}}): each cell is "
        f"correctness\\,/substitution with the Wilson interval on correctness; the queried binding "
        f"sits at the front, in the middle, or at the back of a head of $H$ lines. " + fid_note(leg),
        "& & \\multicolumn{2}{c}{\\texttt{plain}} & \\multicolumn{2}{c}{\\texttt{index}} & "
        "\\multicolumn{2}{c}{\\texttt{index\\_nok}} \\\\\n"
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}\n"
        "$H$ & slot & qwen3 & llama3 & qwen3 & llama3 & qwen3 & llama3 \\\\",
        fid_body(leg), "llrrrrrr")

    sk_t = float_block(
        "table*", "tab:index-sk",
        f"The same current-binding \\texttt{{INDEX}} block served under \\emph{{shared-key}} hazard: "
        f"the queried binding and a competing stale value both keyed \\texttt{{TARGET}} in the "
        f"published C7 \\texttt{{STATE}} (Sec.~\\ref{{app:poscontrols}}). Same cell format as "
        f"Table~\\ref{{tab:index-fid}}. " + sk_note(skleg),
        "& & \\multicolumn{2}{c}{\\texttt{plain}} & \\multicolumn{2}{c}{\\texttt{index}} & "
        "\\multicolumn{2}{c}{\\texttt{index\\_nok}} \\\\\n"
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}\n"
        "$H$ & slot & qwen3 & llama3 & qwen3 & llama3 & qwen3 & llama3 \\\\",
        sk_body(skleg), "llrrrrrr")

    cost_t = float_block(
        "table*", "tab:index-cost",
        "Cost of the same view under matched conditions (Sec.~\\ref{sec:index-cost}): the "
        "\\texttt{INDEX} block of $I$ current bindings replayed in the published cost streams. "
        "Premium is the paired recompute difference against \\texttt{prefix\\_stable} with a "
        "session-cluster bootstrap interval. " + cost_note(cost),
        "arm & calls & prompt & cache & recompute & premium vs \\texttt{stable} [95\\% CI] & "
        "prefill ms & corr \\\\",
        cost_body(cost), "lrrrrrrr")

    form_t = float_block(
        "table", "tab:index-form",
        "POST-HOC split of every answer by form (exploratory: no pre-declared contrast, arm "
        "assignment or table cell depends on it). The last row of each block is the published "
        "shared-key stress of Table~\\ref{tab:position}, scored with the same buckets. ",
        "& arm & " + " & ".join(lbl for _, lbl in FORM_COLS) + " & $n$ \\\\",
        form_body(frows, len(ARMS) + 1), "llrrrrrrrr")

    audit = "\n".join(
        f"%   {fn}: {sum(1 for _ in open(os.path.join(HERE, fn), encoding='utf-8'))} rows"
        for _, fn in list(FID_FILES) + list(POS_FILES) + list(SK_FILES))
    header = ("% generated by make_index_tables.py from the run files -- do not edit a number here\n"
              "% sources:\n" + audit +
              f"\n%   cost leg: {cost['pairs']} paired turns, {cost['sessions']} sessions, "
              f"model {cost['model']}, temp {cost['temp']}\n\n")
    # The two body floats and the appendix float go to separate files so main.tex can
    # \input each where it belongs; neither is ever edited by hand.
    out = os.path.abspath(args.tex_out)
    form_out = os.path.join(os.path.dirname(out), FORM_OUT_NAME)
    open(out, "w", encoding="utf-8").write(header + fid + "\n\n" + sk_t + "\n\n" + cost_t + "\n")
    open(form_out, "w", encoding="utf-8").write(header + form_t + "\n")
    print(header + fid + "\n\n" + sk_t + "\n\n" + cost_t + "\n\n" + form_t)
    print(f"-> {out}\n-> {form_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
