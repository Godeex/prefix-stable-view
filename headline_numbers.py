#!/usr/bin/env python
"""Recompute every headline number from the run files, with the CIs the paper quotes.

Manuscript numbers were transcribed from run reports, and transcription is where arithmetic drifts
(a mean printed as 249 when the components round to 248, a ratio quoted as 6.3x when it is 6.25,
"reproduces to within 0.1 token" for what is a 0.18 difference in ratio). This script recomputes
each such quantity directly from the per-turn records, reports it to enough digits to decide the
wording, and writes headline_numbers.json so the table generator can consume the same values the
text quotes.

Quantities:
  cost run   : per-arm means (prompt / cached / recompute / prefill ms) on both models, paired
               difference and RATIO with session-cluster bootstrap CIs, and how much of the
               naive-to-stable gap the size-matched pinned control closes (absolute tokens).
  bounded run: append-turn vs fold-turn recompute, observed fold period, Eq.(2) prediction from the
               OBSERVED period, amortized mean, fold surcharge CI, saving vs naive in the same run.
  2x2        : per-config substitution / correct / other with Wilson CIs (the three columns now sum
               to 1 by construction), McNemar on the paired discordance, and the factorial
               interaction, for the base, tail-stress and forced-commit protocols.
  position   : Sec. VI-D / tab:position --- per-cell Wilson CIs on both architectures, the paired
               slot contrasts and head-length contrasts quoted in the prose, and the LPM interaction.
  index      : Sec. VI-E / tab:index-fid --- the 3-arm grid, within-arm gradients (P1/P2), arm
               contrasts at matched position (P3) and the difference-in-differences.
  index cost : Sec. VI-E / tab:index-cost --- per-arm recompute and the session-cluster bootstrap
               premium of each index width over the published stable arm.
  index sk   : tab:index-sk --- the shared-key arm: the INDEX block served on top of the C7
               STATE with the distractor keyed TARGET again (plain = the published position cells,
               byte-identical STATE), per-cell rates, the arm contrasts at matched gold position and
               the pre-declared Q-A/Q-B/Q-C readings.

The position and index entries import the shipped analyzers rather than reimplementing them, so one
pass over the logs cannot disagree with the study's own report.
"""
import glob
import json, math, os, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
B = 6000
SEED = 1234


def rows(fn):
    p = os.path.join(HERE, fn)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def sess_groups(rs, keyfun=lambda r: r["session_id"]):
    g = defaultdict(list)
    for r in rs:
        g[keyfun(r)].append(r)
    return g


def boot_ci(vals_fn, groups, B=B, seed=SEED):
    """Bootstrap CI of a statistic computed over resampled sessions."""
    keys = list(groups)
    rng = __import__("numpy").random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = [groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
        flat = [x for grp in pick for x in grp]
        out.append(vals_fn(flat))
    import numpy as np
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(pairs, pred):
    """Exact binomial McNemar on discordant pairs. pairs = [(rowA, rowB)]; pred is applied to each
    row of the same cell, so b counts A-positive/B-negative and c counts the reverse."""
    b = sum(1 for x, y in pairs if pred(x) and not pred(y))
    c = sum(1 for x, y in pairs if pred(y) and not pred(x))
    if b + c == 0:
        return b, c, 1.0
    from scipy import stats
    return b, c, float(stats.binomtest(b, b + c, 0.5).pvalue)


def cost_run(fn, label):
    R = rows(fn)
    if not R:
        return {}
    by = defaultdict(list)
    for r in R:
        by[r["arm"]].append(r)
    o = {"label": label, "n": {a: len(v) for a, v in by.items()}, "arms": {}}
    for a, v in by.items():
        o["arms"][a] = {
            "prompt": st.mean(r["prompt_eval_count"] for r in v),
            "cached": st.mean(r["prompt_eval_cached_count"] for r in v),
            "recompute": st.mean(r["recompute"] for r in v),
            "prefill_ms": st.mean(r["prompt_eval_duration_ms"] for r in v),
            "acc": st.mean(r["correct"] for r in v)}
    # paired per (session,turn)
    idx = {}
    for r in R:
        idx.setdefault((r["session_id"], r["turn"]), {})[r["arm"]] = r
    pairs = [v for v in idx.values() if len(v) == len(by)]
    o["pairs"] = len(pairs)

    grp = defaultdict(list)
    for p in pairs:
        grp[p["prefix_stable"]["session_id"]].append(p)
    res = {}
    for a in by:
        if a == "prefix_stable":
            continue
        diffs = [p[a]["recompute"] - p["prefix_stable"]["recompute"] for p in pairs]
        dgrp = defaultdict(list)
        for p, d in zip(pairs, diffs):
            dgrp[p["prefix_stable"]["session_id"]].append(d)
        lo, hi = boot_ci(lambda f: st.mean(f), dgrp)
        res[f"{a}_minus_stable"] = {"D": st.mean(diffs), "ci": [lo, hi]}
    # ratio of arm means with session-cluster CI (pairs carry all arms for one turn)
    for a in by:
        if a == "prefix_stable":
            continue
        def ratio(ps, a=a):
            return (st.mean(p[a]["recompute"] for p in ps) /
                    max(st.mean(p["prefix_stable"]["recompute"] for p in ps), 1e-9))
        lo, hi = boot_ci(ratio, grp)
        res[f"{a}_over_stable_ratio"] = {"pt": ratio(pairs), "ci": [lo, hi]}
    o["contrasts"] = res
    return o


def bounded_run():
    R = rows("prefix_bounded_scored_qwen3_14b_t0.0.jsonl")
    if not R:
        return {}
    bd = [r for r in R if r["arm"] == "bounded_stable"]
    ap = [r for r in bd if not r.get("fold_event")]
    fo = [r for r in bd if r.get("fold_event")]
    lo = [r for r in R if r["arm"] == "length_only"]
    un = [r for r in R if r["arm"] == "prefix_stable"]
    idx = {}
    for r in R:
        idx.setdefault((r["session_id"], r["turn"]), {})[r["arm"]] = r
    pairs = [v for v in idx.values() if len(v) == 3]
    dgrp = defaultdict(list)
    for p in pairs:
        dgrp[p["bounded_stable"]["session_id"]].append(
            p["length_only"]["recompute"] - p["bounded_stable"]["recompute"])
    lo1, hi1 = boot_ci(lambda f: st.mean(f), dgrp)
    sgrp = defaultdict(list)
    for p in pairs:
        sgrp[p["bounded_stable"]["session_id"]].append(
            p["bounded_stable"]["recompute"] - p["prefix_stable"]["recompute"])
    lo2, hi2 = boot_ci(lambda f: st.mean(f), sgrp)
    nfold, nturn = len(fo), len(bd)
    period = nturn / max(nfold, 1)
    mean_recomp = st.mean(r["recompute"] for r in bd)
    append = st.mean(r["recompute"] for r in ap)
    fold = st.mean(r["recompute"] for r in fo)
    return {"turns": nturn, "folds": nfold, "period_observed": period,
            "append": append, "fold": fold, "mean": mean_recomp,
            "predicted_from_observed_period": append + (fold - append) / period,
            "naive": st.mean(r["recompute"] for r in lo),
            "unbounded_stable": st.mean(r["recompute"] for r in un),
            "saving_vs_naive": st.mean(r["recompute"] for r in lo) - mean_recomp,
            "saving_ci": [lo1, hi1],
            "surcharge_vs_unbounded": mean_recomp - st.mean(r["recompute"] for r in un),
            "surcharge_ci": [lo2, hi2],
            "ratio_vs_naive": st.mean(r["recompute"] for r in lo) / mean_recomp,
            "head_tokens_bounded": st.mean(r["prompt_eval_count"] for r in bd),
            "head_tokens_unbounded": st.mean(r["prompt_eval_count"] for r in un)}


def twoxtwo(fn, label):
    R = rows(fn)
    if not R:
        return {}
    idx = defaultdict(dict)
    for r in R:
        idx[(r["session_id"], r["turn"])][r["cfg"]] = r
    o = {"label": label, "cells": len(R), "cfgs": {}}
    for cfg in sorted(set(r["cfg"] for r in R)):
        s = [r for r in R if r["cfg"] == cfg]
        n = len(s)
        k = sum(r["substitution"] for r in s)
        c = sum(r["correct"] for r in s)
        o["cfgs"][cfg] = {"n": n, "subst": k / n, "subst_ci": list(wilson(k, n)),
                          "correct": c / n, "correct_ci": list(wilson(c, n)),
                          "other": 1 - k / n - c / n,
                          "recompute": st.mean(r["recompute"] for r in s)}
    pairs = [(v["c00"], v["c01"]) for v in idx.values() if "c00" in v and "c01" in v]
    bb, cc, p = mcnemar(pairs, lambda r: r["substitution"])
    o["c00_vs_c01_mcnemar"] = {"b": bb, "c": cc, "p": p}
    bb2, cc2, p2 = mcnemar([(v["c01"], v["c11"]) for v in idx.values() if "c01" in v and "c11" in v],
                           lambda r: r["correct"])
    o["c01_vs_c11_correct_mcnemar"] = {"b": bb2, "c": cc2, "p": p2}
    return o


def _quiet(fn, *a, **kw):
    """Run an analyzer function that prints its own report, keeping only its return value."""
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = fn(*a, **kw)
    return res, buf.getvalue()


def position_leg():
    """tab:position and every paired quantity quoted in the position paragraph, per architecture."""
    import analyze_position as AP
    out = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "position_scored_*.jsonl"))):
        R = rows(os.path.basename(fn))
        if not R:
            continue
        m = R[0]["model"]
        hs = sorted({r["H"] for r in R})
        h0, hlong = hs[0], hs[-1]
        idx = AP.index(R)
        cells = {f"H{h}_{s}": AP.cell_stats(R, h, s) for h in hs for s in AP.SLOTS}
        con = {}
        for h in hs:
            for a, b in (("middle", "first"), ("last", "first"), ("middle", "last")):
                con[f"H{h} {a}-{b}"] = {f: AP.paired(idx, f"H{h}_{a}", f"H{h}_{b}", f)
                                        for f in ("correct", "substitution")}
        for s in AP.SLOTS:
            con[f"H{hlong}-H{h0} at {s}"] = {
                f: AP.paired(idx, f"H{hlong}_{s}", f"H{h0}_{s}", f)
                for f in ("correct", "substitution")}
        lp, _ = _quiet(AP.lpm, R, hlong)
        groups = {}
        for r in R:
            groups.setdefault((r["session_id"], r["H"]), []).append(r["char_len"])
        out[m] = {"n": len(R), "sessions": len({r["session_id"] for r in R}),
                  "prompt_tok": {f"H{h}": st.mean(r["prompt_eval_count"] for r in R if r["H"] == h)
                                 for h in hs},
                  "prompt_tok_span": {f"H{h}": [min(r["prompt_eval_count"] for r in R if r["H"] == h),
                                                max(r["prompt_eval_count"] for r in R if r["H"] == h)]
                                      for h in hs},
                  "chars": {f"H{h}": st.mean([r["char_len"] for r in R if r["H"] == h]) for h in hs},
                  "chars_span": {f"H{h}": [min(r["char_len"] for r in R if r["H"] == h),
                                           max(r["char_len"] for r in R if r["H"] == h)] for h in hs},
                  "max_char_spread_within_session_H": max(max(v) - min(v) for v in groups.values()),
                  "n_session_H_groups": len(groups),
                  "cells": cells, "contrasts": con,
                  "lpm_mid_x_long": {f: lp.get(f, {}).get("mid_x_long") for f in AP.FIELDS}}
    return out


def index_leg():
    """tab:index-fid, P1/P2 gradients, P3 arm contrasts and the DiD, per architecture."""
    import analyze_index as AI
    out = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "index_scored_*.jsonl"))):
        R = rows(os.path.basename(fn))
        if not R:
            continue
        m = R[0]["model"]
        hs = sorted({r["H"] for r in R})
        idx = {AI.key(r): r for r in R}
        cells = {f"{a}_H{h}_{s}": AI.cell_stats(R, a, h, s)
                 for a in AI.ARMS for h in hs for s in AI.SLOTS}
        grad = {f"{a}_H{h}_{b}-first": AI.paired(idx, f"{a}_H{h}_first", f"{a}_H{h}_{b}", "correct")
                for a in AI.ARMS for h in hs for b in ("middle", "last")}
        arm = {}
        for h in hs:
            for s in AI.SLOTS:
                for a in ("index", "index_nok"):
                    arm[f"{a}_H{h}_{s}"] = {
                        f: AI.paired(idx, f"plain_H{h}_{s}", f"{a}_H{h}_{s}", f)
                        for f in ("correct", "substitution")}
        fix = {f"H{h}_{s}": AI.paired(idx, f"index_nok_H{h}_{s}", f"index_H{h}_{s}", "correct")
               for h in hs for s in ("middle", "last")}
        d = {f"H{h}": AI.did(idx, h, "correct") for h in hs}
        lp, _ = _quiet(AI.lpm, R, max(hs))
        out[m] = {"n": len(R), "sessions": len({r["session_id"] for r in R}),
                  "arms": AI.ARMS, "cells": cells, "gradient_correct": grad,
                  "vs_plain": arm, "index_minus_index_nok": fix, "did_middle_minus_first": d,
                  "lpm": {f: lp.get(f, {}) for f in AI.FIELDS}}
    return out


def shared_key_leg():
    """tab:index-sk: the INDEX block served under SHARED-KEY hazard, per architecture.

    plain is the published C7 position cells (byte-identical STATE, distractor keyed TARGET again);
    index/index_nok come from index_sk_scored_*. The estimators are analyze_index_sk's, imported
    rather than reimplemented, so this leg cannot disagree with the shared-key report."""
    import analyze_index_sk as SK
    out = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "index_sk_scored_*.jsonl"))):
        tag = os.path.basename(fn).replace("index_sk_scored_", "").replace(".jsonl", "")
        sk = rows(os.path.basename(fn))
        if not sk:
            continue
        plain = rows(f"position_scored_{tag}.jsonl")
        if not plain:
            continue
        for r in plain:
            r["arm"] = "plain"
        allrows = plain + sk
        m = sk[0]["model"]
        hs = sorted({r["H"] for r in allrows})
        cells = {f"{a}_H{h}_{s}": SK.cell_stats(allrows, a, h, s)
                 for a in SK.ARMS for h in hs for s in SK.SLOTS}
        for k, c in cells.items():
            if c is None:
                raise RuntimeError(f"{m}: empty shared-key cell {k}, the leg has not finished")
        pooled = {}
        for a in SK.ARMS:
            cs = [cells[f"{a}_H{h}_{s}"] for h in hs for s in SK.SLOTS]
            pooled[a] = {"correct": st.mean(c["correct"][0] for c in cs),
                         "substitution": st.mean(c["substitution"][0] for c in cs)}
        idx = {SK.ck(r): r for r in allrows}
        vs_plain, value_only = {}, {}
        for h in hs:
            for s in SK.SLOTS:
                for a in ("index", "index_nok"):
                    vs_plain[f"{a}_H{h}_{s}"] = {
                        f: SK.paired(idx, f"plain_H{h}_{s}", f"{a}_H{h}_{s}", f)
                        for f in ("correct", "substitution")}
                pr = SK.paired(idx, f"index_nok_H{h}_{s}", f"index_H{h}_{s}", "correct")
                if pr:
                    value_only[f"H{h}_{s}"] = pr
        qa = {}
        for s in SK.COLLAPSED:
            ds = [vs_plain[f"index_H{h}_{s}"]["correct"]["D"] for h in hs
                  if vs_plain.get(f"index_H{h}_{s}", {}).get("correct")]
            if ds:
                qa[s] = st.mean(ds)
        qb = [vs_plain[f"index_nok_H{h}_{s}"]["substitution"]["D"] for h in hs for s in SK.SLOTS
              if vs_plain.get(f"index_nok_H{h}_{s}", {}).get("substitution")]
        qc = [v["D"] for v in value_only.values()]
        out[m] = {"n": len(allrows), "sk_calls": len(sk), "plain_calls": len(plain),
                  "sessions": len({r["session_id"] for r in allrows}), "hs": hs,
                  "arms": SK.ARMS, "cells": cells, "pooled": pooled,
                  "vs_plain": vs_plain, "value_only": value_only,
                  "QA_correct_meanRD": qa,
                  "QB_subst_meanRD": st.mean(qb) if qb else None,
                  "QC_valueonly_meanRD": st.mean(qc) if qc else None}
    return out


def index_cost_leg():
    """tab:index-cost: per-arm recompute and the premium of each index width over the stable arm."""
    import analyze_index_cost as AC
    R = AC.load(os.path.join(HERE, "index_cost_scored_*.jsonl"))
    if not R:
        return {}
    pairs = list(AC.by_arm(R).values())
    o = {"model": R[0]["model"], "temp": R[0]["temp"], "pairs": len(pairs),
         "sessions": len({r["session_id"] for r in R}),
         "turns": [min(r["turn"] for r in R), max(r["turn"] for r in R)], "arms": {}}
    for a in AC.ARMS:
        sub = [p[a] for p in pairs]
        rec = {"n": len(sub), "prompt": AC.mean(sub, "prompt_eval_count"),
               "cached": AC.mean(sub, "prompt_eval_cached_count"),
               "recompute": AC.mean(sub, "recompute"),
               "prefill_ms": AC.mean(sub, "prompt_eval_duration_ms"),
               "acc": st.mean(r["correct"] for r in sub)}
        if a.startswith("index_I"):
            I = int(a.split("_I")[1])
            rec["index_rows"] = I
            rec["premium_vs_stable"] = AC.cluster_boot(pairs, "recompute", a, "prefix_stable")
            rec["prefill_premium_vs_stable"] = AC.cluster_boot(
                pairs, "prompt_eval_duration_ms", a, "prefix_stable")
        o["arms"][a] = rec
    d = AC.cluster_boot(pairs, "recompute", "index_I24", "length_only")
    o["index_I24_minus_length_only"] = d
    import pandas as pd
    import statsmodels.api as sm
    df = pd.DataFrame([{"session_id": p["index_I2"]["session_id"], "I": I,
                        "prem": p[f"index_I{I}"]["recompute"] - p["prefix_stable"]["recompute"]}
                       for p in pairs for I in (2, 8, 24)])
    X = sm.add_constant(df[["I"]].astype(float))
    fit = sm.OLS(df["prem"].astype(float), X.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": df["session_id"].to_numpy()}, use_t=True)
    o["slope_on_I"] = {"intercept": [float(b) for b in fit.params],
                       "se": [float(s) for s in fit.bse], "p": [float(p) for p in fit.pvalues],
                       "rows": int(len(df)), "clusters": int(df["session_id"].nunique())}
    # How far the straight line sits from each measured level: the claim we make is about the
    # scaling, so the reader should see the residual too.
    b0, b1 = [float(x) for x in fit.params]
    o["slope_on_I"]["fitted"] = {f"I{I}": b0 + b1 * I for I in (2, 8, 24)}
    o["slope_on_I"]["measured"] = {
        f"I{I}": o["arms"][f"index_I{I}"]["premium_vs_stable"]["D"] for I in (2, 8, 24)}
    o["slope_on_I"]["max_abs_residual"] = max(
        abs(o["slope_on_I"]["fitted"][k] - o["slope_on_I"]["measured"][k])
        for k in o["slope_on_I"]["measured"])
    return o


def comp_leg(comp_file, pub_file):
    """tab:comp: the external content-rule arm (LLMLingua-2 pruning) on the cost-leg sessions.

    Shares its estimators with analyze_prefix_comp.py on purpose: the table cell and the prose
    cannot disagree if they come from the same function. One leg per model; models are never pooled.
    """
    import analyze_prefix_comp as AP
    crows = AP.load([comp_file])
    if not crows:
        return {}
    prows = AP.load([pub_file]) if pub_file else []
    pairs = AP.cells(crows + prows)
    served = sorted({r["arm"] for r in crows} - {"length_only"},
                    key=lambda a: AP.COMP_ORDER.index(a) if a in AP.COMP_ORDER else 99)
    arms = served + ["length_only", "prefix_stable"]
    o = {"model": crows[0]["model"], "temp": crows[0]["temp"], "cells": len(pairs),
         "calls": len(crows), "sessions": len({r["session_id"] for r in crows}),
         "comp_file": comp_file, "cost_file": pub_file, "arms": {}, "roster": arms}
    for a in arms:
        sub = [v[a] for v in pairs.values() if a in v]
        if not sub:
            continue
        rec = AP.arm_stats(sub)
        rec["acc_ci"] = list(AP.wilson(round(rec["acc"] * rec["n"]), rec["n"]))
        if a != "length_only" and a != "prefix_stable":
            for b in ("length_only", "prefix_stable"):
                for fld, lab in (("prompt_eval_count", "B"), ("recompute", "R"), ("correct", "F")):
                    rec[f"{lab}_vs_{b}"] = AP.paired_boot(pairs, a, b, fld)
            rec["mcnemar_vs_length_only"] = AP.mcnemar(
                {k: v[a] for k, v in pairs.items() if a in v},
                {k: v["length_only"] for k, v in pairs.items() if "length_only" in v})
        o["arms"][a] = rec
    return o


def main():
    out = {}
    out["cost_qwen3"] = cost_run("prefix_cost_scored_qwen3_14b_t0.0.jsonl", "qwen3:14b cost run")
    out["cost_qwen25"] = cost_run("prefix_cost_scored_qwen2.5_14b_t0.0.jsonl", "qwen2.5:14b cost run")
    out["bounded"] = bounded_run()
    out["x2_base"] = twoxtwo("joint_scored_qwen3_14b.jsonl", "base 2x2 qwen3:14b")
    out["x2_tail"] = twoxtwo("joint_scored_qwen3_14b_tail.jsonl", "tail stress qwen3:14b")
    out["x2_forced_qwen"] = twoxtwo("joint_scored_qwen3_14b_tailf.jsonl", "forced commit qwen3:14b")
    out["x2_forced_qwen_epoch2"] = twoxtwo("joint_scored_qwen3_14b_tailf_epoch2.jsonl",
                                           "forced commit qwen3:14b, 2nd epoch")
    out["x2_forced_llama3"] = twoxtwo("joint_scored_llama3_latest_tailf.jsonl",
                                      "forced commit llama3:latest")
    out["layout"] = {}
    L = rows("joint_layout_scored_qwen3_14b.jsonl")
    if L:
        idx = defaultdict(dict)
        for r in L:
            idx[(r["session_id"], r["turn"])][r["cell"]] = r
        for cell in sorted(set(r["cell"] for r in L)):
            s = [r for r in L if r["cell"] == cell]
            n = len(s)
            k = sum(r["substitution"] for r in s)
            c = sum(r["correct"] for r in s)
            out["layout"][cell] = {"n": n, "subst": k / n, "subst_ci": list(wilson(k, n)),
                                   "correct": c / n, "other": 1 - k / n - c / n,
                                   "recompute": st.mean(r["recompute"] for r in s),
                                   "prompt": st.mean(r["prompt_eval_count"] for r in s),
                                   "competing": st.mean(r["competing_in_ctx"] for r in s)}
        out["layout"]["cells_complete"] = sum(1 for v in idx.values() if len(v) == 4)

    out["position"] = position_leg()
    out["index"] = index_leg()
    out["index_sk"] = shared_key_leg()
    out["index_cost"] = index_cost_leg()
    import analyze_prefix_comp as AP
    out["comp"] = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "prefix_comp_scored_*.jsonl"))):
        base = os.path.basename(fn)
        pub = AP.PRIMARY_PUB if base == AP.PRIMARY else AP.cost_pair(base)
        out["comp"][base] = comp_leg(base, pub)

    json.dump(out, open(os.path.join(HERE, "headline_numbers.json"), "w"), indent=1, default=str)

    def p2(x):
        return f"{x:.2f}" if isinstance(x, float) else x
    for k in ("cost_qwen3", "cost_qwen25"):
        d = out[k]
        if not d:
            continue
        print(f"\n== {d['label']} (paired cells {d['pairs']}) ==")
        for a, v in d["arms"].items():
            print(f"  {a:14s} prompt {v['prompt']:7.2f} cached {v['cached']:7.2f} "
                  f"recompute {v['recompute']:7.2f} prefill {v['prefill_ms']:7.1f} ms acc {v['acc']:.3f}")
        for k2, v in d["contrasts"].items():
            if "ratio" in k2:
                print(f"  {k2:34s} {v['pt']:.3f}  95%CI[{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
            else:
                print(f"  {k2:34s} {v['D']:+.2f} tok 95%CI[{v['ci'][0]:.2f},{v['ci'][1]:.2f}]")
    b = out["bounded"]
    if b:
        print(f"\n== bounded head ==\n  turns {b['turns']} folds {b['folds']} "
              f"observed period {b['period_observed']:.2f}")
        print(f"  append {b['append']:.1f} fold {b['fold']:.1f} mean {b['mean']:.1f} "
              f"predicted from observed period {b['predicted_from_observed_period']:.1f}")
        print(f"  naive {b['naive']:.1f} unbounded {b['unbounded_stable']:.1f} "
              f"ratio {b['ratio_vs_naive']:.2f}")
        print(f"  saving vs naive {b['saving_vs_naive']:.1f} CI{b['saving_ci']} "
              f"surcharge {b['surcharge_vs_unbounded']:.1f} CI{b['surcharge_ci']}")
    for k in ("x2_base", "x2_tail", "x2_forced_qwen", "x2_forced_qwen_epoch2", "x2_forced_llama3"):
        d = out[k]
        if not d:
            print(f"\n== {k}: pending ==")
            continue
        print(f"\n== {d['label']} ({d['cells']} cells) ==")
        for cfg, v in d["cfgs"].items():
            print(f"  {cfg} n={v['n']} subst {v['subst']:.3f} CI{[round(x,3) for x in v['subst_ci']]} "
                  f"correct {v['correct']:.3f} other {v['other']:.3f} recomp {v['recompute']:.1f}")
        print(f"  McNemar c00 vs c01 subst: b={d['c00_vs_c01_mcnemar']['b']} "
              f"c={d['c00_vs_c01_mcnemar']['c']} p={d['c00_vs_c01_mcnemar']['p']:.3g}")
        print(f"  McNemar c01 vs c11 correct: b={d['c01_vs_c11_correct_mcnemar']['b']} "
              f"c={d['c01_vs_c11_correct_mcnemar']['c']} p={d['c01_vs_c11_correct_mcnemar']['p']:.3g}")
    if out["layout"]:
        print(f"\n== layout 2x2 (labels x stability), complete cells {out['layout']['cells_complete']} ==")
        for cell in ("s0l0", "s0l1", "s1l0", "s1l1"):
            if cell in out["layout"]:
                v = out["layout"][cell]
                print(f"  {cell} n={v['n']} subst {v['subst']:.3f} "
                      f"CI{[round(x,3) for x in v['subst_ci']]} correct {v['correct']:.3f} "
                      f"other {v['other']:.3f} recomp {v['recompute']:.0f} "
                      f"prompt {v['prompt']:.0f} competing {v['competing']:.2f}")
    print("\n== position stress (tab:position) ==")
    for m, d in out["position"].items():
        print(f"  {m}: {d['n']} calls, {d['sessions']} sessions")
        for k, c in d["cells"].items():
            print(f"    {k:12s} n={c['n']:4d} tok {c['tok']:7.1f} corr {c['correct'][0]:.3f} "
                  f"CI[{c['correct'][1]:.3f},{c['correct'][2]:.3f}] subst {c['substitution'][0]:.3f} "
                  f"CI[{c['substitution'][1]:.3f},{c['substitution'][2]:.3f}] gold@{c['gold_pos']}")
        for k, v in d["contrasts"].items():
            for f, pr in v.items():
                if pr:
                    print(f"    {k:22s} {f:12s} {pr['D']:+.3f} CI[{pr['ci'][0]:+.3f},"
                          f"{pr['ci'][1]:+.3f}] b/c={pr['b']}/{pr['c']} p={pr['p']:.3g}")
        for f, c in d["lpm_mid_x_long"].items():
            if c:
                print(f"    LPM {f:14s} mid x long {c[0]:+.3f} se {c[1]:.3f} p={c[2]:.3g}")
    if not out["position"]:
        print("  (no position_scored_*.jsonl)")

    print("\n== rewritable-index fidelity (tab:index-fid) ==")
    for m, d in out["index"].items():
        print(f"  {m}: {d['n']} calls, {d['sessions']} sessions, arms {list(d['arms'])}")
        for k, c in d["cells"].items():
            print(f"    {k:22s} n={c['n']:4d} idx_rows {c['index_lines']:2d} corr {c['correct'][0]:.3f} "
                  f"CI[{c['correct'][1]:.3f},{c['correct'][2]:.3f}] subst {c['substitution'][0]:.3f}")
        for k, pr in d["gradient_correct"].items():
            if pr:
                print(f"    gradient {k:24s} {pr['D']:+.3f} CI[{pr['ci'][0]:+.3f},{pr['ci'][1]:+.3f}] "
                      f"p={pr['p']:.3g}")
        for k, v in d["vs_plain"].items():
            pr = v["correct"]
            if pr:
                print(f"    vs plain {k:24s} corr {pr['D']:+.3f} "
                      f"CI[{pr['ci'][0]:+.3f},{pr['ci'][1]:+.3f}] p={pr['p']:.3g}"
                      + (f" | subst {v['substitution']['D']:+.3f} p={v['substitution']['p']:.3g}"
                         if v["substitution"] else ""))
        for k, pr in d["index_minus_index_nok"].items():
            if pr:
                print(f"    P3 index-index_nok {k:14s} {pr['D']:+.3f} "
                      f"CI[{pr['ci'][0]:+.3f},{pr['ci'][1]:+.3f}] p={pr['p']:.3g}")
        for k, dd in d["did_middle_minus_first"].items():
            if dd:
                print(f"    DiD [middle-first]_index-[middle-first]_plain {k:6s} {dd['D']:+.3f} "
                      f"CI[{dd['ci'][0]:+.3f},{dd['ci'][1]:+.3f}] n={dd['n']} p={dd['p']:.3g}")
    if not out["index"]:
        print("  (no index_scored_*.jsonl)")

    print("\n== shared-key index fidelity (tab:index-sk) ==")
    for m, d in out["index_sk"].items():
        print(f"  {m}: {d['n']} calls ({d['sk_calls']} index/index_nok + {d['plain_calls']} "
              f"published C7 plain), {d['sessions']} sessions")
        for k, c in d["cells"].items():
            print(f"    {k:22s} n={c['n']:4d} corr {c['correct'][0]:.3f} "
                  f"CI[{c['correct'][1]:.3f},{c['correct'][2]:.3f}] subst {c['substitution'][0]:.3f} "
                  f"CI[{c['substitution'][1]:.3f},{c['substitution'][2]:.3f}] other {c['other']:.3f}")
        for a in d["arms"]:
            p = d["pooled"][a]
            print(f"    pooled {a:10s} corr {p['correct']:.3f} subst {p['substitution']:.3f}")
        for k, pr in d["QA_correct_meanRD"].items():
            print(f"    Q-A index-plain correct, mean over H at slot {k}: {pr:+.3f}")
        if d["QB_subst_meanRD"] is not None:
            print(f"    Q-B stale-index substitution vs plain, mean: {d['QB_subst_meanRD']:+.3f}")
        if d["QC_valueonly_meanRD"] is not None:
            print(f"    Q-C index vs index_nok correct, mean: {d['QC_valueonly_meanRD']:+.3f}")
    if not out["index_sk"]:
        print("  (no index_sk_scored_*.jsonl)")

    c = out["index_cost"]
    if c:
        print(f"\n== rewritable-index cost (tab:index-cost): {c['model']} temp {c['temp']}, "
              f"{c['pairs']} paired turns x {c['sessions']} sessions ==")
        for a, v in c["arms"].items():
            line = (f"  {a:14s} n={v['n']:4d} prompt {v['prompt']:7.1f} cached {v['cached']:7.1f} "
                    f"recompute {v['recompute']:7.1f} prefill {v['prefill_ms']:7.1f} ms acc {v['acc']:.3f}")
            if "premium_vs_stable" in v:
                pv = v["premium_vs_stable"]
                line += (f" | premium {pv['D']:+.1f} tok CI[{pv['ci'][0]:+.1f},{pv['ci'][1]:+.1f}]"
                         f" p={pv['p']:.3g}")
            print(line)
        s = c["slope_on_I"]
        print(f"  slope on I: intercept {s['intercept'][0]:+.2f} slope {s['intercept'][1]:+.3f} "
              f"tok/row se {s['se'][1]:.3f} p={s['p'][1]:.3g} "
              f"(rows={s['rows']}, clusters={s['clusters']})")
        d = c["index_I24_minus_length_only"]
        print(f"  index_I24 - length_only recompute {d['D']:+.1f} tok "
              f"CI[{d['ci'][0]:+.1f},{d['ci'][1]:+.1f}]")
    else:
        print("\n== rewritable-index cost: (no index_cost_scored_*.jsonl) ==")

    c = out["comp"]
    if c:
        for base, leg in c.items():
            if not leg:
                continue
            print(f"\n== external content-rule arm (tab:comp): {leg['model']} temp {leg['temp']}, "
                  f"{leg['calls']} calls, {leg['cells']} cells, {leg['sessions']} sessions "
                  f"[{base} vs {leg['cost_file']}] ==")
            for a, v in leg["arms"].items():
                print(f"    {a:14s} n={v['n']:4d} B {v['prompt_mean']:7.1f} R {v['recomp_mean']:7.1f} "
                      f"(med {v['recomp_med']:.0f}) hit {100*v['cached_mean']/max(v['prompt_mean'],1):4.1f}% "
                      f"F {v['acc']:.3f} CI[{v['acc_ci'][0]:.3f},{v['acc_ci'][1]:.3f}] "
                      f"gold@ctx {v['gold_in_ctx']:.3f} prefill {v['prefill_ms']:.1f} ms")
                for b in ("length_only", "prefix_stable"):
                    if a in ("length_only", "prefix_stable") or f"R_vs_{b}" not in v:
                        continue
                    s = "  ".join(f"{lab} {v[f'{lab}_vs_{b}']['D']:+8.1f}"
                                  f"[{v[f'{lab}_vs_{b}']['ci'][0]:+.1f},{v[f'{lab}_vs_{b}']['ci'][1]:+.1f}]"
                                  for lab in ("B", "R", "F") if v[f"{lab}_vs_{b}"])
                    print(f"        vs {b:14s} {s}")
                if "mcnemar_vs_length_only" in v:
                    p, n01, n10 = v["mcnemar_vs_length_only"]
                    print(f"        McNemar vs length_only p={p:.3g} (b={n01}, c={n10})")
    else:
        print("\n== competitor arm: (no prefix_comp_scored_*.jsonl) ==")
    print("\n-> headline_numbers.json")


if __name__ == "__main__":
    main()
