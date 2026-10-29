#!/usr/bin/env python
"""Analysis: does the position of the gold line inside the append-only head matter?

Outcomes, fixed here before the first model call (same order as the layout factorial):
  primary    correct        the gold value is named and no competing value is named;
  secondary  mentions_gold  the gold value is named at all (pure access, immune to hedging);
  secondary  substitution   a competing value is named and gold is not (contamination).
Primary contrasts (paired within session, content identical, only the gold line moved):
  middle - first, middle - last              (is a mid-head value lost?)
  middle@H24 - middle@H6                     (does a LONGER head deepen it?)
Statistics: exact McNemar on discordant pairs, session-cluster bootstrap (6000 reps) for the risk
difference, cluster-robust LPM for the position x head-length interaction, Wilson CIs per cell.
The actionability floor is the paper's pre-declared 0.03 substitution margin, reused unchanged;
It is reported as a confirmatory test of a specific
question, not as a pre-registered primary analysis.

Reads every position_scored_*.jsonl it finds and reports cross-model sign consistency.
Env: PS_LONG (head length treated as the stressed level, default = max H present), PS_REPS,
PS_ONLY (substring filter on the run-file name, for interim single-model reads while a second
model is still serving).
"""
import glob, json, os, statistics as st, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

SESOI = 0.03
REPS = int(os.environ.get("PS_REPS", "6000"))
SLOTS = ("first", "middle", "last")
FIELDS = ("correct", "mentions_gold", "substitution")


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot(d, B=REPS, seed=1234):
    keys = list(d)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = [d[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
        out.append(st.mean([x for g in pick for x in g]))
    return [float(np.percentile(out, q)) for q in (2.5, 97.5)]


def index(rows):
    return {(r["session_id"], f"H{r['H']}_{r['slot']}"): r for r in rows}


def paired(idx, a, b, field):
    by_sess = defaultdict(dict)
    for (sid, k), r in idx.items():
        by_sess[sid][k] = r
    d, bb, cc = defaultdict(list), 0, 0
    for sid, g in by_sess.items():
        if a in g and b in g:
            d[sid].append(g[a][field] - g[b][field])
            x, y = bool(g[a][field]), bool(g[b][field])
            if x and not y:
                bb += 1
            elif y and not x:
                cc += 1
    flat = [x for v in d.values() for x in v]
    if not flat:
        return None
    p = float(stats.binomtest(bb, bb + cc, 0.5).pvalue) if bb + cc else 1.0
    return {"D": st.mean(flat), "ci": boot(d), "n": len(flat), "b": bb, "c": cc, "p": p}


def cell_stats(rows, h, slot):
    s = [r for r in rows if r["H"] == h and r["slot"] == slot]
    n = len(s)
    toks = [r["prompt_eval_count"] for r in s]
    chars = [r["char_len"] for r in s]
    o = {"n": n, "other": st.mean(r["other"] for r in s),
         "tok": st.mean(toks), "tok_min": min(toks), "tok_max": max(toks),
         "gold_pos": s[0]["gold_pos"], "wrong_pos": s[0]["wrong_pos"],
         "chars": st.mean(chars), "chars_min": min(chars), "chars_max": max(chars)}
    for f in FIELDS:
        k = sum(r[f] for r in s)
        lo, hi = wilson(k, n)
        o[f] = (k / n, lo, hi)
    return o


def size_summary(rows):
    """Print the prompt-size facts needed to rule out a length confound:
    per-head-length mean/min/max tokens and chars, and the char spread ACROSS the three
    slots inside one (session, head-length) group, which the design gate asserts is zero."""
    print("  prompt size (not a manipulated axis): ", end="")
    for h in sorted({r["H"] for r in rows}):
        s = [r for r in rows if r["H"] == h]
        toks = [r["prompt_eval_count"] for r in s]
        chars = [r["char_len"] for r in s]
        print(f"H={h}: {st.mean(toks):.1f} tok mean [{min(toks)}, {max(toks)}], "
              f"{st.mean(chars):.0f} chars mean [{min(chars)}, {max(chars)}]  ", end="")
    print()
    groups = {}
    for r in rows:
        groups.setdefault((r["session_id"], r["H"]), []).append(r["char_len"])
    spread = max((max(v) - min(v)) for v in groups.values())
    print(f"  char spread across the three slots within a (session, H) group: "
          f"max {spread} over {len(groups)} groups (permutation of one line: expected 0)")
    return {"by_head": {str(h): {"tok_mean": st.mean([r["prompt_eval_count"] for r in rows if r["H"] == h]),
                                 "tok_min": min(r["prompt_eval_count"] for r in rows if r["H"] == h),
                                 "tok_max": max(r["prompt_eval_count"] for r in rows if r["H"] == h),
                                 "char_mean": st.mean([r["char_len"] for r in rows if r["H"] == h]),
                                 "char_min": min(r["char_len"] for r in rows if r["H"] == h),
                                 "char_max": max(r["char_len"] for r in rows if r["H"] == h)}
                       for h in sorted({r["H"] for r in rows})},
            "max_within_group_char_spread": spread, "n_groups": len(groups)}


def table(rows, tag):
    print(f"\n== {tag}: position of the gold line in the STATE head "
          f"(one competing near-twin value in context in every cell) ==")
    print(f"{'H':>3s} {'slot':7s} {'gold@':6s} {'chars':>6s} {'tok':>6s} {'n':>4s} "
          + " ".join(f"{f:>21s}" for f in FIELDS) + f" {'other':>6s}")
    out = {}
    for h in sorted({r["H"] for r in rows}):
        for slot in SLOTS:
            c = cell_stats(rows, h, slot)
            out[f"H{h}_{slot}"] = c
            cells = " ".join(f"[{c[f][0]:.3f} {c[f][1]:.3f},{c[f][2]:.3f}]".rjust(21) for f in FIELDS)
            print(f"{h:3d} {slot:7s} {c['gold_pos']:6d} {c['chars']:6.0f} {c['tok']:6.1f} {c['n']:4d} "
                  f"{cells} {c['other']:6.3f}")
    return out


def contrast_plan(hs, hlong):
    h0 = min(hs)
    plan = []
    for h in hs:
        plan += [(f"H{h}_middle", f"H{h}_first", f"H={h:<2d} middle - first"),
                 (f"H{h}_middle", f"H{h}_last", f"H={h:<2d} middle - last")]
    for slot in SLOTS:
        plan.append((f"H{hlong}_{slot}", f"H{h0}_{slot}", f"H={hlong}-H={h0} at {slot}"))
    return plan


def contrasts(rows):
    idx = index(rows)
    hs = sorted({r["H"] for r in rows})
    hlong = int(os.environ.get("PS_LONG") or hs[-1])
    res, order = {}, []
    for a, b, lbl in contrast_plan(hs, hlong):
        rec = {}
        line = f"  {lbl:26s}"
        for f in FIELDS:
            pr = paired(idx, a, b, f)
            rec[f] = pr
            line += (f" | {f[:7]} {pr['D']:+.3f} CI[{pr['ci'][0]:+.3f},{pr['ci'][1]:+.3f}] "
                     f"b/c={pr['b']}/{pr['c']} p={pr['p']:.3g}")
        print(line)
        res[f"{a}-{b}"] = rec
        order.append((a, b, lbl))
    return res, hlong


def lpm(rows, hlong):
    df = pd.DataFrame([{
        "mid": int(r["slot"] == "middle"), "last": int(r["slot"] == "last"),
        "long": int(r["H"] == hlong), "correct": r["correct"],
        "mentions_gold": r["mentions_gold"], "substitution": r["substitution"],
        "session_id": r["session_id"]} for r in rows])
    df["mid_x_long"] = df["mid"] * df["long"]
    df["last_x_long"] = df["last"] * df["long"]
    cols = ["mid", "last", "long", "mid_x_long", "last_x_long"]
    X = sm.add_constant(df[cols].astype(float))
    grp = df["session_id"].to_numpy()
    if len(set(grp)) < 3:
        print("\n  LPM skipped: fewer than 3 clusters (session count too small to cluster)")
        return {}
    res = {}
    names = ["const"] + cols          # add_constant prepends, so the name list must too
    for f in FIELDS:
        m = sm.OLS(df[f].astype(float), X.to_numpy()).fit(
            cov_type="cluster", cov_kwds={"groups": grp}, use_t=True)
        res[f] = {n: [float(b), float(se), float(p)]
                  for n, b, se, p in zip(names, m.params, m.bse, m.pvalues)}
        print(f"\n  LPM {f} ~ position*headlength, reference cell = first@short "
              f"(cluster-robust on {len(set(grp))} sessions):")
        for n, b, se, p in zip(names, m.params, m.bse, m.pvalues):
            flag = "  <-- clears SESOI" if n != "const" and abs(b) >= SESOI and p < 0.05 else ""
            print(f"    {n:12s} RD {b:+.3f} se {se:.3f} "
                  f"[{b-1.96*se:+.3f},{b+1.96*se:+.3f}] p={p:.3g}{flag}")
    return res


def sesoi_verdict(res):
    worst, where = 0.0, "-"
    for k, rec in res.items():
        for f in ("correct", "mentions_gold"):
            if -rec[f]["D"] > worst:
                worst, where = -rec[f]["D"], f"{k} on {f}"
    verdict = "POSITION IS ACTIONABLE" if worst >= SESOI else \
        "below the floor: position is not an actionable fidelity lever at this operating point"
    print(f"\n  largest position-induced FIDELITY LOSS over the {len(res)} planned contrasts: "
          f"{worst:.3f} ({where}) vs pre-declared floor {SESOI} -> {verdict}")
    return {"worst_loss": worst, "where": where, "verdict": verdict}


def main():
    files = sorted(glob.glob(os.path.join(HERE, "position_scored_*.jsonl")))
    only = os.environ.get("PS_ONLY")
    if only:
        files = [f for f in files if only in os.path.basename(f)]
    if not files:
        print("no position_scored_*.jsonl yet"); sys.exit(1)
    print("run files:", [os.path.basename(x) for x in files])
    models = {}
    for fn in files:
        rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
        m = rows[0]["model"]
        print(f"\n### {m}: {len(rows)} calls, {len({r['session_id'] for r in rows})} sessions, "
              f"err-answers {sum(str(r.get('answer','')).startswith('__ERR__') for r in rows)}")
        cells = table(rows, m)
        sizes = size_summary(rows)
        res, hlong = contrasts(rows)
        lp = lpm(rows, hlong)
        v = sesoi_verdict(res)
        models[m] = {"rows": rows, "cells": cells, "sizes": sizes, "contrasts": res,
                     "lpm": lp, "sesoi": v}

    if len(models) > 1:
        print("\n== cross-model sign consistency (risk difference, middle - first) ==")
        for key in sorted(next(iter(models.values()))["contrasts"]):
            if "middle" not in key:
                continue
            for f in ("correct", "mentions_gold", "substitution"):
                frag = []
                for m, d in models.items():
                    pr = d["contrasts"].get(key, {}).get(f)
                    if pr:
                        frag.append(f"{m} {pr['D']:+.3f} (p={pr['p']:.3g})")
                print(f"  {key:28s} {f:14s} " + " | ".join(frag))

    prim = next((d for m, d in models.items() if "qwen3_14b" in m), list(models.values())[0])
    dump = {"primary_model": [m for m, d in models.items() if d is prim][0],
            "sesoi_floor": SESOI,
            "cells": prim["cells"], "prompt_sizes": prim["sizes"],
            "contrasts": prim["contrasts"],
            "lpm": prim["lpm"], "sesoi": prim["sesoi"],
            "models": {m: {"n": len(d["rows"]), "cells": d["cells"], "sesoi": d["sesoi"]}
                       for m, d in models.items()}}
    json.dump(dump, open(os.path.join(HERE, "position_analysis.json"), "w"), indent=1, default=str)
    print("\n-> position_analysis.json (written next to this script)")


if __name__ == "__main__":
    main()
