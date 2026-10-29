#!/usr/bin/env python
"""Analysis: does a rewritable current-binding INDEX erase the C7 position gradient?

Outcomes and tests are the ones the C7 stress publishes (same buckets, same pairing, same
pre-declared 0.03 actionability floor reused unchanged), applied to a 3-arm x 2 head-length x 3
slot grid. Reading the alternative as a falsifiable prediction:

  P1  in `plain` the gradient is present: middle - first (and last - first) on `correct` are
      negative and clear the floor. If P1 fails, the INDEX arm has nothing to fix and we say so.
  P2  in `index` the gradient is GONE: the same within-arm contrasts shrink to inside the floor
      with a CI covering zero. Primary statistic = the difference-in-differences
      [middle-first]_index - [middle-first]_plain, cluster-bootstrapped on sessions, tested
      against zero; the per-arm contrasts are reported alongside so the reader sees the shrinkage.
  P3  the fix is carried by the ANSWER ROW, not by having another block near the end: at the
      collapsed slots, index - index_nok is positive and clears the floor, and index_nok still
      shows the gradient.

Statistics: exact McNemar on within-session discordant pairs, session-cluster bootstrap (6000
reps) for risk differences, Wilson CIs per cell, and a cluster-robust LPM with arm x position
interactions. Cross-model sign consistency is printed per model. Reads every
index_scored_*.jsonl it finds. Env: IX_REPS, IX_ONLY.
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
REPS = int(os.environ.get("IX_REPS", "6000"))
SLOTS = ("first", "middle", "last")
ARMS = ("plain", "index", "index_nok")
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


def key(r):
    return (r["session_id"], f"{r['arm']}_H{r['H']}_{r['slot']}")


def paired(idx, a, b, field):
    """Paired risk difference b - a over cells sharing a session, with cluster CI and McNemar."""
    by_sess = defaultdict(dict)
    for (sid, k), r in idx.items():
        by_sess[sid][k] = r
    d, bb, cc = defaultdict(list), 0, 0
    for sid, g in by_sess.items():
        if a in g and b in g:
            d[sid].append(g[b][field] - g[a][field])
            x, y = bool(g[b][field]), bool(g[a][field])
            if x and not y:
                bb += 1
            elif y and not x:
                cc += 1
    flat = [x for v in d.values() for x in v]
    if not flat:
        return None
    p = float(stats.binomtest(bb, bb + cc, 0.5).pvalue) if bb + cc else 1.0
    return {"D": st.mean(flat), "ci": boot(d), "n": len(flat), "b": bb, "c": cc, "p": p}


def did(idx, h, field):
    """[middle-first]_index - [middle-first]_plain, per session, cluster-bootstrapped."""
    by_sess = defaultdict(dict)
    for (sid, k), r in idx.items():
        by_sess[sid][k] = r[field]
    d = {}
    for sid, g in by_sess.items():
        a, b = g.get(f"index_H{h}_middle"), g.get(f"index_H{h}_first")
        c, e = g.get(f"plain_H{h}_middle"), g.get(f"plain_H{h}_first")
        if None in (a, b, c, e):
            continue
        d[sid] = [(a - b) - (c - e)]
    if not d:
        return None
    flat = [x for v in d.values() for x in v]
    ci = boot(d)
    w = stats.wilcoxon(flat) if len(set(flat)) > 1 else None
    return {"D": st.mean(flat), "ci": ci, "n": len(flat),
            "p": float(w.pvalue) if w else 1.0}


def cell_stats(rows, arm, h, slot):
    s = [r for r in rows if r["arm"] == arm and r["H"] == h and r["slot"] == slot]
    n = len(s)
    if not n:
        return None
    o = {"n": n, "other": st.mean(r["other"] for r in s),
         "tok": st.mean(r["prompt_eval_count"] for r in s),
         "gold_pos": s[0]["gold_pos"], "chars": s[0]["char_len"],
         "index_lines": s[0]["index_lines"]}
    for f in FIELDS:
        k = sum(r[f] for r in s)
        lo, hi = wilson(k, n)
        o[f] = (k / n, lo, hi)
    return o


def table(rows, tag):
    print(f"\n== {tag}: arm x head length x gold position "
          f"(one competing near-twin binding in context in every cell) ==")
    print(f"{'arm':10s} {'H':>3s} {'slot':7s} {'gold@':6s} {'idx':>4s} {'chars':>6s} {'tok':>6s} {'n':>4s} "
          + " ".join(f"{f:>21s}" for f in FIELDS) + f" {'other':>6s}")
    out = {}
    hs = sorted({r["H"] for r in rows})
    for arm in ARMS:
        for h in hs:
            for slot in SLOTS:
                c = cell_stats(rows, arm, h, slot)
                if c is None:
                    continue
                out[f"{arm}_H{h}_{slot}"] = c
                cells = " ".join(f"[{c[f][0]:.3f} {c[f][1]:.3f},{c[f][2]:.3f}]".rjust(21) for f in FIELDS)
                print(f"{arm:10s} {h:3d} {slot:7s} {c['gold_pos']:6d} {c['index_lines']:4d} "
                      f"{c['chars']:6d} {c['tok']:6.1f} {c['n']:4d} {cells} {c['other']:6.3f}")
    return out, hs


def gradients(rows, hs):
    """P1 and P2: the within-arm position gradient, slot contrast by slot contrast."""
    idx = {key(r): r for r in rows}
    print("\n== within-arm position gradient (contrast on `correct`, paired within session) ==")
    g = {}
    for arm in ARMS:
        for h in hs:
            for b, a, lbl in ((f"{arm}_H{h}_middle", f"{arm}_H{h}_first", "middle-first"),
                              (f"{arm}_H{h}_last", f"{arm}_H{h}_first", "last-first")):
                pr = paired(idx, a, b, "correct")
                if pr is None:
                    continue
                g[(arm, h, lbl)] = pr
                flag = "  <-- gradient" if -pr["D"] >= SESOI and pr["p"] < 0.05 else ""
                print(f"  {arm:10s} H={h:<3d} {lbl:13s} RD {pr['D']:+.3f} "
                      f"CI[{pr['ci'][0]:+.3f},{pr['ci'][1]:+.3f}] b/c={pr['b']}/{pr['c']} "
                      f"p={pr['p']:.3g}{flag}")
    return g


def arm_contrasts(rows, hs):
    """P3 and the level shift: index / index_nok against plain at each slot."""
    idx = {key(r): r for r in rows}
    print("\n== arm contrast at matched content and matched gold position ==")
    res = {}
    for h in hs:
        for slot in SLOTS:
            for arm in ("index", "index_nok"):
                pr = paired(idx, f"plain_H{h}_{slot}", f"{arm}_H{h}_{slot}", "correct")
                sx = paired(idx, f"plain_H{h}_{slot}", f"{arm}_H{h}_{slot}", "substitution")
                if pr is None:
                    continue
                res[(h, slot, arm)] = pr
                line = (f"  H={h:<3d} {slot:7s} {arm:10s} vs plain: correct RD {pr['D']:+.3f} "
                        f"CI[{pr['ci'][0]:+.3f},{pr['ci'][1]:+.3f}] b/c={pr['b']}/{pr['c']} p={pr['p']:.3g}")
                if sx:
                    line += (f" | subst {sx['D']:+.3f} CI[{sx['ci'][0]:+.3f},{sx['ci'][1]:+.3f}] "
                             f"p={sx['p']:.3g}")
                print(line)
            pr = paired(idx, f"index_nok_H{h}_{slot}", f"index_H{h}_{slot}", "correct")
            if pr:
                res[(h, slot, "index_vs_nok")] = pr
                print(f"  H={h:<3d} {slot:7s} PLACEBO    index vs index_nok: correct RD {pr['D']:+.3f} "
                      f"CI[{pr['ci'][0]:+.3f},{pr['ci'][1]:+.3f}] b/c={pr['b']}/{pr['c']} p={pr['p']:.3g}")
    return res


def dids(hs):
    idx = {}
    for fn in FILES:
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l)
            idx.setdefault(r["model"], {})[key(r)] = r
    print("\n== difference-in-differences: [middle-first]_index - [middle-first]_plain ==")
    out = {}
    for m, per in idx.items():
        for h in hs:
            d = did(per, h, "correct")
            if d is None:
                continue
            out[(m, h)] = d
            off = "offsets the gradient" if d["D"] > 0 and d["ci"][0] > 0 else \
                ("does not offset it" if d["ci"][0] <= 0 <= d["ci"][1] else "moves the gradient further")
            print(f"  {m:14s} H={h:<3d} DiD {d['D']:+.3f} CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] "
                  f"n={d['n']} p={d['p']:.3g} -> {off}; residual gradient in the index arm is "
                  f"reported as P2 above")
    return out


def lpm(rows, hlong):
    """Arm x position interaction in one regression: correct ~ slots + long + arm + arm x slot."""
    df = pd.DataFrame([{
        "mid": int(r["slot"] == "middle"), "last": int(r["slot"] == "last"),
        "long": int(r["H"] == hlong),
        "idx": int(r["arm"] == "index"), "nok": int(r["arm"] == "index_nok"),
        "correct": r["correct"], "substitution": r["substitution"],
        "session_id": r["session_id"]} for r in rows])
    for c in ("mid", "last"):
        df[f"{c}_x_idx"] = df[c] * df["idx"]
        df[f"{c}_x_nok"] = df[c] * df["nok"]
    cols = ["mid", "last", "long", "idx", "nok", "mid_x_idx", "last_x_idx", "mid_x_nok", "last_x_nok"]
    X = sm.add_constant(df[cols].astype(float))
    grp = df["session_id"].to_numpy()
    if len(set(grp)) < 3:
        print("\n  LPM skipped: fewer than 3 clusters")
        return {}
    names = ["const"] + cols
    res = {}
    for f in ("correct", "substitution"):
        m = sm.OLS(df[f].astype(float), X.to_numpy()).fit(
            cov_type="cluster", cov_kwds={"groups": grp}, use_t=True)
        res[f] = {n: [float(b), float(se), float(p)]
                  for n, b, se, p in zip(names, m.params, m.bse, m.pvalues)}
        print(f"\n  LPM {f} ~ slot x arm (reference = plain@first@short, "
              f"cluster-robust on {len(set(grp))} sessions):")
        for n, b, se, p in zip(names, m.params, m.bse, m.pvalues):
            flag = "  <-- clears SESOI" if n != "const" and abs(b) >= SESOI and p < 0.05 else ""
            print(f"    {n:12s} RD {b:+.3f} se {se:.3f} [{b - 1.96 * se:+.3f},{b + 1.96 * se:+.3f}] "
                  f"p={p:.3g}{flag}")
    return res


def verdict(g, ac):
    print("\n== pre-declared reading ==")
    for h in sorted({k[1] for k in g}):
        p_mid = g.get(("plain", h, "middle-first"))
        i_mid = g.get(("index", h, "middle-first"))
        n_mid = g.get(("index_nok", h, "middle-first"))
        fix = ac.get((h, "middle", "index"))
        if not p_mid:
            print(f"  H={h}: not enough cells yet")
            continue
        p1 = -p_mid["D"] >= SESOI and p_mid["p"] < 0.05
        p2 = bool(i_mid and abs(i_mid["D"]) < SESOI and i_mid["ci"][0] <= 0 <= i_mid["ci"][1])
        p3 = bool(fix and fix["D"] >= SESOI and fix["p"] < 0.05)
        msg = (f"  H={h:<3d} P1 gradient in plain: {'YES' if p1 else 'no'} "
               f"({p_mid['D']:+.3f}, p={p_mid['p']:.3g})")
        msg += (" | P2 gradient in index: " + ("GONE" if p2 else "still present")
                + (f" ({i_mid['D']:+.3f})" if i_mid else " (n/a)"))
        msg += (" | P3 carried by the answer row: " + ("YES" if p3 else "no")
                + (f" (index-plain at middle {fix['D']:+.3f})" if fix else " (n/a)"))
        if n_mid:
            msg += f" | placebo (index_nok) gradient {n_mid['D']:+.3f}"
        print(msg)


def main():
    global FILES
    FILES = sorted(glob.glob(os.path.join(HERE, "index_scored_*.jsonl")))
    only = os.environ.get("IX_ONLY")
    if only:
        FILES = [f for f in FILES if only in os.path.basename(f)]
    if not FILES:
        print("no index_scored_*.jsonl yet"); sys.exit(1)
    print("run files:", [os.path.basename(x) for x in FILES])
    hs = None
    for fn in FILES:
        rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
        m = rows[0]["model"]
        print(f"\n### {m}: {len(rows)} calls, {len({r['session_id'] for r in rows})} sessions, "
              f"arms {sorted({r['arm'] for r in rows})}, "
              f"err-answers {sum(str(r.get('answer', '')).startswith('__ERR__') for r in rows)}")
        _, hs_local = table(rows, m)
        hs = hs_local if hs is None else hs
        g = gradients(rows, hs_local)
        ac = arm_contrasts(rows, hs_local)
        lpm(rows, max(hs_local))
        verdict(g, ac)
    dids(hs)


if __name__ == "__main__":
    main()
