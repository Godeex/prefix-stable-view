#!/usr/bin/env python
"""Shared-key analysis: does the current-binding INDEX buy fidelity under SHARED-KEY hazard?

The published INDEX leg (analyze_index.py) re-keyed the near-twin distractor to RELATED1, so a
stale value never competed with the gold under one key. That re-key is the operation that removes
the C7 shared-key hazard, so this leg asks the decisive question: with the distractor
keyed TARGET again (STATE byte-identical to the C7 plain cell the model fails), does adding a
rewritable current-binding INDEX still repair fidelity, and what happens when the INDEX names the
stale value instead of the current one?

Arms on this leg:
  plain      = the published C7 position arm (position_scored_*.jsonl), same shared-key STATE
  index      = STATE || INDEX(`TARGET -> gold`)   || RECENT   (fresh current binding)
  index_nok  = STATE || INDEX(`TARGET -> wrong`)  || RECENT   (stale binding names the competitor)

Pre-declared readings:
  Q-A  index beats plain on `correct` at the collapsed slots (middle, last) and clears the 0.03
       floor: the gain survives real hazard, so C7 is not an artifact of re-keying. If it does not
       clear, the block is inert under conflict and we report the negative.
  Q-B  index_nok drives `substitution` above plain: when STATE carries both TARGET=gold and
       TARGET=wrong and the INDEX names the wrong one, the model follows the view, i.e. it reads
       INDEX over the raw stale line. This is the "which line does the model read" probe; it also
       answers the mutual-negation worry (a decision-bearing index is not merely hazard suppression).
  Q-C  index vs index_nok on `correct` at each slot: the block's VALUE, not its presence, carries
       the effect (the placebo that names the competitor should sit near plain, below index).

Statistics reuse the C7/index口径 unchanged: Wilson per cell, session-cluster bootstrap (6000 reps)
for risk differences, exact McNemar on within-session discordant pairs. Env: IX_SK_REPS.
"""
import glob, json, os, statistics as st, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy import stats

SESOI = 0.03
REPS = int(os.environ.get("IX_SK_REPS", "6000"))
SLOTS = ("first", "middle", "last")
ARMS = ("plain", "index", "index_nok")
FIELDS = ("correct", "mentions_gold", "substitution")
COLLAPSED = ("middle", "last")


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


def ck(r):
    return (r["session_id"], f"{r['arm']}_H{r['H']}_{r['slot']}")


def paired(idx, a, b, field):
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


def collapse_rows(rows):
    return [r for r in rows if r["slot"] in COLLAPSED]


def cell_stats(rows, arm, h, slot):
    s = [r for r in rows if r["arm"] == arm and r["H"] == h and r["slot"] == slot]
    n = len(s)
    if not n:
        return None
    o = {"n": n, "other": st.mean(r["other"] for r in s)}
    for f in FIELDS:
        k = sum(r[f] for r in s)
        lo, hi = wilson(k, n)
        o[f] = (k / n, lo, hi)
    return o


def table(rows, tag):
    print(f"\n== {tag}: shared-key arm x head length x gold position ==")
    print(f"{'arm':10s} {'H':>3s} {'slot':7s} {'n':>4s} " +
          " ".join(f"{f:>21s}" for f in FIELDS) + f" {'other':>6s}")
    hs = sorted({r["H"] for r in rows})
    for arm in ARMS:
        for h in hs:
            for slot in SLOTS:
                c = cell_stats(rows, arm, h, slot)
                if c is None:
                    continue
                cells = " ".join(f"[{c[f][0]:.3f} {c[f][1]:.3f},{c[f][2]:.3f}]".rjust(21) for f in FIELDS)
                print(f"{arm:10s} {h:3d} {slot:7s} {c['n']:4d} {cells} {c['other']:6.3f}")
    return hs


def pooled(rows, hs):
    """Correctness and substitution averaged over slots within an arm (the headline levels)."""
    print(f"\n== pooled over the grid (levels, mean of cell rates) ==")
    out = {}
    for arm in ARMS:
        cells = [cell_stats(rows, arm, h, s) for h in hs for s in SLOTS]
        cells = [c for c in cells if c]
        if not cells:
            continue
        corr = st.mean(c["correct"][0] for c in cells)
        sub = st.mean(c["substitution"][0] for c in cells)
        out[arm] = (corr, sub)
        print(f"  {arm:10s} correct {corr:.3f}  substitution {sub:.3f}")
    return out


def contrasts(rows, hs):
    idx = {ck(r): r for r in rows}
    print("\n== arm contrasts at matched content and matched gold position ==")
    res = {}
    for h in hs:
        for slot in SLOTS:
            for arm in ("index", "index_nok"):
                pr = paired(idx, f"plain_H{h}_{slot}", f"{arm}_H{h}_{slot}", "correct")
                sx = paired(idx, f"plain_H{h}_{slot}", f"{arm}_H{h}_{slot}", "substitution")
                if pr is None:
                    continue
                res[(h, slot, arm, "correct")] = pr
                line = (f"  H={h:<3d} {slot:7s} {arm:10s} vs plain: correct RD {pr['D']:+.3f} "
                        f"CI[{pr['ci'][0]:+.3f},{pr['ci'][1]:+.3f}] b/c={pr['b']}/{pr['c']} p={pr['p']:.3g}")
                if sx:
                    line += (f" | subst {sx['D']:+.3f} CI[{sx['ci'][0]:+.3f},{sx['ci'][1]:+.3f}] p={sx['p']:.3g}")
                    res[(h, slot, arm, "substitution")] = sx
                print(line)
            pr = paired(idx, f"index_nok_H{h}_{slot}", f"index_H{h}_{slot}", "correct")
            if pr:
                res[(h, slot, "index_vs_nok", "correct")] = pr
                print(f"  H={h:<3d} {slot:7s} VALUE-ONLY  index vs index_nok: correct RD {pr['D']:+.3f} "
                      f"CI[{pr['ci'][0]:+.3f},{pr['ci'][1]:+.3f}] b/c={pr['b']}/{pr['c']} p={pr['p']:.3g}")
    return res


def verdict(res, hs):
    print("\n== pre-declared reading ==")
    # Q-A: index - plain at collapsed slots, averaged over H
    for slot in COLLAPSED:
        ds = [res[(h, slot, "index", "correct")]["D"] for h in hs if (h, slot, "index", "correct") in res]
        if not ds:
            continue
        qa = st.mean(ds) >= SESOI
        print(f"  Q-A index repairs shared-key collapse at {slot}: mean RD {st.mean(ds):+.3f} "
              f"-> {'YES (gain survives real hazard)' if qa else 'no (block inert)'}")
    # Q-B: index_nok substitution vs plain (does a stale view pull the model to the competitor?)
    subs = [res[(h, s, "index_nok", "substitution")]["D"] for h in hs for s in SLOTS
            if (h, s, "index_nok", "substitution") in res]
    if subs:
        qb = st.mean(subs) >= SESOI
        print(f"  Q-B stale index drives substitution vs plain: mean subst RD {st.mean(subs):+.3f} "
              f"-> {'YES (model reads INDEX over the raw stale line)' if qb else 'no'}")
    # Q-C: index vs index_nok (value, not presence)
    vc = [res[(h, s, "index_vs_nok", "correct")]["D"] for h in hs for s in SLOTS
          if (h, s, "index_vs_nok", "correct") in res]
    if vc:
        print(f"  Q-C block VALUE carries it (index vs index_nok): mean correct RD {st.mean(vc):+.3f}")


def load_jsonl(fn):
    rows = []
    for l in open(fn, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass          # tolerate a partial trailing line while the leg is still writing
    return rows


def read_pair(fn, tag):
    rows = load_jsonl(fn)
    if not rows:
        return rows, ""
    return rows, rows[0]["model"]


def main():
    files = sorted(glob.glob(os.path.join(HERE, "index_sk_scored_*.jsonl")))
    if not files:
        print("no index_sk_scored_*.jsonl yet"); sys.exit(1)
    for fn in files:
        tag = os.path.basename(fn).replace("index_sk_scored_", "").replace(".jsonl", "")
        rows, model = read_pair(fn, tag)
        for r in rows:
            r["arm"] = r["arm"]
        pubfn = os.path.join(HERE, f"position_scored_{tag}.jsonl")
        if not os.path.exists(pubfn):
            print(f"!! missing published plain arm {pubfn}")
            continue
        plain = load_jsonl(pubfn)
        for r in plain:
            r["arm"] = "plain"
        allrows = plain + rows
        print(f"\n#### {model}: index/index_nok {len(rows)} + plain(C7) {len(plain)} = {len(allrows)} calls, "
              f"{len({r['session_id'] for r in allrows})} sessions")
        hs = table(allrows, model)
        pooled(allrows, hs)
        res = contrasts(allrows, hs)
        verdict(res, hs)


if __name__ == "__main__":
    main()
