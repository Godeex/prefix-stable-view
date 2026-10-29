#!/usr/bin/env python
"""Analysis: does a real external content rule (LLMLingua-2 pruning) buy what the view policy buys?

Reads the competitor leg (prefix_comp_scored_<model>.jsonl: the comp_open*/comp_keep* roster plus a
same-run length_only control) together with the published cost leg (prefix_cost_scored_<model>.jsonl:
length_only, prefix_stable, pinned_raw) on the identical 552 sessions/turns, and reports the three
coordinates the paper is about -- B (measured prompt tokens), R (recompute tokens), F (correct) --
per arm. The arm roster is read off the run file, not hard-coded: an arm that was not served is not
analysed, and an arm added mid-leg is analysed by the same estimators as the arms that were declared.

Models are never pooled: the primary leg (CP_FILE, qwen3:14b) carries the pre-declared verdicts, and
every other competitor run file is reported as its own replication block with the same estimators.
That mirrors the published cost leg, where qwen2.5:14b is a separate row rather than extra n.

Estimators are the ones the earlier legs fixed, unchanged: paired within (session, turn), mean
paired difference with a session-cluster bootstrap (6,000 reps, seed 1234) for the continuous
outcomes, exact McNemar on within-session discordant pairs plus a Wilson CI and a bootstrap risk
difference for `correct`, and the same pre-declared SESOI = 0.03 actionability floor.

The pre-declared reading is fixed in run_prefix_comp.py before any call; it is echoed here verbatim
and judged on the measured numbers, whichever way each item falls. Two disclosures are printed with
it: the C8-P3 mis-specification correction (AMEND) and the addition of the budget-matched arm after
the bracketing arms had been measured (AMEND2).

Env: CP_REPS (6000), CP_FILE (primary competitor run file), CP_PUB (its published-cost counterpart).
"""
import glob, json, math, os, re, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SESOI = 0.03
REPS = int(os.environ.get("CP_REPS", "6000"))
# Display order only -- which arms exist is decided by the run file.
COMP_ORDER = ("comp_open50", "comp_open25", "comp_open20", "comp_open12",
              "comp_keep25", "comp_keep20")
PUB_ARMS = ("prefix_stable", "pinned_raw")
FIELD = "correct"
PRIMARY = os.environ.get("CP_FILE", "prefix_comp_scored_qwen3_14b_t0.0.jsonl")
PRIMARY_PUB = os.environ.get("CP_PUB", "prefix_cost_scored_qwen3_14b_t0.0.jsonl")


def load(files):
    rows = []
    for fn in files:
        for l in open(os.path.join(HERE, fn), encoding="utf-8"):
            rows.append(json.loads(l))
    return rows


def cost_pair(comp_file):
    """The published-cost run file for the same model and temperature, if it exists."""
    m = re.match(r"prefix_comp_scored_(.+_t[0-9.]+)\.jsonl$", comp_file)
    if not m:
        return None
    cand = f"prefix_cost_scored_{m.group(1)}.jsonl"
    return cand if os.path.exists(os.path.join(HERE, cand)) else None


def cells(rows):
    d = defaultdict(dict)
    for r in rows:
        d[(r["session_id"], r["turn"])][r["arm"]] = r
    return d


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def mcnemar(a, b):
    """Exact two-sided binomial on discordant pairs over the cells where both arms exist."""
    n01 = sum(1 for k, v in a.items() if v[FIELD] == 1 and b.get(k, {}).get(FIELD) == 0)
    n10 = sum(1 for k, v in b.items() if v[FIELD] == 1 and a.get(k, {}).get(FIELD) == 0)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    return p, n01, n10


def paired_boot(pairs, a, b, field, B=REPS, seed=1234):
    vals, sess = [], defaultdict(list)
    for (sid, turn), v in pairs.items():
        if a in v and b in v and field in v[a] and field in v[b]:
            dv = v[a][field] - v[b][field]
            vals.append(dv)
            sess[sid].append(dv)
    if not vals:
        return None
    D = float(np.mean(vals))
    keys = list(sess)
    rng = np.random.default_rng(seed)
    boots = [float(np.mean([x for k in [keys[i] for i in rng.integers(0, len(keys), len(keys))]
                            for x in sess[k]])) for _ in range(B)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(D=D, ci=(float(lo), float(hi)), p=float(np.mean([bb <= 0 for bb in boots])),
                n=len(vals), n_sess=len(keys))


def arm_stats(rows):
    n = len(rows)
    return {
        "n": n,
        "acc": sum(r[FIELD] for r in rows) / n if n else 0.0,
        "gold_in_ctx": sum(r.get("gold_in_context", 1) for r in rows) / n if n else 0.0,
        "prompt_mean": float(np.mean([r["prompt_eval_count"] for r in rows])),
        "prompt_med": float(np.median([r["prompt_eval_count"] for r in rows])),
        "cached_mean": float(np.mean([r["prompt_eval_cached_count"] for r in rows])),
        "recomp_mean": float(np.mean([r["recompute"] for r in rows])),
        "recomp_med": float(np.median([r["recompute"] for r in rows])),
        "prefill_ms": float(np.mean([r["prompt_eval_duration_ms"] for r in rows])),
    }


AMEND = (
    "AMENDMENT (post-hoc, written after ~17% of the leg had been scored -- 92-115 of the 552 turns "
    "per competitor arm -- and before the remaining calls; the pre-declared items above are printed "
    "as originally written): C8-P3 as pre-declared compares ABSOLUTE recompute against the naive "
    "window, but a pruned prompt is smaller, so it recomputes fewer tokens mechanically whatever its "
    "byte stability. That contrast is confounded by B and is not the mechanism claim. The "
    "mechanism claim is about the SHARE of the prompt that must be recomputed (R/B) and about the "
    "comparison at matched budget, so both are reported below; the mis-specification is disclosed "
    "here, in the table caption and in the claim ledger.")

AMEND2 = (
    "AMENDMENT 2 (post-hoc, written after 506-529 of the 552 turns had been served per competitor "
    "arm and before any call of the arms it adds; the pre-declared items above are printed as "
    "originally written): the roster as first shipped only BRACKETS the view's prompt budget "
    "(.25 measured above it, .12 far below), while the content rule is evaluated AT "
    "matched budget. comp_open20/comp_keep20 were therefore added, their keep-rate chosen by "
    "matched_budget.py from a fit of measured prompt tokens on kept words over the arms served so "
    "far. The rate choice is informed by partial observation of this leg; whether the resulting arm "
    "is actually matched is decided below by the paired bootstrap on B, not by the fit.")


def table_block(pairs, arms, comp_names):
    print(f"{'arm':14s} {'n':>5s} {'acc':>6s} {'gold@ctx':>9s} {'B mean':>7s} {'B med':>6s} "
          f"{'R mean':>7s} {'R med':>6s} {'hit%':>5s} {'prefill ms':>10s}")
    for a in comp_names + ("length_only",) + PUB_ARMS:
        if a not in arms:
            continue
        s = arms[a]
        lo, hi = wilson(round(s["acc"] * s["n"]), s["n"])
        print(f"{a:14s} {s['n']:5d} {s['acc']:6.3f} {s['gold_in_ctx']:9.3f} {s['prompt_mean']:7.0f} "
              f"{s['prompt_med']:6.0f} {s['recomp_mean']:7.0f} {s['recomp_med']:6.0f} "
              f"{100*s['cached_mean']/max(s['prompt_mean'],1):4.1f}% {s['prefill_ms']:10.1f}")


def paired_block(pairs, arms, comp_names):
    print("\n== paired session-cluster bootstrap (competitor vs same-run naive window) ==")
    res = {}
    for a in comp_names:
        for field, label in (("prompt_eval_count", "B"), ("recompute", "R")):
            r = paired_boot(pairs, a, "length_only", field)
            if r:
                sig = "comp SMALLER" if r["ci"][1] < 0 else ("comp LARGER" if r["ci"][0] > 0 else "n.s.")
                print(f"  {label}  {a:13s} - length_only = {r['D']:8.1f} tok  "
                      f"95%CI[{r['ci'][0]:7.1f},{r['ci'][1]:7.1f}] P(<=0)={r['p']:.2g} -> {sig}")
                res[(a, label)] = r
        r = paired_boot(pairs, a, "length_only", FIELD)
        p, n01, n10 = mcnemar({k: v[a] for k, v in pairs.items() if a in v},
                              {k: v["length_only"] for k, v in pairs.items() if "length_only" in v})
        accs = {k: v[a][FIELD] for k, v in pairs.items() if a in v}
        lo, hi = wilson(sum(accs.values()), len(accs))
        print(f"  F   {a:13s} - length_only = {r['D']:+.3f}  95%CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
              f"McNemar p={p:.3g} (b={n01} c={n10})  acc CI[{lo:.3f},{hi:.3f}]")
        res[(a, "F")] = r
    return res


def view_block(pairs, comp_names):
    print("\n== competitor vs the append-only view (prefix_stable, published cost leg) ==")
    res = {}
    for a in comp_names:
        out = []
        for b, field, lbl in (("prefix_stable", "prompt_eval_count", "B"),
                              ("prefix_stable", "recompute", "R")):
            r = paired_boot(pairs, a, b, field)
            if r:
                out.append(f"{lbl} {r['D']:+8.1f} CI[{r['ci'][0]:.1f},{r['ci'][1]:.1f}]")
                res[(a, lbl)] = r
        print(f"  {a:13s} " + "  ".join(out))
    return res


def verdict_block(pairs, arms, res, comp_names):
    ps = arms.get("prefix_stable")
    v = {}
    print("\n== pre-declared reading (fixed in run_prefix_comp.py before any call) ==")
    # C8-P1 bloat: is any competitor arm at or below the append-only view's measured budget?
    if ps:
        small = [a for a in comp_names if a in arms and arms[a]["prompt_mean"] <= ps["prompt_mean"]]
        v["P1_below_view_budget"] = small
        print(f"  C8-P1 BLOAT: prefix_stable B = {ps['prompt_mean']:.0f} tok; comp B = "
              + ", ".join(f"{a} {arms[a]['prompt_mean']:.0f}" for a in comp_names if a in arms)
              + f" -> {'AT/BELOW the view budget: ' + ','.join(small) if small else 'NO comp arm reaches the view budget'}")
    # C8-P2 fidelity of the benefit-of-the-doubt arm (pre-declared as comp_keep25)
    protected = [a for a in comp_names if a.startswith("comp_keep")]
    for a in protected:
        r = res.get((a, "F"))
        if not r:
            continue
        ok = r["ci"][0] >= -SESOI
        v.setdefault(a, {})["P2_fidelity_preserved"] = bool(ok)
        print(f"  C8-P2 FIDELITY (gold-protected arm): {a} - length_only = {r['D']:+.3f} "
              f"95%CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}], floor -{SESOI} -> "
              f"{'fidelity preserved' if ok else 'FIDELITY LOST even with the answer row force-kept'}"
              + ("" if a == "comp_keep25" else "   [arm added by AMENDMENT 2]"))
    for a in comp_names:
        if a.startswith("comp_open"):
            rr = res.get((a, "F"))
            if rr:
                print(f"      open arm: {a} - length_only = {rr['D']:+.3f} "
                      f"CI[{rr['ci'][0]:+.3f},{rr['ci'][1]:+.3f}], gold in context "
                      f"{arms[a]['gold_in_ctx']:.3f}")
    # C8-P3 the falsifiable prediction: compression must not buy recompute
    for a in comp_names:
        r = res.get((a, "R"))
        if r:
            best = None
            rb_view = res.get((a, "R_vs_view"))
            if ps and rb_view is not None:
                best = rb_view["ci"][1] < 0
            v[a] = {"vs_naive": [r["D"], *r["ci"]],
                    "naive_prediction_holds": bool(r["ci"][0] >= 0)}
            print(f"  C8-P3 RECOMPUTE: {a} - length_only = {r['D']:+8.1f} tok "
                  f"CI[{r['ci'][0]:.1f},{r['ci'][1]:.1f}] -> "
                  f"{'NO recompute win from pruning (prediction holds)' if r['ci'][0] >= 0 else 'pruning DID reduce recompute vs the naive window -- see CI'}"
                  + (f"; vs prefix_stable {'BEATS the view' if best else 'does not beat the view'}" if best is not None else ""))
    print(AMEND)
    if any(a in comp_names for a in ("comp_open20", "comp_keep20")):
        print(AMEND2)
    if ps:
        shares = {a: arms[a]["recomp_mean"] / max(arms[a]["prompt_mean"], 1)
                  for a in arms if a in comp_names}
        print("  reuse share R/B per arm: " + ", ".join(f"{a} {shares[a]:.3f}" for a in comp_names)
              + f" | length_only {arms['length_only']['recomp_mean']/arms['length_only']['prompt_mean']:.3f}"
              + f" | prefix_stable {ps['recomp_mean']/max(ps['prompt_mean'],1):.3f}")
        v["reuse_share"] = {**shares, "length_only": arms["length_only"]["recomp_mean"] / arms["length_only"]["prompt_mean"],
                            "prefix_stable": ps["recomp_mean"] / max(ps["prompt_mean"], 1)}
        v["matched_budget"] = {}
        for a in comp_names:
            rb, rr = (paired_boot(pairs, a, "prefix_stable", "prompt_eval_count"),
                      paired_boot(pairs, a, "prefix_stable", "recompute"))
            if not (rb and rr):
                continue
            # the budget relation to the view is MEASURED, three-way, from the paired bootstrap on B
            if rb["ci"][1] < 0:
                rel = "SMALLER"
            elif rb["ci"][0] > 0:
                rel = "LARGER"
            else:
                rel = "MATCHED"
            rf = paired_boot(pairs, a, "prefix_stable", FIELD)
            rec = {"relation_to_view_budget": rel, "dB": [rb["D"], *rb["ci"]],
                   "dR": [rr["D"], *rr["ci"]],
                   "dF": [rf["D"], *rf["ci"]] if rf else None}
            v["matched_budget"][a] = rec
            note = ("THE VIEW WINS ON R WITH A SMALLER BUDGET" if rel == "SMALLER" and rr["D"] > 0 else
                    "the smaller pruned prompt did recompute less" if rel == "SMALLER" else
                    "AT THE SAME BUDGET THE VIEW RECOMPUTES LESS" if rel == "MATCHED" and rr["D"] > 0 else
                    "at the same budget the pruned arm recomputes no more" if rel == "MATCHED" else
                    "larger budget than the view: not the matched comparison")
            print(f"  matched-budget check at {a}: B {rb['D']:+7.1f} CI[{rb['ci'][0]:.1f},"
                  f"{rb['ci'][1]:.1f}] -> {rel} than the view; R {rr['D']:+8.1f} CI[{rr['ci'][0]:.1f},"
                  f"{rr['ci'][1]:.1f}]"
                  + (f"; F {rf['D']:+.3f} CI[{rf['ci'][0]:+.3f},{rf['ci'][1]:+.3f}]" if rf else "")
                  + f" -> {note}"
                  + ("" if a not in ("comp_open20", "comp_keep20") else "   [AMENDMENT 2 arm]"))
    # C8-P4 Pareto: weak dominance either way at matched-or-smaller budget
    if ps:
        dom = []
        for a in comp_names:
            if a not in arms:
                continue
            rb = paired_boot(pairs, a, "prefix_stable", "prompt_eval_count")
            rr = paired_boot(pairs, a, "prefix_stable", "recompute")
            rf = paired_boot(pairs, a, "prefix_stable", FIELD)
            if not (rb and rr and rf):
                continue
            comp_dom = (rb["ci"][1] <= 0 and rr["ci"][1] <= 0 and rf["ci"][0] >= 0)
            view_dom = (rb["ci"][0] >= 0 and rr["ci"][0] >= 0 and rf["ci"][1] <= 0)
            dom.append((a, comp_dom, view_dom))
            v[a] = {**v.get(a, {}), "vs_view": [rb["D"], rr["D"], rf["D"]],
                    "competitor_dominates": bool(comp_dom), "view_dominates": bool(view_dom)}
            print(f"  C8-P4 {a:13s} vs prefix_stable: B {rb['D']:+7.1f} R {rr['D']:+8.1f} "
                  f"F {rf['D']:+.3f} -> "
                  f"{'COMPETITOR WEAKLY DOMINATES' if comp_dom else ('view weakly dominates' if view_dom else 'no dominance (trade-off)')}")
        if dom and all(v2 for _, _, v2 in dom):
            print("  C8-P4 verdict: the append-only view weakly dominates every competitor arm.")
            v["P4_view_dominates_all"] = True
    return v


def leg(comp_file, pub_file, primary):
    comp_rows = load([comp_file])
    pub_rows = load([pub_file]) if pub_file else []
    if not comp_rows:
        print(f"no competitor-leg rows in {comp_file}"); return None
    rows = comp_rows + pub_rows
    pairs = cells(rows)
    served = [r["arm"] for r in comp_rows if r["arm"] != "length_only"]
    order = {a: i for i, a in enumerate(COMP_ORDER)}
    comp_names = tuple(sorted(dict.fromkeys(served), key=lambda a: order.get(a, 99)))
    arms = {}
    for a in list(comp_names) + ["length_only"] + list(PUB_ARMS):
        sub = [v[a] for v in pairs.values() if a in v]
        if sub:
            arms[a] = arm_stats(sub)

    tag = "primary leg" if primary else "replication"
    print(f"\n######## {comp_rows[0]['model']}  temp={comp_rows[0]['temp']}  "
          f"cells={len(pairs)}  sessions={len(set(r['session_id'] for r in rows))}  [{tag}]")
    if not pub_rows:
        print(f"  (no published-cost counterpart for this model: {comp_file} analysed against its "
              "own same-run length_only control only)")
    print()
    table_block(pairs, arms, comp_names)
    res = paired_block(pairs, arms, comp_names)
    rv = view_block(pairs, comp_names)
    for k, val in rv.items():
        res[(k[0], "R_vs_view")] = val
    v = verdict_block(pairs, arms, res, comp_names) if pub_rows else {}
    return {"model": comp_rows[0]["model"], "temp": comp_rows[0]["temp"],
            "comp_file": comp_file, "cost_file": pub_file, "n_cells": len(pairs),
            "n_sessions": len(set(r["session_id"] for r in rows)),
            "arms": arms, "verdicts": v, "primary": primary}


def main():
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(HERE, "prefix_comp_scored_*.jsonl")))
    if not files:
        print("no competitor-leg rows yet"); return
    if PRIMARY not in files:
        sys.exit(f"primary competitor file {PRIMARY} not present (found {files})")
    out = {}
    for fn in files:                                   # primary first, then replications
        pub = PRIMARY_PUB if fn == PRIMARY else cost_pair(fn)
        res = leg(fn, pub, fn == PRIMARY)
        if res:
            out[f"{res['model']}|{res['comp_file']}"] = res
    js = {"sesoi": SESOI, "reps": REPS, "legs": out}
    first = next(r for r in out.values() if r["primary"])
    js["arms"] = first["arms"]                          # back-compat: primary model at top level
    js["model"] = first["model"]
    js["temp"] = first["temp"]
    js["n_cells"] = first["n_cells"]
    json.dump(js, open(os.path.join(HERE, "comp_analysis_summary.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
