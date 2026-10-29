#!/usr/bin/env python
"""Where does a keep-rate have to sit for the pruner to be at the APPEND-ONLY VIEW'S budget?

The external content rule must be evaluated "at matched prompt budget". The rates we first
shipped (.50/.25/.12) bracket that budget but do not hit it: at .25 the pruned prompt is still
larger than the view's and at .12 it is far smaller. This script computes the rate that lands on the
view's measured budget, from the run files only -- no number here is typed by hand.

Two estimators are computed and both are reported, because they disagree and the disagreement is
itself a finding:

  * NAIVE PROXY -- a pruned prompt's tokens are (words kept) x (tokens-per-word of the naive
    window). Available before any competitor call is served, but it is wrong here: the pruner keeps
    the high-information words, which are the token-dense ones, so pruned text costs MORE tokens per
    kept word than the window it was pruned from. This proxy targets .27.
  * MEASURED FIT -- regress the arms' server-measured mean prompt tokens on their mean kept-word
    count, over the open arms served in this leg so far. This is the estimator used to pick the
    rate. It is fit on PARTIAL serving of the very leg the rate is chosen for, which is disclosed
    here, in run_prefix_comp.py's pre-declared reading and in the paper.

The fit is an ESTIMATE used only to pick the rate. What the arm actually costs is measured from the
served leg and reported by analyze_prefix_comp.py: this file never claims the match was achieved.
After the leg finished the script was run again, and it now reports two different things side by
side: the rate the fit would pick on the COMPLETE leg, and the arm the leg actually served that
lands nearest the view's measured budget. Which arm is "the matched one" in the paper is decided by
the second, never by the first.

Writes matched_budget.json. Env: MB_STEP (0.01), MB_MAX (0.50).
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COST_TRIALS = os.path.join(HERE, "prefix_cost_trials.jsonl")
COST_SCORED = os.path.join(HERE, "prefix_cost_scored_qwen3_14b_t0.0.jsonl")
COMP_TRIALS = os.path.join(HERE, "prefix_comp_trials.jsonl")
COMP_SCORED = os.path.join(HERE, "prefix_comp_scored_qwen3_14b_t0.0.jsonl")
OUT = os.path.join(HERE, "matched_budget.json")
STEP = float(os.environ.get("MB_STEP", "0.01"))
RMAX = float(os.environ.get("MB_MAX", "0.50"))
VIEW = "prefix_stable"
NAIVE = "length_only"


def rate_of(arm):
    """comp_open20 / comp_keep20 -> 0.20: the rate is in the arm's name, put there by the generator."""
    return int("".join(ch for ch in arm if ch.isdigit())) / 100.0


def main():
    scored = defaultdict(dict)
    for l in open(COST_SCORED, encoding="utf-8"):
        r = json.loads(l)
        scored[(r["session_id"], r["turn"])][r["arm"]] = r
    trials = {(r["session_id"], r["turn"]): r
              for r in (json.loads(l) for l in open(COST_TRIALS, encoding="utf-8"))}
    comp = {f"{r['session_id']}|{r['turn']}": r
            for r in (json.loads(l) for l in open(COMP_TRIALS, encoding="utf-8"))}

    # ---- the target: the published view's measured budget on the same sessions/turns ----
    b_view = np.array([s[VIEW]["prompt_eval_count"] for s in scored.values() if VIEW in s], dtype=float)
    ratio, b_naive = [], []
    for k, s in scored.items():
        if NAIVE in s and k in trials:
            w = len(trials[k][f"prompt_{NAIVE}"].split())
            if w:
                ratio.append(s[NAIVE]["prompt_eval_count"] / w)
                b_naive.append(s[NAIVE]["prompt_eval_count"])
    rv = float(np.mean(ratio))

    # ---- estimator A: naive proxy, tokens-per-word of the window being pruned ----
    n_words = np.array([x["n_words"] for x in comp.values()], dtype=float)
    tail_words = np.array([len(x["probe_tail"].split()) for x in comp.values()], dtype=float)
    grid = [round(i * STEP, 4) for i in range(1, int(RMAX / STEP) + 1)]
    naive_pred = [{"rate": r, "pred_mean_B": float(np.mean((r * n_words + tail_words) * rv))}
                  for r in grid]
    naive_best = min(naive_pred, key=lambda p: abs(p["pred_mean_B"] - b_view.mean()))

    # ---- estimator B: measured fit over the open arms already served in this leg ----
    served = defaultdict(list)
    for l in open(COMP_SCORED, encoding="utf-8"):
        d = json.loads(l)
        served[d["arm"]].append(d)
    pts, used = [], []
    for a in sorted(served):
        if not a.startswith("comp_open"):
            continue
        w = np.array([comp[f"{d['session_id']}|{d['turn']}"]["words_" + a] for d in served[a]], dtype=float)
        b = np.array([d["prompt_eval_count"] for d in served[a]], dtype=float)
        pts.append((w.mean(), b.mean()))
        used.append({"arm": a, "calls": len(served[a]), "mean_kept_words": float(w.mean()),
                     "mean_measured_B": float(b.mean()),
                     "tokens_per_kept_word": float((b / np.maximum(w, 1)).mean())})
    if len(pts) < 2:
        raise SystemExit("need at least two served open arms to fit the budget -- do not guess a rate")
    X, Y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    slope, intercept = np.polyfit(X, Y, 1)
    resid = Y - (intercept + slope * X)
    fit_pred = [{"rate": r, "pred_mean_B": float(np.mean(intercept + slope * np.round(r * n_words)))}
                for r in grid]
    fit_best = min(fit_pred, key=lambda p: abs(p["pred_mean_B"] - b_view.mean()))

    arms = {}
    for a in sorted({k[len("words_"):] for x in comp.values() for k in x if k.startswith("words_")}):
        w = np.array([x["words_" + a] for x in comp.values()], dtype=float)
        arms[a] = {"median_words": float(np.median(w)),
                   "pred_mean_B_naive_proxy": float(np.mean((w + tail_words) * rv)),
                   "pred_mean_B_measured_fit": float(np.mean(intercept + slope * w)),
                   "measured_mean_B": next((u["mean_measured_B"] for u in used if u["arm"] == a), None),
                   "gold_survives": int(sum(x["survives_" + a] for x in comp.values()))}

    # Which served open arm actually lands nearest the view's measured budget -- a measurement, not
    # a prediction. The fit's own pick is kept beside it so a one-step disagreement stays visible.
    landed = min(used, key=lambda u: abs(u["mean_measured_B"] - float(b_view.mean())))
    o = {
        "purpose": "keep-rate for the budget-matched competitor arm",
        "n_trials": len(comp),
        "view_budget_B": {"arm": VIEW, "mean": float(b_view.mean()), "median": float(np.median(b_view)),
                          "p25": float(np.percentile(b_view, 25)), "p75": float(np.percentile(b_view, 75)),
                          "n_calls": int(len(b_view))},
        "naive_window": {"mean_B": float(np.mean(b_naive)), "tokens_per_word_mean": rv},
        "raw_history_words": {"median": float(np.median(n_words)), "mean": float(np.mean(n_words))},
        "estimator_naive_proxy": {"target_rate": naive_best["rate"],
                                  "pred_mean_B": naive_best["pred_mean_B"],
                                  "gap_vs_view": naive_best["pred_mean_B"] - float(b_view.mean()),
                                  "why_rejected": ("assumes pruned text has the window's tokens-per-"
                                                   "kept-word; the pruner keeps token-dense words")},
        "estimator_measured_fit": {"fit_points": used, "intercept": float(intercept),
                                   "slope_tokens_per_kept_word": float(slope),
                                   "max_abs_residual_tokens": float(np.max(np.abs(resid))),
                                   "target_rate": fit_best["rate"],
                                   "pred_mean_B": fit_best["pred_mean_B"],
                                   "gap_vs_view": fit_best["pred_mean_B"] - float(b_view.mean())},
        "served_matched_arm": {
            "arm": landed["arm"], "rate": rate_of(landed["arm"]),
            "measured_mean_B": landed["mean_measured_B"],
            "gap_vs_view": landed["mean_measured_B"] - float(b_view.mean()),
            "calls": landed["calls"],
            "fit_rate_on_complete_leg": fit_best["rate"],
            "fit_pick_was_served": (f"comp_open{int(round(fit_best['rate']*100))}" == landed["arm"]),
            "how_matched_is_decided": ("by the measured prompt tokens of the served arms and the "
                                       "paired bootstrap CI in analyze_prefix_comp.py; the fit pick "
                                       "above is shown only to keep a disagreement visible")},
        "arms_served_by_this_roster": [u["arm"] for u in used],
        "nearest_rates_on_the_fit": sorted(fit_pred, key=lambda p: abs(p["pred_mean_B"] - b_view.mean()))[:5],
        "arms_and_predictions": arms,
        "honest_boundary": ("the rate is CHOSEN from this fit; the budget the arm actually costs is "
                            "measured from the served leg (analyze_prefix_comp.py). The fit is on "
                            "partial serving of the same leg, which is a disclosed post-hoc element "
                            "of the design, not a blind prediction."),
        "inputs": [os.path.basename(COST_TRIALS), os.path.basename(COST_SCORED),
                   os.path.basename(COMP_TRIALS), os.path.basename(COMP_SCORED)],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2, ensure_ascii=False)

    print(f"target: view {VIEW} mean B {o['view_budget_B']['mean']:.1f} tok "
          f"(median {o['view_budget_B']['median']:.0f}, n={o['view_budget_B']['n_calls']} calls)")
    print(f"naive window mean B {o['naive_window']['mean_B']:.1f}, tokens/word {rv:.3f}; "
          f"raw history median {o['raw_history_words']['median']:.0f} words")
    print("\nestimator A (naive proxy):  target rate %.2f -> pred B %.1f (gap %+.1f)"
          % (naive_best["rate"], naive_best["pred_mean_B"], o["estimator_naive_proxy"]["gap_vs_view"]))
    print("estimator B (measured fit): B = %.1f + %.3f * kept_words  (max residual %.1f tok)"
          % (intercept, slope, o["estimator_measured_fit"]["max_abs_residual_tokens"]))
    for u in used:
        print("    %-13s %4d calls  mean kept %6.1f words -> mean measured B %7.1f  (%.3f tok/word)"
              % (u["arm"], u["calls"], u["mean_kept_words"], u["mean_measured_B"], u["tokens_per_kept_word"]))
    print("    target rate %.2f -> pred B %.1f (gap %+.1f)"
          % (fit_best["rate"], fit_best["pred_mean_B"], o["estimator_measured_fit"]["gap_vs_view"]))
    print("\nper arm: " + ", ".join(f"{a} pred {v['pred_mean_B_measured_fit']:.0f}"
                                   f"{'' if v['measured_mean_B'] is None else ' obs %.0f' % v['measured_mean_B']}"
                                   for a, v in arms.items()))
    sm = o["served_matched_arm"]
    print(f"-> arm nearest the view's measured budget: {sm['arm']} at mean B "
          f"{sm['measured_mean_B']:.1f} ({sm['gap_vs_view']:+.1f} tok vs the view, "
          f"{sm['calls']} calls); the fit on the complete leg would pick rate "
          f"{sm['fit_rate_on_complete_leg']:.2f}"
          + (" (the same arm)" if sm["fit_pick_was_served"] else " (a different arm: the leg "
             "reports the measured one, not the fit's pick)"))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
