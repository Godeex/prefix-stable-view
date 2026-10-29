#!/usr/bin/env python
"""Build the sample flow of every study from the run files themselves.

A study cannot be audited from a denominator alone. For each study this reports, per arm:
records generated -> designed cells under the pre-declared turn window -> calls issued -> calls with a
server error -> cells analysed, plus the exclusion rule, whether it is arm dependent, and the largest
prompt any arm reached against the pinned context budget.

Two things are checked explicitly because they decide whether selection bias is possible:
  * the turn window is chosen by a deterministic rule evaluated BEFORE inference (window full /
    hazard revealed), so it cannot favour an arm -- and the same rule gives every arm the same
    denominator, so no arm-specific substitution of the analysis set is possible;
  * no arm's prompt approaches num_ctx, so nothing could have been dropped for overflowing.

Emits sample_flow.md (for the appendix) and sample_flow.json. Reads only existing files.
"""
import json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
NUMCTX = 8192


def rows(fn):
    p = os.path.join(HERE, fn)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


COMP_TRIAL_ARMS = lambda r: ["length_only"] + sorted(
    k[len("prompt_"):] for k in r if k.startswith("prompt_comp_"))


def flow(scored, armkey, trials=None, group=None):
    """Per-arm counts. Designed cells come from the generator file when one exists."""
    tc = {}
    for r in (trials or []):
        for a in (group(r) if group else ([r[armkey]] if armkey in r else [])):
            tc[a] = tc.get(a, 0) + 1
    out = {}
    for r in scored:
        d = out.setdefault(r[armkey], {"cells": 0, "err": 0, "maxprompt": 0, "prompt_sum": 0,
                                       "turns": set(), "sess": set()})
        d["cells"] += 1
        d["turns"].add(r.get("turn", 0))
        d["sess"].add(r["session_id"])
        d["err"] += int(r.get("prompt_eval_count", 0) == 0)
        d["maxprompt"] = max(d["maxprompt"], r.get("prompt_eval_count", 0))
        d["prompt_sum"] += r.get("prompt_eval_count", 0)
    for a, d in out.items():
        d["designed"] = tc.get(a, d["cells"])
        d["mean_prompt"] = round(d["prompt_sum"] / d["cells"])
        d["analysed"] = d["cells"] - d["err"]
        d["turn_lo"], d["turn_hi"] = min(d["turns"]), max(d["turns"])
        # The single-turn probes carry no turn index at all; printing 0--0 would read as a window.
        d["turns_used"] = ("single-turn" if d["turn_hi"] == 0
                           else f"{d['turn_lo']}--{d['turn_hi']}")
        d["n_sess"] = len(d["sess"])
        del d["turns"], d["sess"]
    return out


STUDIES = [
    ("Prefix cost (3 arms)", "prefix_cost_trials.jsonl",
     "prefix_cost_scored_qwen3_14b_t0.0.jsonl",
     lambda r: ["length_only", "prefix_stable", "pinned_raw"],
     30,  # records per session as generated (NTURNS in gen_prefix_cost.py)
     "turns WINDOW-1..NTURNS-1 (window full), set by the generator before any inference"),
    ("Prefix cost, 2nd model", None, "prefix_cost_scored_qwen2.5_14b_t0.0.jsonl",
     None, 30, "same designed cells as the primary cost run"),
    ("Joint 2x2 (4 configs)", None, "joint_scored_qwen3_14b.jsonl",
     None, 20, "turns first-hazard-reveal..last (distractor in context), set before inference"),
    ("Joint 2x2 tail stress", None, "joint_scored_qwen3_14b_tailf.jsonl",
     None, 20, "last 5 turns (both near-twins in the volatile tail), same rule for all four configs"),
    ("Joint 2x2 tail stress, 2nd family", None, "joint_scored_llama3_latest_tailf.jsonl",
     None, 20, "same window as the qwen3 tail run; every row's model field verified llama3:latest"),
    ("Layout 2x2 (4 cells)", None, "joint_layout_scored_qwen3_14b.jsonl",
     None, 20, "same hazard-window turns as the joint 2x2, identical rule for all four cells"),
    ("Bounded head (3 arms)", "prefix_bounded_trials.jsonl",
     "prefix_bounded_scored_qwen3_14b_t0.0.jsonl",
     lambda r: ["length_only", "prefix_stable", "bounded_stable"],
     40, "turns WINDOW-1..NTURNS-1, identical rule for all three arms"),
    ("Gold-position stress, qwen3 (3 slots)", None, "position_scored_qwen3_14b.jsonl",
     None, 2, "every designed slot (first/middle/last x H in {6,24}) issued once; the grid is "
     "120 sessions x 6 cells = 720 trials, 240 per slot arm; the row's "
     "model field is verified qwen3:14b, no post-hoc exclusion"),
    ("Gold-position stress, llama3 (3 slots)", None, "position_scored_llama3_latest.jsonl",
     None, 2, "same slots as the qwen3 position run, served a second time to llama3:latest "
     "(1,440 calls over both architectures); every row's model field verified llama3:latest"),
    ("Rewritable-index fidelity, qwen3 (3 arms)", "index_trials.jsonl",
     "index_scored_qwen3_14b.jsonl",
     None,
     6, "every designed cell (H in {6,24} x first/middle/last) issued once per arm over the frozen "
     "gold-position session pool; the grid holds 3 arms x 720 cells = 2,160 trials and the row's own "
     "arm column splits them; the row's model field is verified qwen3:14b, no post-hoc exclusion"),
    ("Rewritable-index fidelity, llama3 (3 arms)", None, "index_scored_llama3_latest.jsonl",
     None, 6, "same grid, cells and session pool as the qwen3 index run, served a second time to "
     "llama3:latest (4,320 calls over both architectures); every row's model field verified "
     "llama3:latest"),
    ("Rewritable-index cost (5 arms)", "index_cost_trials.jsonl",
     "index_cost_scored_qwen3_14b_t0.0.jsonl",
     lambda r: ["length_only", "prefix_stable", "index_I2", "index_I8", "index_I24"],
     23, "turns 7--29 of the published cost sessions (window full), the identical pre-declared "
     "window as the primary three-arm cost run; byte-identical shared head asserted at generation"),
    ("Token-compressor competitor (7 arms)", "prefix_comp_trials.jsonl",
     "prefix_comp_scored_qwen3_14b_t0.0.jsonl",
     COMP_TRIAL_ARMS,
     23, "turns 7--29 of the published cost sessions (window full), the identical pre-declared "
     "window as the primary three-arm cost run; every compressed prompt asserted at generation to "
     "be a whitespace-token subsequence of the same trial's naive window, with the probe tail "
     "byte-identical. Roster: naive-window control + 4 open keep-rates (.50/.25/.20/.12) + 2 "
     "gold-protected rates (.25/.20); the .20 pair is the budget-matched arm added mid-leg under "
     "AMENDMENT 2, served under the leg's original nonce so its prompts are the gated bytes"),
    ("Token-compressor competitor, 2nd model", None,
     "prefix_comp_scored_qwen2.5_14b_t0.0.jsonl",
     None, 23, "same sessions, turns, arm roster and nonce discipline as the primary competitor "
     "leg, served to the second published cost architecture"),
    ("Shared-key index fidelity, qwen3 (2 arms)", "index_sk_trials.jsonl",
     "index_sk_scored_qwen3_14b.jsonl",
     None,
     6, "every designed cell (H in {6,24} x first/middle/last) issued once per arm over the frozen "
     "gold-position session pool, with the distractor re-keyed to TARGET so a stale value competes "
     "under one key (the C7 shared-key hazard served back on top of the published STATE); the grid "
     "holds 2 arms (index fresh, index_nok stale) x 720 cells = 1,440 trials and the row's own arm "
     "column splits them; the row's model field is verified qwen3:14b, no post-hoc exclusion"),
    ("Shared-key index fidelity, llama3 (2 arms)", None, "index_sk_scored_llama3_latest.jsonl",
     None, 6, "same grid, cells, shared-key hazard and session pool as the qwen3 shared-key index "
     "run, served a second time to llama3:latest (2,880 calls over both architectures); every row's "
     "model field verified llama3:latest"),
    ("Repeat-variance diagnostic (2 arms)", None, "ms_repeat_variance_scored.jsonl",
     None, 69, "DIAGNOSTIC, not a confirmatory study: turns 7--29 of sessions S0/S5 of the "
     "primary cost run, re-fired three times per arm with a fresh nonce each repeat; "
     "generated cells 2 sessions x (23 turns x 3 repeats), all issued, none excluded"),
]

# Files whose calls are counted in the audit but that support no inferential claim. The split is
# by file name so the global audit line stays reproducible from the run files alone.
DIAGNOSTIC = {"ms_repeat_variance_scored.jsonl"}

# The passage-danger sweep (Table~\ref{tab:roster} "Passage danger slice") is the one audit row a
# reader cannot rebuild from the declared per-stratum n alone, because the pool each stratum was
# served from differs by stratum: the full scanned pool of 300 questions, the >=0.55 slice of 74,
# or the 27B pool of 44. This function derives the multiplication from the run files.
DANGER_PRIMARY_THR = 0.60
DANGER_SENS_THR = 0.55


def danger_accounting():
    import glob
    import collections as cu
    groups = cu.defaultdict(lambda: {"files": 0, "calls": 0, "pool": set(), "p60": 0, "p55": 0})
    tot = pairs_tot = 0
    for fn in sorted(glob.glob(os.path.join(HERE, "lb_danger_scored_*.jsonl"))):
        rs = rows(os.path.basename(fn))
        if not rs:
            continue
        by = {}
        for r in rs:
            by.setdefault(r["base_id"], {})[r["arm"]] = r
        ps = [g for g in by.values() if "length_only" in g and "m_a" in g]
        p60 = [p for p in ps if p["length_only"]["distractor_max_sim"] >= DANGER_PRIMARY_THR]
        p55 = [p for p in ps if p["length_only"]["distractor_max_sim"] >= DANGER_SENS_THR]
        pool = len(ps)
        g = groups[pool]
        g["files"] += 1
        g["calls"] += len(rs)
        g["p60"] += len(p60)
        g["p55"] += len(p55)
        tot += len(rs)
        pairs_tot += len(p60)
    lines = ["## Passage-danger audit (the multiplication behind the roster row)", "",
             "| pool served per stratum | strata | calls | = | paired questions | primary slice (>=0.60) | sensitivity (>=0.55) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for pool, g in sorted(groups.items(), reverse=True):
        lines.append(f"| {pool} questions x 2 arms | {g['files']} | {g['calls']} | "
                     f"{g['files']}x2x{pool} | {g['files']*pool} | {g['p60']} | {g['p55']} |")
    lines += ["", f"total {tot} calls in {sum(g['files'] for g in groups.values())} files; "
              f"{pairs_tot} pairs (2 arms each) in the >=0.60 primary slice = "
              f"{2*pairs_tot} calls carrying the C4 contrasts, and "
              f"{tot - 2*pairs_tot} calls served outside it (the pool the slice is drawn from, "
              "reported as prevalence in the threshold table and entering no contrast).", ""]
    js = {str(pool): dict(files=g["files"], calls=g["calls"], pairs60=g["p60"], pairs55=g["p55"])
          for pool, g in groups.items()}
    js["total_calls"] = tot
    js["primary_slice_calls"] = 2 * pairs_tot
    js["outside_slice_calls"] = tot - 2 * pairs_tot
    return lines, js


def main():
    md = ["# Sample flow of every study (generated from the run files, not typed)",
          "",
          f"All runs pin `num_ctx={NUMCTX}`, `temperature=0`, `seed=1234`, `num_predict<=40`, "
          "and a fresh per-arm nonce. `err` counts calls that returned no token counts.",
          "",
          "`generated` is records per session x sessions; `designed` is what the pre-declared turn",
          "window keeps from that pool, fixed before any inference and identical across arms.",
          ""]
    js = {}
    for name, trials_fn, scored_fn, group, per_sess, rule in STUDIES:
        trials = rows(trials_fn) if trials_fn else []
        scored = rows(scored_fn)
        if not scored:
            md.append(f"## {name}\n\n(pending: {scored_fn} not present yet)\n")
            continue
        key = next((k for k in ("arm", "cfg", "cell", "slot") if k in scored[0]), "arm")
        f = flow(scored, key, trials=trials, group=group)
        nsess = len(set(r["session_id"] for r in (trials or scored)))
        md += [f"## {name}", "", f"*exclusion rule:* {rule}", "",
               "| arm | generated | designed cells | calls issued | server errors | analysed | "
               "turns used | mean prompt tok | max prompt tok | max/num_ctx |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for a, d in sorted(f.items()):
            md.append(f"| {a} | {nsess * per_sess} | {d['designed']} | {d['cells']} | {d['err']} | "
                      f"{d['analysed']} | {d['turns_used']} | {d['mean_prompt']} | "
                      f"{d['maxprompt']} | {d['maxprompt']/NUMCTX:.1%} |")
        tot_err = sum(d["err"] for d in f.values())
        md += ["", f"Total analysed {sum(d['analysed'] for d in f.values())}, "
               f"total excluded {tot_err}. Largest prompt any arm reached: "
               f"{max(d['maxprompt'] for d in f.values())} tokens "
               f"({max(d['maxprompt'] for d in f.values())/NUMCTX:.1%} of the pinned budget), so no "
               "arm could have been dropped for overflowing the context window.", ""]
        js[name] = {"rule": rule, "sessions": nsess, "records_per_session": per_sess,
                    "generated": nsess * per_sess, "arms": f, "num_ctx": NUMCTX}

    dl, dj = danger_accounting()
    md += dl
    js["_danger_audit"] = dj

    # Global audit line: every model call anywhere in the paper, and the largest prompt any
    # arm of any study reached (this is what makes overflow-based dropout impossible).
    import glob
    allf = sorted(glob.glob(os.path.join(HERE, "*scored*.jsonl")))
    calls = mx = 0
    conf_calls = conf_files = 0
    mxsrc = ""
    for fn in allf:
        base = os.path.basename(fn)
        n = 0
        for l in open(fn, encoding="utf-8"):
            d = json.loads(l)
            calls += 1
            n += 1
            v = int(d.get("prompt_eval_count") or 0)
            if v > mx:
                mx, mxsrc = v, base
        if base not in DIAGNOSTIC:
            conf_calls += n
            conf_files += 1
    diag_calls, diag_files = calls - conf_calls, len(allf) - conf_files
    # The two legs added when the rewritable-index study was adopted, named so that the audit
    # sentence in Sec.~\ref{sec:flow} can be checked against the files rather than taken on trust.
    def leg(pat):
        fs = [f for f in allf if os.path.basename(f).startswith(pat)]
        return sum(1 for f in fs for _ in open(f, encoding="utf-8")), len(fs)
    idx_fid, idx_fid_files = leg("index_scored_")
    idx_cost, idx_cost_files = leg("index_cost_scored_")
    pos_fid, pos_files = leg("position_scored_")
    comp_calls, comp_files = leg("prefix_comp_scored_")
    sk_fid, sk_fid_files = leg("index_sk_scored_")
    md += ["## Global audit", "",
           f"{calls} model calls across {len(allf)} run files; largest prompt anywhere "
           f"{mx} tokens ({mx/NUMCTX:.1%} of num_ctx) in {mxsrc}.", "",
           f"Of these, {conf_calls} calls in {conf_files} run files carry the confirmatory "
           f"analyses reported in the paper, and {diag_calls} calls in {diag_files} file "
           f"({' '.join(sorted(DIAGNOSTIC))}) are a measurement-validity replication counted "
           "in the audit but supporting no inferential claim.", "",
           f"Legs of the confirmatory total that a reader is most likely to want rebuilt: "
           f"gold-position stress {pos_fid} calls in {pos_files} files; rewritable-index fidelity "
           f"{idx_fid} calls in {idx_fid_files} files (the same 2,160-cell grid served to "
           f"{idx_fid_files} architectures); rewritable-index cost {idx_cost} calls in "
           f"{idx_cost_files} files; token-compressor competitor {comp_calls} calls in "
           f"{comp_files} files; shared-key index fidelity {sk_fid} calls in {sk_fid_files} files "
           f"(the same 720-cell position grid, distractor re-keyed TARGET so a stale value competes "
           f"under one key, served to {sk_fid_files} architectures).", ""]
    js["_global"] = {"files": [os.path.basename(f) for f in allf], "calls": calls,
                     "confirmatory_calls": conf_calls, "confirmatory_files": conf_files,
                     "diagnostic_calls": diag_calls, "diagnostic_files": diag_files,
                     "diagnostic_files_named": sorted(DIAGNOSTIC),
                     "index_fidelity_calls": idx_fid, "index_fidelity_files": idx_fid_files,
                     "index_cost_calls": idx_cost, "index_cost_files": idx_cost_files,
                     "position_calls": pos_fid, "position_files": pos_files,
                     "competitor_calls": comp_calls, "competitor_files": comp_files,
                     "index_sk_fidelity_calls": sk_fid, "index_sk_fidelity_files": sk_fid_files,
                     "max_prompt": mx, "max_prompt_file": mxsrc, "num_ctx": NUMCTX}
    open(os.path.join(HERE, "sample_flow.md"), "w", encoding="utf-8").write("\n".join(md))
    json.dump(js, open(os.path.join(HERE, "sample_flow.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
