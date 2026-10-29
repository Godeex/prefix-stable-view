#!/usr/bin/env python
"""Put the cost of the maintenance operator itself into the ledger.

Algorithm 1's fold step is a deterministic operator, not a model call: the reference view
(co/summary_view.py) appends (key, value) rows, and the bounded-head variant in
gen_prefix_bounded.py rolls the oldest rows into one SUMMARY line. This measures what that costs so
the paper can state the total per-turn maintenance cost instead of only the prefill saving:

  * CPU time per fold, and per turn of the whole append/fold pipeline;
  * tokens the fold adds to the prompt (the surcharge the reader sees in the cost table);
  * what an LLM-based fold WOULD cost, expressed as the measured prefill saving it has to beat,
    rather than as a number we did not measure.

Reads the run files; no model calls.
"""
import json, os, statistics as st, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from co.summary_view import SummaryView


def bench_reference_fold(n=4000):
    sv = SummaryView()
    t = time.perf_counter()
    for i in range(n):
        sv.fold({f"OBS{i}": f"value{i}"})
    dt = (time.perf_counter() - t) / n
    return dt * 1e6


def bench_bounded_fold(trials_fn="prefix_bounded_trials.jsonl"):
    """Replay the deterministic rollup the bounded arm uses and time it, on the real record stream."""
    rows = [json.loads(l) for l in open(os.path.join(HERE, trials_fn), encoding="utf-8")]
    by_sess = {}
    for r in rows:
        by_sess.setdefault(r["session_id"], []).append(r)
    times, folds, surcharges = [], 0, []
    for sid, rs in by_sess.items():
        head = []
        prev = None
        for r in sorted(rs, key=lambda x: x["turn"]):
            head.append(f"{r['key']}={r['gold']}")
            t0 = time.perf_counter()
            if len(head) > 10:                      # H_MAX as generated
                oldest, keep = head[:len(head) - 4], head[len(head) - 4:]
                summary = ("SUMMARY: " + ", ".join(oldest)
                           + f". (folded {len(oldest)} earlier observations)")
                head = [summary] + keep
                folds += 1
            times.append((time.perf_counter() - t0) * 1e6)
            cur = r.get("prompt_bounded_stable", "")
            if prev and cur:
                surcharges.append(len(cur.split()) - len(prev.split()))
            prev = cur
    return times, folds, surcharges


def main():
    ref_us = bench_reference_fold()
    print(f"reference fold (append one key=value row): {ref_us:.2f} us/op over 4000 ops, "
          f"0 model calls, 0 output tokens")

    times, folds, sur = bench_bounded_fold()
    rows = [json.loads(l) for l in open(os.path.join(HERE, "prefix_bounded_scored_qwen3_14b_t0.0.jsonl"),
                                        encoding="utf-8")]
    bd = [r for r in rows if r["arm"] == "bounded_stable"]
    fold_rows = [r for r in bd if r.get("fold_event")]
    app_rows = [r for r in bd if not r.get("fold_event")]
    print(f"bounded-head fold replay: {folds} fold events over {len(times)} turns, "
          f"mean {st.mean(times):.1f} us, p95 {sorted(times)[int(.95*len(times))]:.1f} us, "
          f"max {max(times):.1f} us; total operator time "
          f"{sum(times)/1e6:.3f} s for the whole {len(times)}-turn stream")

    fa = st.mean(r["recompute"] for r in fold_rows)
    aa = st.mean(r["recompute"] for r in app_rows)
    print(f"measured prompt effect of a fold turn: recompute {aa:.0f} tok on append turns vs "
          f"{fa:.0f} tok on fold turns (delta {fa-aa:+.0f} tok), "
          f"word-count delta on fold turns {st.mean(sur):+.1f}")
    mean_recomp = st.mean(r["recompute"] for r in bd)
    naive = st.mean(r["recompute"] for r in rows if r["arm"] == "length_only")
    saved = naive - mean_recomp
    print(f"tokens saved per turn by the bounded policy: {saved:.0f} "
          f"(naive {naive:.0f} - bounded {mean_recomp:.0f})")
    print(f"break-even for a hypothetical LLM fold: it may add at most {saved:.0f} tokens of "
          f"prompt per turn before the measured saving disappears; the deterministic fold adds "
          f"{fa-aa:+.0f} tokens of recompute on the {folds} turns where it fires and "
          f"{st.mean(times):.1f} us of CPU elsewhere")
    print("\nNOT MEASURED HERE (stated as such in the paper): the token cost of an LLM-based fold, "
          "because no arm in this study uses one.")


if __name__ == "__main__":
    main()
