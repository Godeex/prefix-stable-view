#!/usr/bin/env python
"""Gate (no model calls): prove the LLMLingua-2 arms are what their caption will claim.

Exit 1 on any failure, so no inference is spent on a competitor arm that does not test the thing
it is meant to test. Checks over prefix_comp_trials.jsonl (+ the published
prefix_cost_trials.jsonl it is derived from):

  1. PRUNING, NOT REWRITING. Every compressed history is a word-level SUBSEQUENCE of the
     `length_only` history it was built from, and the probe tail is byte-identical in every arm and
     byte-identical to `length_only`'s tail. A content rule may delete; it may not paraphrase or
     re-tokenise the surface.
  2. GOLD SURVIVAL IS COUNTED. The recorded survives_* flag is recomputed from the emitted bytes;
     a mismatch is fatal. The open arms do NOT protect the answer row, so "how much fidelity the
     rule throws away" is a measured property of the compressor, printed for the paper.
  3. MONOTONE BUDGET. Per trial words(smallest rate) <= ... <= words(largest rate) < words(raw).
  4. EACH keep ARM IS ITS OPEN ARM PLUS THE GOLD ROW AND NOTHING ELSE. For every rate that has both,
     removing the gold words' bytes from both bodies leaves the same string -- so the
     fidelity-vs-cost tradeoff is read as one variable at each budget.
  5. PREFIX INSTABILITY IS REAL AND MEASURED OFFLINE. For every arm, the fraction of consecutive
     turns whose prompts share a long byte prefix is computed. The comp arms re-prune the whole
     window each turn, so they must NOT be append-only; prefix_stable must be. This is the
     mechanism the recompute comparison predicts, checked before any call is made.
  6. DETERMINISM. Re-running the scorer on the first CC_RERUN trials (default 25) must reproduce
     those lines of prefix_comp_trials.jsonl byte-for-byte -- INCLUDING for arms added after the
     first serving pass, since each rate is a prefix of one rank order.

The arms are read off the trials file (every `prompt_*` key), not hard-coded, so a rate added later
is checked by the same gate rather than silently skipped.

Env: CC_RERUN (0 disables the rerun), CC_MODEL, HF_HUB_OFFLINE as needed by the scorer.
"""
import json, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TRIALS = os.path.join(HERE, "prefix_cost_trials.jsonl")
COMP = os.path.join(HERE, "prefix_comp_trials.jsonl")
PROBE_MARK = "\nUsing the results above, answer."


def roster(rows):
    """(open arms by descending rate, keep arms by descending rate) read off the trials file."""
    def rates(prefix):
        out = {}
        for r in rows:
            for k in r:
                if k.startswith("prompt_" + prefix):
                    out[int(k[len("prompt_" + prefix):])] = prefix + k[len("prompt_" + prefix):]
        return [out[i] for i in sorted(out, reverse=True)]
    return rates("comp_open"), rates("comp_keep")


fails = []


def bad(msg):
    fails.append(msg)
    print("FAIL " + msg)


def toks(s):
    return s.split()


def is_subsequence(sub, full):
    it = iter(full)
    return all(x in it for x in sub)


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    all_rows = [json.loads(l) for l in open(COMP, encoding="utf-8")]
    OPEN_ARMS, KEEP_ARMS = roster(all_rows)
    ALL_ARMS = OPEN_ARMS + KEEP_ARMS
    if not OPEN_ARMS or not KEEP_ARMS:
        print(f"FAIL no open/keep arms in {COMP}")
        sys.exit(1)
    # each keep arm must have a same-rate open arm, or "keep vs open at one variable" is untestable
    for a in KEEP_ARMS:
        if "comp_open" + a[len("comp_keep"):] not in OPEN_ARMS:
            bad(f"{a}: no same-rate open arm, so the gold-protection contrast is not one variable")
    rows = {f"{r['session_id']}|{r['turn']}": r for r in all_rows}
    pub = {f"{r['session_id']}|{r['turn']}": r for r in
           (json.loads(l) for l in open(TRIALS, encoding="utf-8"))}
    if len(rows) != len(pub):
        bad(f"trial count {len(rows)} != published {len(pub)}")

    surv = {a: 0 for a in ALL_ARMS}
    words_med = {}
    prefix_frac = {}
    import statistics as st

    sess = {}
    for k, r in rows.items():
        sess.setdefault(r["session_id"], []).append(r)

    # 5. consecutive-turn byte-prefix share, per arm (comp arms + the published arms for contrast).
    # A radix cache can reuse prompt[t-1] inside prompt[t] only up to their longest common prefix,
    # so LCP share is the offline predictor of the recompute measurement to come.
    def stream_stats(getter):
        hits, ratios, n = 0, [], 0
        for sid, lst in sess.items():
            lst.sort(key=lambda x: x["turn"])
            for prev, cur in zip(lst, lst[1:]):
                a, b = getter(prev), getter(cur)
                n += 1
                c = lcp(a, b)
                ratios.append(c / max(1, len(a)))
                if b.startswith(a):
                    hits += 1
        return (hits / max(1, n), st.median(ratios), n)

    for a in ALL_ARMS:
        prefix_frac[a] = stream_stats(lambda r, a=a: r.get("prompt_" + a))
    prefix_frac["length_only"] = stream_stats(lambda r: pub[f"{r['session_id']}|{r['turn']}"]["prompt_length_only"])
    prefix_frac["prefix_stable"] = stream_stats(lambda r: pub[f"{r['session_id']}|{r['turn']}"]["prompt_prefix_stable"])

    for k, r in sorted(rows.items()):
        p = pub[k]
        raw, tail = p["prompt_length_only"].split(PROBE_MARK, 1)
        if r["probe_tail"] != PROBE_MARK + tail:
            bad(f"{k} probe tail is not byte-identical to length_only's")
        for a in ALL_ARMS:
            body = r["prompt_" + a][: -len(r["probe_tail"])]
            if not is_subsequence(toks(body), toks(raw)):
                bad(f"{k} {a}: pruned history is not a word subsequence of length_only")
            if r["prompt_" + a][len(body):] != r["probe_tail"]:
                bad(f"{k} {a}: tail not byte-identical inside the arm")
            g = f"{r['key']}={r['gold']}"
            now = int(g in body)
            if now != r["survives_" + a]:
                bad(f"{k} {a}: survives flag {r['survives_'+a]} != measured {now}")
            surv[a] += now
            if r["words_" + a] != len(toks(body)):
                bad(f"{k} {a}: words field != recomputed word count")
        seq = [r["words_" + a] for a in reversed(OPEN_ARMS)]      # ascending rate -> ascending words
        if any(x > y for x, y in zip(seq, seq[1:])):
            bad(f"{k}: budget not monotone in the keep rate: {dict(zip(reversed(OPEN_ARMS), seq))}")
        if r["words_" + OPEN_ARMS[0]] >= r["n_words"]:
            bad(f"{k}: {OPEN_ARMS[0]} not strictly smaller than the raw window")
        # 4. each keep arm == its same-rate open arm, plus the gold tokens IF the pruner dropped them
        g = f"{r['key']}={r['gold']}"
        gtoks = toks(g)
        for ka in KEEP_ARMS:
            oa = "comp_open" + ka[len("comp_keep"):]
            bk = r["prompt_" + ka][: -len(r["probe_tail"])]
            bo = r["prompt_" + oa][: -len(r["probe_tail"])]
            kt, ot = toks(bk), toks(bo)
            if r["words_" + ka] < r["words_" + oa]:
                bad(f"{k}: {ka} smaller than {oa}")
            i = next((j for j in range(len(kt) - len(gtoks) + 1) if kt[j:j + len(gtoks)] == gtoks), None)
            if kt == ot:
                if r["survives_" + oa] != 1:
                    bad(f"{k}: {ka} == {oa} but {oa} does not report the gold row")
            elif i is None:
                bad(f"{k}: {ka} does not contain the gold row as contiguous tokens")
            elif kt[:i] + kt[i + len(gtoks):] != ot:
                bad(f"{k}: {ka} differs from {oa} by more than the gold row")

    for a in ALL_ARMS:
        words_med[a] = st.median([r["words_" + a] for r in rows.values()])
    words_med["raw_window"] = st.median([r["n_words"] for r in rows.values()])
    words_med["prefix_stable"] = st.median([r["stable_words"] for r in rows.values()])

    # 6. determinism: rescore a prefix of the file and diff those lines
    nrerun = int(os.environ.get("CC_RERUN", "25"))
    if nrerun:
        first = [k for k, _ in sorted(rows.items())][:nrerun]
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "prefix_cost_trials.jsonl")
            with open(src, "w", encoding="utf-8") as f:
                for k in first:
                    f.write(json.dumps(pub[k], ensure_ascii=False) + "\n")
            script = os.path.join(td, "compress_llmlingua2.py")
            import shutil
            shutil.copyfile(os.path.join(HERE, "compress_llmlingua2.py"), script)
            env = dict(os.environ)
            env["CC_RERUN"] = "0"
            res = subprocess.run([sys.executable, script], cwd=td, env=env,
                                 capture_output=True, text=True)
            if res.returncode != 0:
                bad(f"rerun exited {res.returncode}: {res.stderr[-300:]}")
            else:
                got = open(os.path.join(td, "prefix_comp_trials.jsonl"), encoding="utf-8").read()
                lines = [l for l in got.splitlines() if l.strip()]
                if len(lines) != len(first):
                    bad(f"rerun produced {len(lines)} lines, expected {len(first)}")
                for line, k in zip(lines, first):
                    if json.loads(line) != rows[k]:
                        bad(f"rerun differs from shipped file at {k} -- the scorer is not deterministic")
                        break

    print("== competitor control gate ==")
    print(f"trials {len(rows)}  sessions {len(sess)}")
    print(f"{'arm':16s} {'gold survives':>14s} {'median words':>13s} "
          f"{'turn-pairs prefix-hit':>22s} {'median LCP share':>17s}")
    for a in ALL_ARMS + ["length_only", "prefix_stable"]:
        s = surv.get(a, "-")
        w = words_med.get(a, "-")
        pf = prefix_frac.get(a)
        ss = f"{s:4d}/{len(rows)} ({s/len(rows):.3f})" if isinstance(s, int) else "n/a (published arm)"
        ws = f"{w:.0f}" if isinstance(w, float) else "-"
        print(f"{a:16s} {ss:>14s} {ws:>13s} "
              f"{(f'{pf[0]:.3f}' if pf else '-'):>22s} {(f'{pf[1]:.3f}' if pf else '-'):>17s}")
    print(f"raw sliding window median words {words_med['raw_window']:.0f}, "
          f"prefix_stable arm median words {words_med['prefix_stable']:.0f}")

    # Structural expectations of the design, checked not assumed. These byte-level prefix shares are
    # the OFFLINE predictor of the recompute measurement (computed before any model call): a pruner
    # that re-scores the whole window every turn rewrites the leading bytes, so nothing downstream
    # of byte 0 can be reused. If a comp arm were MORE prefix-stable than the append-only arm, the
    # orthogonality framing would be wrong and this gate says so rather than spending inference.
    best_comp = max(prefix_frac[a][1] for a in ALL_ARMS)
    for a in ALL_ARMS:
        if prefix_frac[a][1] > 0.10:
            bad(f"{a}: median leading-byte share with the previous turn is {prefix_frac[a][1]:.3f} "
                f"-- a per-turn re-prune should not preserve the prefix")
    ps = prefix_frac["prefix_stable"]
    if ps[1] < 3 * max(best_comp, 1e-9):
        bad(f"prefix_stable shares only {ps[1]:.3f} of its leading bytes turn to turn, within 3x of "
            f"the best comp arm {best_comp:.3f} -- the append-only head is not buying the reuse claimed")
    if prefix_frac["length_only"][1] > ps[1]:
        bad("length_only is more prefix-stable than prefix_stable -- published cost leg would be wrong")

    json.dump({"n_trials": len(rows), "n_sessions": len(sess),
               "gold_survives": {a: surv[a] for a in ALL_ARMS},
               "median_words": {a: words_med[a] for a in ALL_ARMS},
               "median_words_raw_window": words_med["raw_window"],
               "median_words_prefix_stable": words_med["prefix_stable"],
               "offline_prefix": {a: {"full_prefix_hit_rate": prefix_frac[a][0],
                                      "median_lcp_share": prefix_frac[a][1],
                                      "turn_pairs": prefix_frac[a][2]}
                                  for a in prefix_frac},
               "rerun_trials_checked": nrerun},
              open(os.path.join(HERE, "comp_controls_report.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    if fails:
        print(f"\n{len(fails)} FAILURES -- do not spend inference on this arm set")
        sys.exit(1)
    print("\nOK: subsequence pruning, counted gold survival, monotone budgets, per-turn churn, "
          "deterministic rescore.")


if __name__ == "__main__":
    main()
