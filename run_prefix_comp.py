#!/usr/bin/env python
"""Competitor leg: replay the LLMLingua-2 arms on Ollama and capture cache + fidelity metrics.

Missing control: "no token-level compressor arm (LLMLingua-class pruning) is
in the roster" -- the paper argued content rules are orthogonal to prefix stability but never ran
one. This leg runs one, on the SAME sessions as the published cost leg (prefix_comp_trials.jsonl,
built by compress_llmlingua2.py and gated by check_comp_controls.py), with the SAME protocol:
per-arm ascending-turn cache streams, temp 0, seed 1234, num_predict 40, num_ctx 8192, one per-run
nonce prepended to every prompt so stale whole-prompt radix nodes from an earlier invocation can
never fake a hit. `length_only` is replayed in this same invocation as a same-run control, so no
comparison rests on trusting two different server sessions.

Protocol is otherwise identical to run_prefix_cost.py; the published arms of that leg are read from
prefix_cost_scored_*.jsonl by analyze_prefix_comp.py rather than re-measured here.

PRE-DECLARED READING (fixed here, before any call in this leg; analyze_prefix_comp.py prints it
verbatim and judges each item on the measured numbers).
  Primary outcome, per call: recompute = prompt_eval_count - prompt_eval_cached_count (the tokens
  the engine must actually prefill; cached tokens are free). Corroborating real-clock measure:
  prompt_eval_duration_ms. Fidelity: `correct` under the same forced single-value commit protocol
  as the published cost leg, reported next to gold-in-context (counted offline, never assumed).
  All comparisons are paired within (session, turn) and session-cluster bootstrapped (6,000 reps),
  the same estimator and the same 0.03 actionability floor on `correct` the earlier legs used.
  C8-P1  BLOAT: at least one comp arm's measured prompt budget is at or below the published
         prefix_stable budget. If none is, this leg is a size-unmatched comparison and says so
         instead of claiming a bloat result.
  C8-P2  FIDELITY: comp_keep25 (the benefit-of-the-doubt arm, answer row force-kept) must not fall
         below length_only by more than 0.03 on `correct`. If even the gold-protected arm loses
         fidelity, pruning costs fidelity and that is the reported result. The open arms are free to
         fall; their gold-in-context rate is already measured.
  C8-P3  RECOMPUTE -- the falsifiable prediction. Every comp arm must recompute no better than
         length_only at any budget: mean paired [recompute(comp) - recompute(length_only)] has a
         bootstrap CI at or above 0. The mechanism is byte-level and already measured offline
         (comp_controls_report.json: a comp arm preserves ~0.00 of its previous turn's leading
         bytes, prefix_stable preserves 0.24). If instead a comp arm at a prefix_stable-sized
         budget recomputes like prefix_stable, the paper's claim that the prefix axis is ORTHOGONAL
         to the content rule is wrong and the headline must change.
  C8-P4  PARETO: at matched-or-smaller budget, prefix_stable must not be weakly dominated by any
         comp arm on (fidelity, recompute). Dominance in either direction is reported as measured.
  AMENDMENT -- added to this file after ~17% of the leg had been scored (92-115 of the 552 turns per
  competitor arm) and before the remaining calls; C8-P3 above is left exactly as pre-declared. That
  item compares ABSOLUTE recompute against the naive window, which is confounded by B: a pruned
  prompt is smaller, so it recomputes fewer tokens mechanically whatever its byte stability, and the
  contrast cannot separate "sent less" from "reused more". The mechanism claim this leg was built to
  test is about the SHARE of the prompt that must be recomputed (R/B) and about the comparison at
  matched budget; analyze_prefix_comp.py prints both, and the paper labels this item as corrected
  after partial observation rather than as a pre-registered prediction.
  AMENDMENT 2 (matched-budget arm) -- added after 506-529 of the 552 turns had been served per
  competitor arm (~95% of the original five-arm roster) and before any call of the arms it adds; the
  items above are left exactly as written. The external content rule has to be served "at matched prompt
  budget", and the roster as first shipped only BRACKETS it: at .25 the pruned prompt measures 527
  tokens against the published view's 445.5, at .12 it measures 292. So a rate that lands on the
  view's budget is added -- comp_open20 / comp_keep20 (keep-rate .20). The rate is not eyeballed:
  matched_budget.py fits mean measured prompt tokens on mean kept words over the open arms served so
  far (B = 71.9 + 1.795 x kept_words, max residual 1.7 tokens across the three bracket arms) and
  selects the grid rate whose predicted mean budget is closest to the view's (.20 -> predicted 437.4,
  8.1 tokens BELOW the view, i.e. the competitor is not given the larger budget). Two things follow
  and are reported as such: the rate choice is POST-HOC, informed by partial observation of this leg,
  and the match itself is only established by the measurement, never by the fit. The added arms are
  served under the same per-run nonce as this leg (so prompt bytes are identical to the gated
  trial file and the arms stay paired cell-by-cell), and the two arms already in flight keep their
  pre-declared names and rates.
Honest ceiling: llama.cpp in-process radix prefix cache, single-request sequential replay, no
concurrency, batching or vLLM -- directional mechanism evidence, not a production TTFT headline.

Env: CMP_MODEL (qwen3:14b), CMP_TEMP (0.0), CMP_NUMCTX (8192), CMP_NONCE (auto), CMP_ARMS
(all by default). Resumable by (session, turn, arm).
"""
import json, os, re, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("CMP_MODEL", "qwen3:14b")
TEMP = float(os.environ.get("CMP_TEMP", "0.0"))
NUMCTX = int(os.environ.get("CMP_NUMCTX", "8192"))
COMP = os.path.join(HERE, "prefix_comp_trials.jsonl")
BASE = os.path.join(HERE, "prefix_cost_trials.jsonl")
OUT = os.path.join(HERE, f"prefix_comp_scored_{MODEL.replace(':', '_')}_t{TEMP}.jsonl")
NONCE = os.environ.get("CMP_NONCE") or str(int(time.time()))
COMP_ARMS = ("comp_open50", "comp_open25", "comp_open20", "comp_open12",
             "comp_keep25", "comp_keep20")
ARMS = tuple(os.environ.get("CMP_ARMS", ",".join(COMP_ARMS + ("length_only",))).split(","))


def norm(s):
    s = str(s).lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", "", s))


def call(prompt):
    body = json.dumps({"model": MODEL, "stream": False, "think": False,
                       "options": {"temperature": TEMP, "seed": 1234,
                                   "num_predict": 40, "num_ctx": NUMCTX},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for a in range(3):
        try:
            req = urllib.request.Request(CHAT, data=body,
                                         headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            d = json.loads(urllib.request.urlopen(req, timeout=300).read())
            wall = time.perf_counter() - t0
            return d, wall
        except Exception as e:
            if a == 2:
                return {"__err__": str(e)}, 0.0
            time.sleep(2)


def metric(d, key):
    return int(d.get(key) or 0)


def main():
    comp = [json.loads(l) for l in open(COMP, encoding="utf-8")]
    base = {f"{r['session_id']}|{r['turn']}": r
            for r in (json.loads(l) for l in open(BASE, encoding="utf-8"))}
    sess = {}
    for r in comp:
        sess.setdefault(r["session_id"], []).append(r)
    for v in sess.values():
        v.sort(key=lambda r: r["turn"])

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            j = json.loads(l)
            done.add((j["session_id"], j["turn"], j["arm"]))
    f = open(OUT, "a", encoding="utf-8")
    n = err = 0

    def emit(sid, turn, arm, prompt, gold, key, survives):
        nonlocal n, err
        if (sid, turn, arm) in done:
            return
        d, wall = call(f"#run={NONCE}_{arm}\n" + prompt)
        if "__err__" in d:
            print(f"ERR {sid} t{turn} {arm}: {d['__err__']}")
            err += 1
            return
        cnt = metric(d, "prompt_eval_count")
        cached = metric(d, "prompt_eval_cached_count")
        ans = (d.get("message") or {}).get("content", "").strip()
        rec = {"session_id": sid, "turn": turn, "arm": arm, "model": MODEL, "temp": TEMP,
               "key": key, "gold": gold, "answer": ans,
               "correct": int(bool(norm(gold)) and norm(gold) in norm(ans)),
               "gold_in_context": survives,
               "prompt_eval_count": cnt, "prompt_eval_cached_count": cached,
               "recompute": cnt - cached,
               "prompt_eval_duration_ms": round(metric(d, "prompt_eval_duration") / 1e6, 1),
               "total_duration_ms": round(metric(d, "total_duration") / 1e6, 1),
               "wall_ms": round(wall * 1e3, 1)}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        n += 1

    print(f"[{MODEL}] nonce={NONCE} arms={ARMS} trials={len(comp)} -> {os.path.basename(OUT)}")
    t0 = time.time()
    for sid in sorted(sess):
        for arm in ARMS:                      # each arm is its own ascending-turn cache stream
            for r in sess[sid]:
                if arm == "length_only":
                    b = base[f"{sid}|{r['turn']}"]
                    emit(sid, r["turn"], arm, b["prompt_length_only"], r["gold"], r["key"], 1)
                else:
                    emit(sid, r["turn"], arm, r["prompt_" + arm], r["gold"], r["key"],
                         r["survives_" + arm])
        print(f"  {sid} done ({n} calls, {err} errors, {time.time()-t0:.0f}s elapsed)")
    f.close()
    print(f"leg finished: {n} new calls, {err} errors")


if __name__ == "__main__":
    main()
