#!/usr/bin/env python
"""B-lite-bounded: replay the fold sessions as sequential cache streams, capture the sawtooth.

Same replay discipline as run_prefix_cost.py: each arm is its OWN ascending-turn stream so the
radix prefix cache warms exactly as a real multi-turn conversation would; a per-run + per-arm
NONCE (in the prompt) defeats stale cross-run whole-prompt nodes. The three arms here are
length_only (naive slide), prefix_stable (UNBOUNDED head = idealized ceiling), and
bounded_stable (budgeted head that FOLDS). The novelty we want to see: bounded_stable's
recompute is low on append turns and SPIKES on fold turns (the head's leading bytes get
rewritten, invalidating the prefix), while its head size stays flat (does not grow to the KV
budget the way prefix_stable's unbounded head does).

Captured per call: prompt_eval_count, prompt_eval_cached_count, prompt_eval_duration, and we
carry fold_event/head_len from the trial so the analysis can split append vs fold turns.
recompute = count - cached (actual GPU prefill work). Correctness judged by gold substring to
substantiate matched quality.

Honest ceiling: llama.cpp in-process radix cache, single-request sequential replay, no
concurrency / batching / vLLM. Directional mechanism evidence, not an end-to-end TTFT headline.

Env: PC_MODEL (default qwen3:14b), PC_TEMP (0.0), PC_NUMCTX (8192), PC_NONCE. Resumable by
(session,turn,arm).
"""
import json, os, re, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("PC_MODEL", "qwen3:14b")
TEMP = float(os.environ.get("PC_TEMP", "0.0"))
NUMCTX = int(os.environ.get("PC_NUMCTX", "8192"))
TRIALS = os.path.join(HERE, "prefix_bounded_trials.jsonl")
tag = f"{MODEL.replace(':','_')}_t{TEMP}"
OUT = os.path.join(HERE, f"prefix_bounded_scored_{tag}.jsonl")
NONCE = os.environ.get("PC_NONCE") or str(int(time.time()))

ARMS = (("length_only", "prompt_length_only"),
        ("prefix_stable", "prompt_prefix_stable"),
        ("bounded_stable", "prompt_bounded_stable"))


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
    rows = [json.loads(l) for l in open(TRIALS, encoding="utf-8")]
    sess = {}
    for r in rows:
        sess.setdefault(r["session_id"], []).append(r)
    for v in sess.values():
        v.sort(key=lambda r: r["turn"])

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            j = json.loads(l)
            done.add((j["session_id"], j["turn"], j["arm"]))
    f = open(OUT, "a", encoding="utf-8")

    def emit(sid, turn, arm, prompt, gold, key, new_tok, fold, head_len):
        if (sid, turn, arm) in done:
            return
        d, wall = call(f"#run={NONCE}_{arm}\n" + prompt)
        if "__err__" in d:
            print(f"ERR {sid} t{turn} {arm}: {d['__err__']}")
            return
        cnt = metric(d, "prompt_eval_count")
        cached = metric(d, "prompt_eval_cached_count")
        ans = (d.get("message") or {}).get("content", "").strip()
        rec = {"session_id": sid, "turn": turn, "arm": arm, "model": MODEL,
               "temp": TEMP, "key": key, "gold": gold, "answer": ans,
               "correct": int(bool(norm(gold)) and norm(gold) in norm(ans)),
               "fold_event": fold, "head_len": head_len,
               "prompt_eval_count": cnt, "prompt_eval_cached_count": cached,
               "recompute": cnt - cached,
               "prompt_eval_duration_ms": round(metric(d, "prompt_eval_duration") / 1e6, 1),
               "total_duration_ms": round(metric(d, "total_duration") / 1e6, 1),
               "wall_ms": round(wall * 1e3, 1), "new_tokens_approx": new_tok}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()

    n = 0
    for sid in sorted(sess):
        for arm, pk in ARMS:                    # each arm = its own ascending-turn stream
            for r in sess[sid]:
                emit(sid, r["turn"], arm, r[pk], r["gold"], r["key"],
                     r["new_tokens_approx"], r["fold_event"], r["head_len"])
                n += 1
        print(f"[{MODEL}] {sid} done ({n} calls)")
    f.close()

    # ---- summary ----
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    import collections
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])   # arm->[n,recomp,cnt,correct,ms]
    app = collections.defaultdict(lambda: [0, 0])            # (arm,fold)->[n,recomp]
    for r in rows:
        a = agg[r["arm"]]
        a[0] += 1; a[1] += r["recompute"]; a[2] += r["prompt_eval_count"]
        a[3] += r["correct"]; a[4] += r["prompt_eval_duration_ms"]
        k = (r["arm"], r["fold_event"])
        app[k][0] += 1; app[k][1] += r["recompute"]
    print(f"\n{'arm':16s} {'n':>4s} {'acc':>6s} {'mean_prompt':>11s} {'mean_recompute':>14s} {'mean_prefill_ms':>15s}")
    for arm in ("length_only", "prefix_stable", "bounded_stable"):
        c, sr, sc, scorr, sms = agg[arm]
        if not c:
            continue
        print(f"{arm:16s} {c:4d} {scorr/c:6.3f} {sc/c:11.0f} {sr/c:14.0f} {sms/c:15.1f}")
    print("\nbounded_stable sawtooth (append vs fold turns):")
    for fold in (0, 1):
        c, sr = app[("bounded_stable", fold)]
        if c:
            lbl = "FOLD (head rewritten)" if fold else "append (head stable)"
            print(f"  {lbl:24s} n={c:4d} mean_recompute={sr/c:8.0f}")


if __name__ == "__main__":
    main()
