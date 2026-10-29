#!/usr/bin/env python
"""B-lite: replay prefix-cost sessions as two sequential cache streams, capture cache metrics.

For each session we replay arm-length_only turns in ascending order, then arm-prefix_stable
turns in ascending order. Each arm is its own stream so the prefix cache warms exactly as it
would in a real multi-turn conversation. The arms' leading bytes differ ("RECENT..." vs
"STATE..."), so there is no cross-arm cache contamination.

Captured per call: prompt_eval_count, prompt_eval_cached_count, prompt_eval_duration,
total_duration. recompute = count - cached (the actual GPU prefill work = the cost driver).
Also judges correctness (gold substring) to substantiate the MATCHED-QUALITY premise: both
arms must answer the recall query, so any recompute gap is a pure cache/mechanism effect.

Honest ceiling: llama.cpp in-process radix prefix cache, single-request sequential replay,
no concurrency / batching / vLLM. Directional mechanism evidence, not an end-to-end TTFT headline.

Env: PC_MODEL (default qwen3:14b), PC_TEMP (0.0), PC_NUMCTX (8192). Resumable by (session,turn,arm).
"""
import json, os, re, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("PC_MODEL", "qwen3:14b")
TEMP = float(os.environ.get("PC_TEMP", "0.0"))
NUMCTX = int(os.environ.get("PC_NUMCTX", "8192"))
TRIALS = os.path.join(HERE, "prefix_cost_trials.jsonl")
tag = f"{MODEL.replace(':','_')}_t{TEMP}"
OUT = os.path.join(HERE, f"prefix_cost_scored_{tag}.jsonl")
# Per-run nonce prepended to EVERY prompt identically: constant within a run so the genuine
# intra-session cross-turn prefix warming is preserved, unique across runs so stale whole-prompt
# radix nodes from a previous invocation never match (they would fake a cache hit for the sliding
# length_only window and destroy the measurement).
NONCE = os.environ.get("PC_NONCE") or str(int(time.time()))


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
    # group by session, keep turn order
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

    def emit(sid, turn, arm, prompt, gold, key, new_tok):
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
               "prompt_eval_count": cnt, "prompt_eval_cached_count": cached,
               "recompute": cnt - cached,
               "prompt_eval_duration_ms": round(metric(d, "prompt_eval_duration") / 1e6, 1),
               "total_duration_ms": round(metric(d, "total_duration") / 1e6, 1),
               "wall_ms": round(wall * 1e3, 1), "new_tokens_approx": new_tok}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()

    n = 0
    for sid in sorted(sess):
        # each arm is its own ascending-turn sequential cache stream
        for arm, pk in (("length_only", "prompt_length_only"),
                        ("prefix_stable", "prompt_prefix_stable"),
                        ("pinned_raw", "prompt_pinned_raw")):
            for r in sess[sid]:
                emit(sid, r["turn"], arm, r[pk], r["gold"], r["key"], r["new_tokens_approx"])
                n += 1
        print(f"[{MODEL}] {sid} done ({n} calls)")
    f.close()

    # summary
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    import collections
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])   # arm -> [n, sum_recomp, sum_cnt, sum_correct, sum_ms]
    for r in rows:
        a = agg[r["arm"]]
        a[0] += 1; a[1] += r["recompute"]; a[2] += r["prompt_eval_count"]
        a[3] += r["correct"]; a[4] += r["prompt_eval_duration_ms"]
    print(f"\n{'arm':16s} {'n':>4s} {'acc':>6s} {'mean_prompt':>11s} {'mean_recompute':>14s} {'mean_prefill_ms':>15s}")
    for arm in ("length_only", "prefix_stable", "pinned_raw"):
        c, sr, sc, scorr, sms = agg[arm]
        if not c:
            continue
        print(f"{arm:16s} {c:4d} {scorr/c:6.3f} {sc/c:11.0f} {sr/c:14.0f} {sms/c:15.1f}")


if __name__ == "__main__":
    main()
