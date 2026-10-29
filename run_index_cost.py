#!/usr/bin/env python
"""Cost leg: replay the INDEX arms as sequential cache streams and capture the counters.

Same protocol as run_prefix_cost.py: one stream per (session, arm) in ascending turn order, so the
radix prefix warms exactly as it would in a real multi-turn conversation; a per-run nonce on the
first line, per arm, so stale whole-prompt nodes from an earlier invocation cannot fake a hit;
recompute = prompt_eval_count - prompt_eval_cached_count (the GPU prefill work). Correctness is
judged on the newest record's value to substantiate the matched-quality premise (every arm carries
the answer, so a recompute gap is a placement effect, not an answerability effect).

The claim this tests: the head is a shared byte prefix, the rewrite is confined to the index block,
so per-turn recompute should grow as Theta(I) on top of the prefix_stable baseline instead of
jumping to Theta(H + V) the way a relocated head line or a sliding window does.

Env: IC_MODEL (qwen3:14b), IC_TEMP (0.0), IC_NUMCTX (8192), IC_TRIALS, IC_OUT, IC_NONCE.
Resumable by (session, turn, arm).
"""
import json, os, re, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("IC_MODEL", "qwen3:14b")
TEMP = float(os.environ.get("IC_TEMP", "0.0"))
NUMCTX = int(os.environ.get("IC_NUMCTX", "8192"))
TRIALS = os.environ.get("IC_TRIALS") or os.path.join(HERE, "index_cost_trials.jsonl")
ARMS = ("prefix_stable", "index_I2", "index_I8", "index_I24", "length_only")
tag = f"{MODEL.replace(':', '_')}_t{TEMP}"
OUT = os.environ.get("IC_OUT") or os.path.join(HERE, f"index_cost_scored_{tag}.jsonl")
NONCE = os.environ.get("IC_NONCE") or str(int(time.time()))


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
            req = urllib.request.Request(CHAT, data=body, headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            d = json.loads(urllib.request.urlopen(req, timeout=300).read())
            return d, time.perf_counter() - t0
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
    n = [0]

    def emit(sid, turn, arm, prompt, gold, key, new_tok, head_bytes):
        if (sid, turn, arm) in done:
            return
        d, wall = call(f"#run={NONCE}_{arm}\n" + prompt)
        if "__err__" in d:
            print(f"ERR {sid} t{turn} {arm}: {d['__err__']}", flush=True)
            return
        cnt = metric(d, "prompt_eval_count")
        cached = metric(d, "prompt_eval_cached_count")
        ans = (d.get("message") or {}).get("content", "").strip()
        rec = {"session_id": sid, "turn": turn, "arm": arm, "model": MODEL, "temp": TEMP,
               "key": key, "gold": gold, "answer": ans,
               "correct": int(bool(norm(gold)) and norm(gold) in norm(ans)),
               "prompt_eval_count": cnt, "prompt_eval_cached_count": cached,
               "recompute": cnt - cached,
               "prompt_eval_duration_ms": round(metric(d, "prompt_eval_duration") / 1e6, 1),
               "total_duration_ms": round(metric(d, "total_duration") / 1e6, 1),
               "wall_ms": round(wall * 1e3, 1), "new_tokens_approx": new_tok,
               "head_bytes": head_bytes}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        n[0] += 1

    for sid in sorted(sess):
        for arm in ARMS:
            for r in sess[sid]:
                emit(sid, r["turn"], arm, r[f"prompt_{arm}"], r["gold"], r["key"],
                     r["new_tokens_approx"], r.get("head_bytes", 0))
        print(f"[{MODEL}] {sid} done ({n[0]} calls)", flush=True)
    f.close()
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    import collections
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in rows:
        a = agg[r["arm"]]
        a[0] += 1; a[1] += r["recompute"]; a[2] += r["prompt_eval_count"]
        a[3] += r["correct"]; a[4] += r["prompt_eval_duration_ms"]
    print(f"\n{'arm':14s} {'n':>5s} {'acc':>6s} {'mean_prompt':>11s} {'mean_recompute':>14s} {'mean_prefill_ms':>15s}")
    for arm in ARMS:
        c, sr, sc, scorr, sms = agg[arm]
        if not c:
            continue
        print(f"{arm:14s} {c:5d} {scorr/c:6.3f} {sc/c:11.0f} {sr/c:14.1f} {sms/c:15.1f}")


if __name__ == "__main__":
    main()
