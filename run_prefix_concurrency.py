#!/usr/bin/env python
"""B-concurrency: is the prefix-cache recompute advantage invariant to concurrent load?

External-validity gap: the published cost leg is single-request sequential
replay through llama.cpp's in-process radix cache; a production engine (vLLM/SGLang) batches
many sessions against a shared, paged KV pool. We cannot install a paged engine here (no new
runtime), but ollama DOES serve concurrent requests off one shared KV pool when several
sessions are in flight at once. That is a second, load-sharing regime we CAN exercise, and it
answers the narrower external-validity question: does the recompute GAP between the arms
survive when sessions contend, or was it an artifact of exclusive single-stream access?

Method: take the existing 3-arm prefix-cost trials. Instead of replaying each arm as its own
sequential stream, we fire turns from MANY sessions at the same time (a thread pool of width
`--parallel`), each request still carrying the per-run nonce so no stale cross-run node matches.
Concurrent in-flight sessions share the KV pool the way a batched server's requests do.

We capture the SAME per-call metrics (recompute = count - cached, prefill ms) PLUS a wall-clock
latency and a throughput proxy (requests/s under load, mean+p95 wall ms). Two claims are tested:
  (1) INVARIANCE: mean recompute/turn per arm is stable between the single-stream run and this
      concurrent run (the cache is content-keyed, not access-exclusive).
  (2) EXTERNAL VALIDITY PROXY: the recompute ordering (length_only >> stable/pinned) and its
      magnitude hold under shared-pool contention, not just single-stream.

Honest ceiling: still llama.cpp/ollama's radix cache with a shared pool, NOT a paged-attention
scheduler (no block-hash prefix reuse across DIFFERENT requests, no continuous batching of
prefill chunks). This tests contention-invariance of the token-level cache effect and gives a
load latency proxy; the vLLM/SGLang TTFT headline remains B-full and out of scope.

Env: PC_MODEL, PC_TEMP, PC_NUMCTX, PC_NONCE, PARALLEL, NSAMPLE.
"""
import json, os, re, time, argparse, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("PC_MODEL", "qwen3:14b")
TEMP = float(os.environ.get("PC_TEMP", "0.0"))
NUMCTX = int(os.environ.get("PC_NUMCTX", "8192"))
PARALLEL = int(os.environ.get("PARALLEL", "8"))
NSAMPLE = int(os.environ.get("NSAMPLE", "6"))   # sessions to load concurrently
TRIALS = os.path.join(HERE, "prefix_cost_trials.jsonl")
NONCE = os.environ.get("PC_NONCE") or str(int(time.time()))
tag = f"{MODEL.replace(':','_')}_t{TEMP}_p{PARALLEL}"
OUT = os.path.join(HERE, f"prefix_concurrency_scored_{tag}.jsonl")

ARMS = (("length_only", "prompt_length_only"),
        ("prefix_stable", "prompt_prefix_stable"),
        ("pinned_raw", "prompt_pinned_raw"))


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


_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=PARALLEL)
    ap.add_argument("--nsample", type=int, default=NSAMPLE)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(TRIALS, encoding="utf-8")]
    sess = {}
    for r in rows:
        sess.setdefault(r["session_id"], []).append(r)
    for v in sess.values():
        v.sort(key=lambda r: r["turn"])
    sids = sorted(sess)[:args.nsample]

    # Build one flat task list: a mid-session turn per arm per session. We take a turn window in
    # the middle (already-warmed prefix) so contention hits the same steady-state the single-run
    # measured. Fire them ALL concurrently across sessions and arms.
    tasks = []
    for sid in sids:
        rs = sess[sid]
        mid = rs[len(rs) // 2: len(rs) // 2 + 5]        # 5 warmed turns per session
        for r in mid:
            for arm, pk in ARMS:
                tasks.append((sid, r["turn"], arm, r[pk], r["gold"], r["key"]))

    f = open(OUT, "w", encoding="utf-8")

    def work(t):
        sid, turn, arm, prompt, gold, key = t
        d, wall = call(f"#run={NONCE}_c{args.parallel}_{arm}\n" + prompt)
        if "__err__" in d:
            return None
        cnt = int(d.get("prompt_eval_count") or 0)
        cached = int(d.get("prompt_eval_cached_count") or 0)
        ans = (d.get("message") or {}).get("content", "").strip()
        return {"session_id": sid, "turn": turn, "arm": arm, "model": MODEL,
                "temp": TEMP, "parallel": args.parallel, "nonce": NONCE,
                "key": key, "gold": gold, "answer": ans,
                "correct": int(bool(norm(gold)) and norm(gold) in norm(ans)),
                "prompt_eval_count": cnt, "prompt_eval_cached_count": cached,
                "recompute": cnt - cached,
                "prompt_eval_duration_ms": round(int(d.get("prompt_eval_duration") or 0) / 1e6, 1),
                "wall_ms": round(wall * 1e3, 1)}

    t_start = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for fu in as_completed(futs):
            rec = fu.result()
            if rec:
                with _lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                    done += 1
    elapsed = time.perf_counter() - t_start
    f.close()

    # ---- summary: recompute per arm under load + throughput/latency proxy ----
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    import collections
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, []])   # arm->[n,recomp,cnt,correct,[wall]]
    for r in rows:
        a = agg[r["arm"]]
        a[0] += 1; a[1] += r["recompute"]; a[2] += r["prompt_eval_count"]
        a[3] += r["correct"]; a[4].append(r["wall_ms"])
    print(f"\nparallel={args.parallel} sessions={len(sids)} elapsed={elapsed:.1f}s "
          f"throughput={done/elapsed:.2f} req/s (wall)  calls={done}")
    print(f"{'arm':16s} {'n':>4s} {'acc':>6s} {'mean_recompute':>14s} {'mean_wall_ms':>12s} {'p95_wall_ms':>11s}")
    for arm in ("length_only", "prefix_stable", "pinned_raw"):
        c, sr, sc, scorr, walls = agg[arm]
        if not c:
            continue
        walls.sort()
        p95 = walls[int(0.95 * (len(walls) - 1))]
        print(f"{arm:16s} {c:4d} {scorr/c:6.3f} {sr/c:14.0f} {sum(walls)/len(walls):12.1f} {p95:11.1f}")


if __name__ == "__main__":
    main()
