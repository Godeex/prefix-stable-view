#!/usr/bin/env python
"""Serve the INDEX fidelity trials and score the same three outcome buckets as C7.

Judging口径 is byte-for-byte the one run_position_stress.py publishes (correct = gold named and no
competing value named; substitution = competing value named and gold not; other = neither),
temp=0, seed=1234, num_predict=40, per-run nonce. Single-shot probes, so prompt_eval_cached_count
is not meaningful here; prompt_eval_count is kept to report the true token size of each arm. The
cost side of the same idea is measured separately by run_index_cost.py.

Env: IX_MODEL (qwen3:14b), IX_NUMCTX (8192), IX_TRIALS, IX_OUT, IX_NONCE. Resumable by trial_id.
"""
import json, os, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("IX_MODEL", "qwen3:14b")
NUMCTX = int(os.environ.get("IX_NUMCTX", "8192"))
TRIALS = os.environ.get("IX_TRIALS") or os.path.join(HERE, "index_trials.jsonl")
tag = MODEL.replace(":", "_")
OUT = os.environ.get("IX_OUT") or os.path.join(HERE, f"index_scored_{tag}.jsonl")
NONCE = os.environ.get("IX_NONCE") or str(int(time.time()))


def call(nonce, prompt):
    body = json.dumps({"model": MODEL, "stream": False, "think": False,
                       "options": {"temperature": 0.0, "seed": 1234,
                                   "num_predict": 40, "num_ctx": NUMCTX},
                       "messages": [{"role": "user", "content": f"{nonce}\n" + prompt}]}).encode()
    for a in range(3):
        try:
            req = urllib.request.Request(CHAT, data=body, headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            dd = json.loads(urllib.request.urlopen(req, timeout=300).read())
            return dd, time.perf_counter() - t0
        except Exception as e:
            if a == 2:
                return {"__err__": str(e)}, 0.0
            time.sleep(2)


def main():
    trials = [json.loads(l) for l in open(TRIALS, encoding="utf-8")]
    done = {json.loads(l)["trial_id"] for l in open(OUT, encoding="utf-8")} if os.path.exists(OUT) else set()
    f = open(OUT, "a", encoding="utf-8")
    tstart = time.time()
    for i, t in enumerate(trials):
        if t["trial_id"] in done:
            continue
        dd, wall = call(f"#run={NONCE}_{t['arm']}_H{t['H']}_{t['slot']}", t["prompt"])
        if "__err__" in dd:
            print("ERR", t["trial_id"], dd["__err__"], flush=True)
            continue
        ans = (dd.get("message") or {}).get("content", "").strip()
        low = ans.lower()
        hit_wrong = [w for w in t["wrong_vals"] if w.lower() in low]
        mg = t["gold"].lower() in low
        rec = {k: t[k] for k in ("trial_id", "arm", "session_id", "H", "slot", "gold", "gold_pos",
                                 "wrong_pos", "n_comp_in_ctx", "index_lines", "char_len",
                                 "distractor_sim")}
        rec.update({"model": MODEL, "temp": 0.0, "seed": 1234, "answer": ans,
                    "correct": int(mg and not hit_wrong),
                    "substitution": int(bool(hit_wrong) and not mg),
                    "other": int(not mg and not hit_wrong),
                    "mentions_gold": int(mg), "mentions_wrong": int(bool(hit_wrong)),
                    "prompt_eval_count": int(dd.get("prompt_eval_count") or 0),
                    "wall_ms": round(wall * 1e3, 1)})
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        if (i + 1) % 40 == 0:
            print(f"[{MODEL}] {i+1}/{len(trials)} ({int(time.time()-tstart)}s)", flush=True)
    f.close()
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
