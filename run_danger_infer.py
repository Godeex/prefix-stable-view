#!/usr/bin/env python
"""A2: run the danger-slice paired trials on live ollama across families / temperature.

Reuses chat()/judge() from run_policy_infer (same contamination / fidelity口径 as (1) and the
frozen primary endpoint). Configurable via env:
  MA_MODEL   (default qwen3:8b)
  MA_TEMP    (default 0.0)  -> set >0 to break the temp=0 family collapse (soft point i)
  MA_SEED    (default 1234) -> passed to ollama options for reproducibility at temp>0
  TRIALS     (default danger_trials.jsonl)
  OUT_TAG    (suffix for the scored file)
Writes danger_scored_<model>_t<temp>_s<seed>.jsonl.
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_policy_infer as R

MODEL = os.environ.get("MA_MODEL", "qwen3:8b")
TEMP = float(os.environ.get("MA_TEMP", "0.0"))
SEED = int(os.environ.get("MA_SEED", "1234"))
TRIALS = os.path.join(HERE, os.environ.get("TRIALS", "danger_trials.jsonl"))
tag = os.environ.get("OUT_TAG") or f"{MODEL.replace(':','_')}_t{TEMP}_s{SEED}"
OUT = os.path.join(HERE, f"danger_scored_{tag}.jsonl")


def chat(model, prompt, temperature, seed, num_predict=48):
    body = {"model": model, "stream": False, "think": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": num_predict},
            "messages": [{"role": "user", "content": prompt}]}
    import urllib.request
    req = urllib.request.Request(R.CHAT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == 2:
                return {"_error": str(e)}
            time.sleep(2)


def main():
    trials = [json.loads(l) for l in open(TRIALS, encoding="utf-8")]
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for t in trials:
            d0 = time.perf_counter()
            resp = chat(MODEL, t["prompt"], TEMP, SEED)
            ms = (time.perf_counter() - d0) * 1000
            if "_error" in resp:
                ans, ptec, err = "", 0, resp["_error"]
            else:
                ans = (resp.get("message") or {}).get("content", "").strip()
                ptec = int(resp.get("prompt_eval_count") or 0)
                err = None
            correct, gp, hits, contam, subst = R.judge(t, ans)
            row = {**{k: t[k] for k in ("trial_id", "base_id", "K", "arm", "gold_key",
                                        "gold_val", "distractor_vals", "distractor_mean_sim",
                                        "distractor_max_sim", "n_fields_shown", "char_len")},
                   "model": MODEL, "temp": TEMP, "seed": SEED, "answer": ans,
                   "prompt_eval_count": ptec, "wall_ms": round(ms, 1), "correct": correct,
                   "gold_present": gp, "hit_distractor_vals": hits,
                   "contamination": contam, "substitution": subst, "error": err}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            n += 1
            if n % 30 == 0:
                print(f"  {n}/{len(trials)} done")
    import collections
    per = collections.defaultdict(list)
    for l in open(OUT, encoding="utf-8"):
        r = json.loads(l); per[r["arm"]].append(int(r["contamination"]))
    print(f"scored {n} -> {OUT}")
    for arm in ("length_only", "m_a"):
        v = per[arm]
        print(f"  {arm:11s} n={len(v)} contam_rate={sum(v)/max(1,len(v)):.3f}")


if __name__ == "__main__":
    main()
