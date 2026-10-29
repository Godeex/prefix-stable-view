#!/usr/bin/env python
"""E5: run the three-arm matched-token trials on live ollama and judge contamination/fidelity.

temp=0, one model family (configurable). Records per trial: answer, prompt_eval_count (to verify the
matched-token claim on the tokenizer, not just chars), contamination (answer contains a value present
ONLY in shown distractors and not the gold), and gold-present (fidelity). Writes policy_scored_<model>.jsonl.
"""
import json, os, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CHAT = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("MA_MODEL", "qwen3:8b")
TRIALS = os.path.join(HERE, "trials_policy.jsonl")
OUT = os.path.join(HERE, f"policy_scored_{MODEL.replace(':','_')}.jsonl")


def chat(model, prompt, num_predict=48, temperature=0.0):
    body = {"model": model, "stream": False, "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(CHAT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == 2:
                return {"_error": str(e)}
            time.sleep(2)


def norm(s):
    return str(s).strip().lower().strip('.,"\'')


def judge(trial, ans):
    a = norm(ans)
    gold = norm(trial["gold_val"])
    gold_present = gold in a and gold != ""
    # contamination: an answer value that exists ONLY among shown distractors (not the gold)
    hits = [v for v in trial["distractor_vals"]
            if norm(v) and norm(v) in a and norm(v) != gold]
    contam = bool(hits)
    substitution = bool(hits) and not gold_present     # dropped gold, gave a distractor value
    correct = gold_present and not hits
    return correct, gold_present, hits, contam, substitution


def main():
    trials = [json.loads(l) for l in open(TRIALS, encoding="utf-8")]
    rows, n = [], 0
    with open(OUT, "w", encoding="utf-8") as f:
        for t in trials:
            d0 = time.perf_counter()
            resp = chat(MODEL, t["prompt"])
            ms = (time.perf_counter() - d0) * 1000
            if "_error" in resp:
                ans, ptec = "", 0
                err = resp["_error"]
            else:
                ans = (resp.get("message") or {}).get("content", "").strip()
                ptec = int(resp.get("prompt_eval_count") or 0)
                err = None
            correct, gp, hits, contam, subst = judge(t, ans)
            row = {**{k: t[k] for k in ("trial_id", "base_id", "K", "arm", "gold_key",
                                         "gold_val", "distractor_vals", "distractor_mean_sim",
                                         "distractor_max_sim", "n_fields_shown", "char_len")},
                   "model": MODEL, "answer": ans, "prompt_eval_count": ptec, "wall_ms": round(ms, 1),
                   "correct": correct, "gold_present": gp, "hit_distractor_vals": hits,
                   "contamination": contam, "substitution": subst, "error": err}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            rows.append(row); n += 1
            if n % 30 == 0:
                print(f"  {n}/{len(trials)} done")
    # matched-token verification: prompt_eval_count per arm should be near-equal
    import collections
    per_arm = collections.defaultdict(list)
    for r in rows:
        per_arm[r["arm"]].append(r["prompt_eval_count"])
    print(f"scored {len(rows)} -> {OUT}")
    for arm in ("length_only", "m_a", "worst"):
        v = per_arm[arm]
        print(f"  {arm:11s} mean prompt_tokens={sum(v)/len(v):.1f} contam_rate="
              f"{sum(1 for r in rows if r['arm']==arm and r['contamination'])/len(v):.3f}")


if __name__ == "__main__":
    main()
