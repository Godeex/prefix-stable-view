#!/usr/bin/env python
"""B2: passage-level inference + substitution judge (danger-slice test).

Same judge口径 as EX-2b (correct = gold span contained; sub_from_distractor = wrong AND the model's
short answer span appears in a shown distractor passage but not the gold passage). Adds model/temp/seed/
arm fields so analyze_lb_danger can pair length_only vs m_a and check cross-family sign consistency.
Env: LB_MODEL, LB_TEMP, LB_SEED, LB_TRIALS (default trials_lb_paired.jsonl). Resumable by trial_id.
"""
import json, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("LB_MODEL", "qwen3:8b")
TEMP = float(os.environ.get("LB_TEMP", "0.0"))
SEED = int(os.environ.get("LB_SEED", "1234"))
TRIALS = os.environ.get("LB_TRIALS", "trials_lb_paired.jsonl")
tag = os.environ.get("LB_TAG") or f"{MODEL.replace(':','_')}_t{TEMP}_s{SEED}"
OUT = os.path.join(HERE, f"lb_danger_scored_{tag}.jsonl")


def norm(s):
    s = str(s).lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", "", s))


def chat(prompt):
    body = json.dumps({"model": MODEL, "stream": False, "think": False,
                       "options": {"temperature": TEMP, "seed": SEED, "num_predict": 20},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for a in range(3):
        try:
            req = urllib.request.Request(CHAT, data=body, headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=200).read())
            return (r.get("message", {}).get("content", "") or "").strip(), r.get("prompt_eval_count")
        except Exception as e:
            if a == 2:
                return f"__ERR__{e}", None
            time.sleep(2)


def main():
    trials = [json.loads(l) for l in open(os.path.join(HERE, TRIALS), encoding="utf-8")]
    done = {json.loads(l)["trial_id"] for l in open(OUT, encoding="utf-8")} if os.path.exists(OUT) else set()
    f = open(OUT, "a", encoding="utf-8")
    for i, t in enumerate(trials):
        if t["trial_id"] in done:
            continue
        ans, pec = chat(t["prompt"])
        na, ng = norm(ans), norm(t["gold"])
        correct = bool(ng) and ng in na
        sub = False
        if not correct and na and len(na) >= 2:
            gpn = norm(t.get("gold_passage", ""))
            for p in t["distractor_vals"]:
                if na in norm(p) and na not in gpn:
                    sub = True
                    break
        rec = {k: t[k] for k in ("trial_id", "base_id", "dataset", "arm", "gold",
                                 "distractor_mean_sim", "distractor_max_sim", "K",
                                 "n_fields_shown", "char_len")}
        rec.update({"model": MODEL, "temp": TEMP, "seed": SEED, "answer": ans,
                    "prompt_eval_count": pec, "correct": int(correct),
                    "wrong": int(not correct), "sub_from_distractor": int(sub)})
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        if (i + 1) % 30 == 0:
            print(f"[{MODEL} t{TEMP}] {i+1}/{len(trials)} {t['arm']}")
    f.close()
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    import collections
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["arm"]][0] += r["sub_from_distractor"]; per[r["arm"]][1] += 1
    print(f"scored {len(rows)} -> {OUT}")
    for arm in ("length_only", "m_a"):
        s, n = per[arm]
        print(f"  {arm:11s} n={n} substitution_rate={s/max(n,1):.3f}")


if __name__ == "__main__":
    main()
