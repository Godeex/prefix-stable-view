#!/usr/bin/env python
"""Run-to-run variance of the prefill-duration counter on a shared-GPU host.

The published cost leg reports a session-cluster bootstrap CI over prompt_eval_duration. That
interval resamples SESSIONS within one run, so it answers "how much does the arm differ across
conversations?", NOT "how much does a repeated measurement of the same turns drift on this box?"
Two of the four GPUs here are shared with other users' processes, so the second question is the
one any reader is entitled to ask. We re-fire a fixed subset of the published cost prompts three
times, each repeat with a fresh per-run nonce (so no stale radix node fakes a warm cache), and
report the spread of the per-repeat arm means alongside the session-cluster CI of a single run.

Output: ms_repeat_variance.json (per-call rows + per-repeat arm means + spreads).
"""
import json, os, statistics as st, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("RMV_MODEL", "qwen3:14b")
NUMCTX = int(os.environ.get("RMV_NUMCTX", "8192"))
SESSIONS = ["S0", "S5"]
ARMS = ("length_only", "prefix_stable")
REPEATS = int(os.environ.get("RMV_REPEATS", "3"))
FIELD = {"length_only": "prompt_length_only", "prefix_stable": "prompt_prefix_stable"}


def call(prompt, nonce):
    body = json.dumps({"model": MODEL, "stream": False, "think": False,
                       "options": {"temperature": 0.0, "num_ctx": NUMCTX},
                       "messages": [{"role": "user", "content": nonce + "\n" + prompt}]}).encode()
    req = urllib.request.Request(CHAT, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return {"prompt_eval_count": d.get("prompt_eval_count", 0),
            "prompt_eval_cached_count": d.get("prompt_eval_cached_count", 0),
            "prompt_eval_duration_ms": d.get("prompt_eval_duration", 0) / 1e6,
            "total_duration_ms": d.get("total_duration", 0) / 1e9}


def main():
    rows = {}
    for l in open(os.path.join(HERE, "prefix_cost_trials.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        rows[(r["session_id"], r["turn"])] = r
    turns = sorted({t for (s, t) in rows if s in SESSIONS})
    out, per = [], {a: [] for a in ARMS}
    for rep in range(REPEATS):
        nonce = f"REPEAT-VARIANCE {rep} {int(time.time())}"
        for arm in ARMS:                      # each arm is its own sequential cache stream
            ms = []
            for t in turns:
                for s in SESSIONS:
                    r = rows[(s, t)]
                    j = call(r[FIELD[arm]], nonce)
                    j.update({"repeat": rep, "arm": arm, "session_id": s, "turn": t,
                              "recompute": j["prompt_eval_count"] - j["prompt_eval_cached_count"]})
                    out.append(j)
                    ms.append(j["prompt_eval_duration_ms"])
            per[arm].append(st.mean(ms))
            print(f"repeat {rep} {arm:14s} mean {per[arm][-1]:7.1f} ms  n={len(ms)}", flush=True)
    with open(os.path.join(HERE, "ms_repeat_variance_scored.jsonl"), "w", encoding="utf-8") as fh:
        for j in out:                          # one row per call, like every other *scored* file
            fh.write(json.dumps(j) + "\n")
    summ = {"model": MODEL, "repeats": REPEATS, "sessions": SESSIONS, "turns": len(turns),
            "calls": len(out), "per_repeat_mean_ms": per, "summary": {}}
    for arm in ARMS:
        v = per[arm]
        rng = max(v) - min(v)
        summ["summary"][arm] = {"min": min(v), "max": max(v), "range": rng,
                                "range_pct_of_mean": 100 * rng / st.mean(v),
                                "sd": st.stdev(v) if len(v) > 1 else 0.0,
                                "recompute_range": None}
    # recompute is the counter the headline claims anchor on: report its spread too
    for arm in ARMS:
        rc = [st.mean(j["recompute"] for j in out if j["arm"] == arm and j["repeat"] == r)
              for r in range(REPEATS)]
        summ["summary"][arm]["recompute_per_repeat_mean"] = rc
        summ["summary"][arm]["recompute_range"] = max(rc) - min(rc)
    json.dump(summ, open(os.path.join(HERE, "ms_repeat_variance.json"), "w"), indent=1)
    print(json.dumps(summ["summary"], indent=1))
    print("-> ms_repeat_variance.json")


if __name__ == "__main__":
    main()
