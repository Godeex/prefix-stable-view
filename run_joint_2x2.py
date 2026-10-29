#!/usr/bin/env python
"""B: replay the DISTRACTION x PREFIXDEPTH 2x2 over multi-turn sessions; capture cost + quality.

Faithful to the M-A scorer (co/admission.py: score = utility - lam_d*distraction - lam_pd*prefix_depth)
by ablating its two lambda terms:
  d (DISTRACTION) : 1 -> neutralise near-twin distractors (fold them WITHOUT their competing TARGET
                      value, and exclude them from the verbatim tail); 0 -> naive carry (recency).
  p (PREFIXDEPTH) : 1 -> layout = append-only STABLE head (prior bytes never rewritten) + volatile
                      verbatim tail => prefix cache hit; 0 -> single recent-first rebuilt window
                      (front shifts every turn) => re-prefill.
Configs: c00 naive / c10 semantic-only / c01 prefix-only / c11 joint.

Gold is available in EVERY config and turn (fidelity floor) -> any substitution gap is about
carrying a competing value, never gold loss. Content set is matched per d across p (same records)
-> prompt size matched -> the only cost difference is slide vs stable (p), the only quality
difference is competing-value presence (d). Both axes are measured; interaction tested, not forced.

Each config is its OWN ascending-turn sequential cache stream (cache warms exactly as in a live
conversation). Per-run/per-config nonce defeats stale radix nodes from prior invocations (same bug
class fixed in B-lite). temp=0. Env RJ_MODEL (qwen3:14b), RJ_NUMCTX. Resumable by (session,turn,cfg).
"""
import json, os, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("RJ_MODEL", "qwen3:14b")
NUMCTX = int(os.environ.get("RJ_NUMCTX", "8192"))
MODE = os.environ.get("RJ_MODE", "")            # e.g. "tail"; "" = base
sfx = f"_{MODE}" if MODE else ""
SESS = os.environ.get("RJ_SESS") or os.path.join(HERE, f"joint_sessions{sfx}.jsonl")
tag = MODEL.replace(":", "_")
OUT = os.environ.get("RJ_OUT") or os.path.join(HERE, f"joint_scored_{tag}{sfx}.jsonl")
NONCE = os.environ.get("RJ_NONCE") or str(int(time.time()))
CONFIGS = [("c00", 0, 0), ("c10", 1, 0), ("c01", 0, 1), ("c11", 1, 1)]
FORCE = os.environ.get("RJ_FORCE", "") == "1"   # force a single committed value (removes list-all hedging)
FORCE_LINE = ("\nAnswer with ONLY the single recorded value for TARGET, e.g. R7. No other text.") if FORCE else ""


class Maint:
    def __init__(self, records, V):
        self.records = records
        self.V = V

    def tail_ids(self, t):
        return list(range(max(0, t - self.V + 1), t + 1))

    def disp(self, r, d):
        if r["is_gold"]:
            return f"TARGET={r['value']}"                       # fidelity floor: gold always verbatim
        if r["is_distractor"]:
            return "TARGET=[dropped]" if d == 1 else f"TARGET={r['value']}"
        return f"NOTE=aux{r['idx']}"

    def item(self, r, d):
        return f"- {r['body']} | {self.disp(r, d)}"

    def prompt(self, t, d, p, query):
        recs = self.records
        tail = self.tail_ids(t)
        tl = set(tail)
        older = [i for i in range(t + 1) if i not in tl]        # folded into the stable head
        if p == 1:
            # append-only head (older, oldest->newest, FULL body) + volatile recent tail: leading
            # bytes are a growing byte-prefix -> prefix cache hit.
            head = "\n".join(self.item(recs[i], d) for i in sorted(older))
            headsec = ("STATE (accumulated, append-only):\n" + head + "\n") if older else ""
            tailsec = "RECENT TOOL RESULTS:\n" + "\n".join(self.item(recs[i], d) for i in sorted(tail))
            return (f"{headsec}{tailsec}\nUsing STATE and RECENT above, answer.\n"
                    f"QUESTION: {query}{FORCE_LINE}\nANSWER:")
        else:
            # single recent-first window: the newest record sits at the FRONT every turn -> front
            # shifts -> whole prefix re-prefilled. Same items, so size is matched to p==1.
            items = "\n".join(self.item(recs[i], d) for i in range(t, -1, -1))
            return ("TOOL RESULTS (recent first):\n" + items +
                    f"\nAnswer from the results above.\nQUESTION: {query}{FORCE_LINE}\nANSWER:")

    def competing(self, t, d, p):
        """number of WRONG values that appear in the assembled context (quality hazard, structural)."""
        prompt = self.prompt(t, d, p, "x")
        return sum(1 for w in self._wrong if f"TARGET={w}" in prompt), any(
            f"TARGET={w}" in prompt for w in self._wrong)

    _wrong = []


def call(nonce, prompt):
    body = json.dumps({"model": MODEL, "stream": False, "think": False,
                       "options": {"temperature": 0.0, "seed": 1234,
                                   "num_predict": 40, "num_ctx": NUMCTX},
                       "messages": [{"role": "user", "content": f"#run={nonce}\n" + prompt}]}).encode()
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
    rows = [json.loads(l) for l in open(SESS, encoding="utf-8")]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            j = json.loads(l)
            done.add((j["session_id"], j["turn"], j["cfg"]))
    f = open(OUT, "a", encoding="utf-8")
    tstart = time.time()
    for s in rows:
        recs = s["records"]
        m = Maint(recs, s["budget_v"])
        m._wrong = s["wrong_vals"]
        # measure turns: probe_range (tail-distractor stress) or from hazard reveal to last
        if s.get("probe_range"):
            ta, tb = s["probe_range"]
            turns = list(range(ta, min(tb, len(recs) - 1) + 1))
        else:
            t0 = min(r["idx"] for r in recs if r["is_distractor"])
            turns = list(range(max(0, t0), len(recs)))
        for (cfg, d, p) in CONFIGS:
            nonce = f"{NONCE}_{cfg}"
            for t in turns:
                if (s["session_id"], t, cfg) in done:
                    continue
                prompt = m.prompt(t, d, p, s["query"])
                dd, wall = call(nonce, prompt)
                if "__err__" in dd:
                    print("ERR", s["session_id"], t, cfg, dd["__err__"]); continue
                ans = (dd.get("message") or {}).get("content", "").strip()
                low = ans.lower()
                comp, anycomp = m.competing(t, d, p)
                hit_wrong = [w for w in s["wrong_vals"] if w.lower() in low]
                subst = bool(hit_wrong) and s["gold_val"].lower() not in low
                cnt = int(dd.get("prompt_eval_count") or 0)
                cac = int(dd.get("prompt_eval_cached_count") or 0)
                f.write(json.dumps({
                    "session_id": s["session_id"], "turn": t, "cfg": cfg, "d": d, "p": p,
                    "model": MODEL, "gold": s["gold_val"], "answer": ans,
                    "correct": int(s["gold_val"].lower() in low and not hit_wrong),
                    "substitution": int(subst), "competing_in_ctx": comp, "any_competing": int(anycomp),
                    "prompt_eval_count": cnt, "prompt_eval_cached_count": cac, "recompute": cnt - cac,
                    "prompt_eval_duration_ms": round(int(dd.get("prompt_eval_duration") or 0) / 1e6, 1),
                    "wall_ms": round(wall * 1e3, 1)}) + "\n")
                f.flush()
        print(f"[{MODEL}] {s['session_id']} done  ({int(time.time()-tstart)}s)")
    f.close()
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
