#!/usr/bin/env python
"""Control: full 2x2 over (block LABELS) x (prefix STABILITY) at fixed content.

The main 2x2 ablates (distractor folding d) x (prefix depth p) and reports that the stable,
labelled arm reaches substitution ~0 while competing values are still in context. That leaves two
candidate mechanisms on the same axis: the STATE/RECENT block structure (an authority cue), or
prefix stability itself. This script separates them with two extra cells, so all four cells of the
(L x S) factorial are measured in ONE epoch on the SAME sessions, turns, model, seed:

  cell       labels  stability   construction
  s0l0       no      sliding     naive recent-first window, generic header            (= main c00)
  s0l1       yes     sliding     same blocks/headers, older block re-sorted by recency
  s1l0       no      stable      same older-first->newer ordering, neutral header
  s1l1       yes     stable      append-only STATE + volatile RECENT                  (= main c11 arm)

Only the ORDER of identical content differs between paired cells: s0l1 vs s1l1 keeps every label and
every record and permutes the older block, so a substitution gap between them is a stability effect
and not a content/size effect; s1l0 vs s1l1 keeps the order and strips the words STATE/RECENT, so a
gap there is a labelling effect. Cost is predicted to track stability only.

s0l0 and s1l1 are re-measured here rather than read from the frozen main run, which doubles as a
cross-epoch reproducibility check for the two published cells.

Env: RJ_MODEL (qwen3:14b), RJ_NUMCTX, RJ_SESS, RJ_OUT, RJ_NONCE. Resumable by (session,turn,cell).
"""
import json, os, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("RJ_MODEL", "qwen3:14b")
NUMCTX = int(os.environ.get("RJ_NUMCTX", "8192"))
SESS = os.environ.get("RJ_SESS") or os.path.join(HERE, "joint_sessions.jsonl")
tag = MODEL.replace(":", "_")
OUT = os.environ.get("RJ_OUT") or os.path.join(HERE, f"joint_layout_scored_{tag}.jsonl")
NONCE = os.environ.get("RJ_NONCE") or str(int(time.time()))
CELLS = ("s0l0", "s0l1", "s1l0", "s1l1")


class Layout:
    """d fixed at 0 (naive carry): gold and both competing values stay verbatim in every cell."""

    def __init__(self, records, V):
        self.records = records
        self.V = V

    def item(self, r):
        if r["is_gold"]:
            return f"- {r['body']} | TARGET={r['value']}"
        if r["is_distractor"]:
            return f"- {r['body']} | TARGET={r['value']}"
        return f"- {r['body']} | NOTE=aux{r['idx']}"

    def tail_ids(self, t):
        return list(range(max(0, t - self.V + 1), t + 1))

    def prompt(self, t, cell, query):
        recs = self.records
        tail = set(self.tail_ids(t))
        older = [i for i in range(t + 1) if i not in tail]
        newest = self.item(recs[t])
        tail_lines = "\n".join(self.item(recs[i]) for i in sorted(tail))
        # labelled, append-only (oldest first): leading bytes only grow -> prefix stable
        older_stable = "\n".join(self.item(recs[i]) for i in sorted(older))
        # labelled, re-sorted by recency: the first line under the header changes every turn -> slides
        older_sliding = "\n".join(self.item(recs[i]) for i in sorted(older, reverse=True))

        if cell == "s0l0":
            items = "\n".join(self.item(recs[i]) for i in range(t, -1, -1))
            return ("TOOL RESULTS (recent first):\n" + items +
                    f"\nAnswer from the results above.\nQUESTION: {query}\nANSWER:")
        if cell == "s0l1":
            headsec = ("STATE (accumulated, append-only):\n" + older_sliding + "\n") if older else ""
            return (f"{headsec}RECENT TOOL RESULTS:\n{tail_lines}\n"
                    f"Using STATE and RECENT above, answer.\nQUESTION: {query}\nANSWER:")
        if cell == "s1l0":
            headsec = ("CONTEXT:\n" + older_stable + "\n") if older else ""
            return (f"{headsec}MORE:\n{tail_lines}\n"
                    f"Answer from the lines above.\nQUESTION: {query}\nANSWER:")
        headsec = ("STATE (accumulated, append-only):\n" + older_stable + "\n") if older else ""
        return (f"{headsec}RECENT TOOL RESULTS:\n{tail_lines}\n"
                f"Using STATE and RECENT above, answer.\nQUESTION: {query}\nANSWER:")

    def competing(self, t, cell, query):
        p = self.prompt(t, cell, query)
        return sum(1 for w in self._wrong if f"TARGET={w}" in p)

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
            done.add((j["session_id"], j["turn"], j["cell"]))
    f = open(OUT, "a", encoding="utf-8")
    tstart = time.time()
    for s in rows:
        recs = s["records"]
        m = Layout(recs, s["budget_v"])
        m._wrong = s["wrong_vals"]
        t0 = min(r["idx"] for r in recs if r["is_distractor"])
        turns = list(range(max(0, t0), len(recs)))
        for cell in CELLS:
            nonce = f"{NONCE}_{cell}"
            for t in turns:
                if (s["session_id"], t, cell) in done:
                    continue
                prompt = m.prompt(t, cell, s["query"])
                dd, wall = call(nonce, prompt)
                if "__err__" in dd:
                    print("ERR", s["session_id"], t, cell, dd["__err__"], flush=True)
                    continue
                ans = (dd.get("message") or {}).get("content", "").strip()
                low = ans.lower()
                hit_wrong = [w for w in s["wrong_vals"] if w.lower() in low]
                subst = bool(hit_wrong) and s["gold_val"].lower() not in low
                cnt = int(dd.get("prompt_eval_count") or 0)
                cac = int(dd.get("prompt_eval_cached_count") or 0)
                f.write(json.dumps({
                    "session_id": s["session_id"], "turn": t, "cell": cell,
                    "labels": int(cell[3] == "1"), "stable": int(cell[1] == "1"),
                    "model": MODEL, "gold": s["gold_val"], "answer": ans,
                    "correct": int(s["gold_val"].lower() in low and not hit_wrong),
                    "substitution": int(subst), "mentions_gold": int(s["gold_val"].lower() in low),
                    "mentions_wrong": int(bool(hit_wrong)),
                    "competing_in_ctx": m.competing(t, cell, s["query"]),
                    "prompt_chars": len(prompt),
                    "prompt_eval_count": cnt, "prompt_eval_cached_count": cac, "recompute": cnt - cac,
                    "prompt_eval_duration_ms": round(int(dd.get("prompt_eval_duration") or 0) / 1e6, 1),
                    "wall_ms": round(wall * 1e3, 1)}) + "\n")
                f.flush()
        print(f"[{MODEL}] {s['session_id']} done ({int(time.time()-tstart)}s)", flush=True)
    f.close()
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
