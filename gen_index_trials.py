#!/usr/bin/env python
"""Build the current-binding INDEX trials (head || INDEX || RECENT).

Why. Sec. VI.C (C7) measures where the gold line sits INSIDE the append-only head: first is
privileged (correct 0.94/0.97) and middle/last collapse (0.04/0.11 on qwen3:14b), and the
controller has only an expensive handle on that -- moving a head line invalidates every byte
after it, or the value has to be pushed into the volatile tail. A classical materialized-view
answer exists that is BOTH prefix-stable and near the end: keep a second, small, REWRITABLE block
holding the derived predicate "current binding" (key -> its latest value). Rewriting it invalidates
only its own bytes plus the blocks behind it, so the recurring recompute is Theta(I + V) for
I = active keys, instead of Theta(H) for a relocated head line or the Theta(H_max * delta) sawtooth
of a bounded fold.

Manipulation (single factor: the INDEX block; everything else byte-for-byte the C7 construction):
  H     = head length in digest lines, {6, 24};  slot = gold line at first / middle / last.
  plain        STATE || RECENT                                    (the C7 layout, re-measured here)
  index        STATE || INDEX(all keys in context) || RECENT       (carries TARGET -> gold)
  index_nok    STATE || INDEX(same lines MINUS the queried key) || RECENT
               -- placebo: same block boundary, same position, same headers, no answer inside.
  Block order and the closing instruction line are identical in every arm; only the INDEX block
  differs, and within one (session, H, slot) cell the STATE bytes and the RECENT bytes are
  byte-identical across arms (asserted by check_index_controls.py).

One deliberate deviation from C7, required for "current binding" to be well defined: the near-twin
distractor's line is keyed RELATED1 (C7 used TARGET for every value-bearing record, which is the
d = 0 naive-carry hazard). Because BOTH arms of a cell carry that re-key, the arm contrast is
internal; the plain arm re-measures the position gradient under the new key scheme instead of
borrowing C7's numbers. The gold line itself, the distractor body, the tail and the query are
byte-identical to C7.

Single-shot fidelity probes, not cost probes (the cost side is gen_index_cost_trials.py). Values,
bodies, distractor selection and session count come from the frozen position_sessions.jsonl.
Deterministic. Env: IX_SESS, IX_OUT, VTAIL, IX_HS.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.environ.get("IX_SESS") or os.path.join(HERE, "position_sessions.jsonl")
OUT = os.environ.get("IX_OUT") or os.path.join(HERE, "index_trials.jsonl")
VTAIL = int(os.environ.get("VTAIL", "5"))
HS = [int(x) for x in os.environ.get("IX_HS", "6,24").split(",")]
SLOTS = ("first", "middle", "last")
ARMS = ("plain", "index", "index_nok")
QUERY_KEY = "TARGET"
IDX_HEAD = "INDEX (current bindings, rewritable):\n"


def key_of(r):
    """The binding key a record carries: gold keeps TARGET, distractors are keyed RELATEDk."""
    if r["is_gold"]:
        return QUERY_KEY
    if r["is_distractor"]:
        return f"RELATED{r['rel']}"
    return None


def item(r):
    k = key_of(r)
    if k is not None:
        return f"- {r['body']} | {k}={r['value']}"
    return f"- {r['body']} | NOTE=aux{r['idx']}"


def head_order(ids, recs, slot, H):
    """Non-gold lines in arrival order, gold line inserted at `slot` (exactly as C7)."""
    gi = next(i for i in ids if recs[i]["is_gold"])
    rest = [i for i in ids if i != gi]
    gpos = {"first": 0, "middle": H // 2, "last": H - 1}[slot]
    return gi, rest[:gpos] + [gi] + rest[gpos:]


def index_lines(order_ids, recs, keep_target):
    """One line per key bound by a record in context, in arrival order of the records."""
    out = []
    for i in order_ids:
        k = key_of(recs[i])
        if k is None:
            continue
        if k == QUERY_KEY and not keep_target:
            continue
        out.append(f"- {k} -> {recs[i]['value']}")
    return out


def prompt_for(s, H, slot, arm):
    recs = s["records"]
    t = H + VTAIL - 1
    tail = list(range(t - VTAIL + 1, t + 1))
    ids = list(range(0, t - VTAIL + 1))
    if not any(r["is_gold"] for r in recs[:len(ids)]) or t >= len(recs):
        return None
    gi, order = head_order(ids, recs, slot, H)
    head_lines = "\n".join(item(recs[i]) for i in order)
    tail_lines = "\n".join(item(recs[i]) for i in tail)
    idx = ""
    if arm != "plain":
        lines = index_lines(order + tail, recs, keep_target=(arm == "index"))
        idx = IDX_HEAD + "\n".join(lines) + "\n"
    p = ("STATE (accumulated, append-only):\n" + head_lines + "\n"
         + idx
         + "RECENT TOOL RESULTS:\n" + tail_lines + "\n"
         "Using the blocks above, answer.\n"
         f"QUESTION: {s['query']}\nANSWER:")
    return {"prompt": p, "turn": t, "gold_pos": order.index(gi),
            "wrong_pos": [k for k, i in enumerate(order) if recs[i]["is_distractor"]],
            "n_comp_in_ctx": len([i for i in ids + tail if recs[i]["is_distractor"]]),
            "head_lines": len(order), "tail_lines": len(tail),
            "index_lines": idx.count("-> ")}


def load():
    rows = [json.loads(l) for l in open(SESS, encoding="utf-8")]
    for s in rows:
        rel = 0
        for r in s["records"]:
            if r["is_distractor"]:
                rel += 1
                r["rel"] = rel
    return rows


def main():
    rows = load()
    out, skipped = [], 0
    for s in rows:
        for H in HS:
            for slot in SLOTS:
                for arm in ARMS:
                    b = prompt_for(s, H, slot, arm)
                    if b is None:
                        skipped += 1
                        continue
                    out.append({"trial_id": f"I{len(out)}", "arm": arm,
                                "session_id": s["session_id"], "H": H, "slot": slot,
                                "gold": s["gold_val"], "wrong_vals": s["wrong_vals"],
                                "query": s["query"], "turn": b["turn"],
                                "gold_pos": b["gold_pos"], "wrong_pos": b["wrong_pos"],
                                "n_comp_in_ctx": b["n_comp_in_ctx"],
                                "head_lines": b["head_lines"], "tail_lines": b["tail_lines"],
                                "index_lines": b["index_lines"],
                                "distractor_sim": s["distractor_sim"],
                                "char_len": len(b["prompt"]), "prompt": b["prompt"]})
    with open(OUT, "w", encoding="utf-8") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    per = {}
    for t in out:
        per.setdefault((t["arm"], t["H"], t["slot"]), []).append(t["char_len"])
    print(f"sessions={len(rows)} skipped={skipped} trials={len(out)} -> {OUT}")
    for arm in ARMS:
        detail = " ".join(f"H{h}_{sl}(n={len(v)},{sum(v) / len(v):.0f}ch)"
                          for (a, h, sl), v in sorted(per.items()) if a == arm)
        print(f"  {arm:10s} {detail}")


if __name__ == "__main__":
    main()
