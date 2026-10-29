#!/usr/bin/env python
"""Shared-key hazard: the current-binding INDEX arms served under SHARED-KEY hazard.

Why. The published INDEX leg (gen_index_trials.py) re-keys the near-twin distractor to RELATED1 so
that a stale value never competes with the gold under one key. That re-key is itself the operation
that removes the C7 shared-key hazard, so the index view's quality gain was demonstrated only after
the hazard was taken away. This is the one cheap experiment that would
liberate C7 from the scope "two rows, no superseded key": serve the INDEX block ON TOP OF the C7
shared-key STATE, where gold and distractor BOTH carry TARGET, and ask whether an explicit current
binding still buys fidelity when a stale value genuinely competes under one key.

Construction (single factor: the INDEX block; STATE and RECENT byte-for-byte the published C7 cell):
  H     = head length in digest lines, {6, 24};  slot = gold line first / middle / last.
  STATE = shared-key digest, gold line `TARGET=gold` and the one distractor line `TARGET=wrong`
          (d = 0 naive carry), aux lines `NOTE=aux<idx>`. This is gen_position_trials.item() verbatim.
  plain        STATE || RECENT                     == the published position cell (not re-served).
  index        STATE || INDEX(`TARGET -> gold`)   || RECENT   current binding names the answer.
  index_nok    STATE || INDEX(`TARGET -> wrong`)  || RECENT   stale binding names the competitor:
               the decision probe for "when STATE and INDEX conflict, which line does the model read".
  Every arm uses the position closer verbatim ("Using STATE and RECENT above, answer.") so that plain
  is byte-identical to C7 and the arm contrast is internal; the two index arms differ from plain only
  by the INDEX block (asserted by check_index_sk_controls.py).

The INDEX block is one row: under shared-key the only bound key in context is TARGET, so the current
binding is exactly the line that must be kept fresh. `index` and `index_nok` hold the block shape,
header and position fixed and change only the value it names (gold vs the competing stale value).

Single-shot fidelity probes, not cost probes. Sessions, values, bodies, distractor selection and the
head/tail split come from the frozen position_sessions.jsonl, and the cells are the same
(session, H, slot) triples as position_trials.jsonl. Deterministic. Env: IX_SESS, IX_OUT, VTAIL, IX_HS.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.environ.get("IX_SESS") or os.path.join(HERE, "position_sessions.jsonl")
OUT = os.environ.get("IX_OUT") or os.path.join(HERE, "index_sk_trials.jsonl")
VTAIL = int(os.environ.get("VTAIL", "5"))
HS = [int(x) for x in os.environ.get("IX_HS", "6,24").split(",")]
SLOTS = ("first", "middle", "last")
ARMS = ("index", "index_nok")            # plain == the published position cell, not re-served
QUERY_KEY = "TARGET"
IDX_HEAD = "INDEX (current bindings, rewritable):\n"
CLOSER = "Using STATE and RECENT above, answer."   # verbatim from gen_position_trials


def item(r):
    """Shared-key digest line: both gold and distractor carry TARGET (the C7 hazard)."""
    if r["is_gold"] or r["is_distractor"]:
        return f"- {r['body']} | {QUERY_KEY}={r['value']}"
    return f"- {r['body']} | NOTE=aux{r['idx']}"


def head_order(ids, recs, slot, H):
    """Non-gold lines in arrival order, gold line inserted at `slot` (exactly as C7)."""
    gi = next(i for i in ids if recs[i]["is_gold"])
    rest = [i for i in ids if i != gi]
    gpos = {"first": 0, "middle": H // 2, "last": H - 1}[slot]
    return gi, rest[:gpos] + [gi] + rest[gpos:]


def competitor_value(order_ids, tail_ids, recs, gold):
    """The single competing TARGET value carried by a record in context (head or tail)."""
    for i in order_ids + tail_ids:
        r = recs[i]
        if r["is_distractor"]:
            return r["value"]
    return None


def prompt_for(s, H, slot, arm):
    recs = s["records"]
    t = H + VTAIL - 1
    tail = list(range(t - VTAIL + 1, t + 1))
    ids = list(range(0, t - VTAIL + 1))
    if not any(r["is_gold"] for r in recs[:len(ids)]) or t >= len(recs):
        return None
    gi, order = head_order(ids, recs, slot, H)
    gold = recs[gi]["value"]
    wrong = competitor_value(order, tail, recs, gold)
    if wrong is None:
        return None
    head_lines = "\n".join(item(recs[i]) for i in order)
    tail_lines = "\n".join(item(recs[i]) for i in tail)
    named = gold if arm == "index" else wrong
    idx = IDX_HEAD + f"- {QUERY_KEY} -> {named}\n"
    p = ("STATE (accumulated, append-only):\n" + head_lines + "\n"
         + idx
         + "RECENT TOOL RESULTS:\n" + tail_lines + "\n"
         + CLOSER + "\n"
         f"QUESTION: {s['query']}\nANSWER:")
    return {"prompt": p, "turn": t, "gold_pos": order.index(gi),
            "wrong_pos": [k for k, i in enumerate(order) if recs[i]["is_distractor"]],
            "n_comp_in_ctx": len([i for i in ids + tail if recs[i]["is_distractor"]]),
            "head_lines": len(order), "tail_lines": len(tail),
            "index_lines": 1, "index_value": named}


def main():
    rows = [json.loads(l) for l in open(SESS, encoding="utf-8")]
    out, skipped = [], 0
    for s in rows:
        for H in HS:
            for slot in SLOTS:
                for arm in ARMS:
                    b = prompt_for(s, H, slot, arm)
                    if b is None:
                        skipped += 1
                        continue
                    out.append({"trial_id": f"K{len(out)}", "arm": arm,
                                "session_id": s["session_id"], "H": H, "slot": slot,
                                "gold": s["gold_val"], "wrong_vals": s["wrong_vals"],
                                "query": s["query"], "turn": b["turn"],
                                "gold_pos": b["gold_pos"], "wrong_pos": b["wrong_pos"],
                                "n_comp_in_ctx": b["n_comp_in_ctx"],
                                "head_lines": b["head_lines"], "tail_lines": b["tail_lines"],
                                "index_lines": b["index_lines"], "index_value": b["index_value"],
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
