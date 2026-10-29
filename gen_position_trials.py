#!/usr/bin/env python
"""Build the gold-POSITION stress trials for the stable (append-only) head.

Why: Sec. VI.C keeps the accumulated record older-first, which reads as if the front of the head
were privileged. Liu et al. (lost-in-the-middle) report the opposite ordering of positions for
multi-document retrieval, so every published arm in this paper places the gold line at the FRONT of
what is shown (field danger: gold field first; passage danger: gold passage first; cost leg: the
probed value is the NEWEST record, i.e. in the volatile tail). The one position condition never
measured is the one a real long conversation actually produces: the value the user asks for is an
OLD record that sits in the MIDDLE of a long head.

Manipulation (single factor, everything else held to the byte):
  H    = head length in digest lines, {6, 24}. H=6 is the mature operating point of the layout
         factorial (12-record sessions, V=5 tail); H=24 is the long-head stress.
  slot = where the gold line is inserted in the STATE block: first (0), middle (H//2), last (H-1).
  The non-gold head lines keep their ascending arrival order; only the gold line moves. Inserting
  gold shifts later lines by exactly one index, which the generator records so the gate can bound it.

Content = the frozen joint_sessions construction (real LongBench sentence bodies, near-twin
distractor carrying a competing TARGET=WRONG value verbatim, i.e. d=0 naive carry, plus far
fillers), so the hazard is the same one the 2x2 measures and the labels are the published
STATE/RECENT template (identical to layout cell s1l1 = main c11).

These are single-shot fidelity probes, NOT cost probes: moving the gold line necessarily breaks the
append-only byte invariant, which is a hypothetical ordering the shipped controller cannot choose
(arrival order fixes it). prompt_eval_count is logged only to report the actual token size of the
two head lengths. Env: PS_NS, PS_HS, VTAIL, PS_SESS, PS_OUT. Deterministic.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = os.environ.get("PS_SESS") or os.path.join(HERE, "position_sessions.jsonl")
OUT = os.environ.get("PS_OUT") or os.path.join(HERE, "position_trials.jsonl")
VTAIL = int(os.environ.get("VTAIL", "5"))
HS = [int(x) for x in os.environ.get("PS_HS", "6,24").split(",")]
SLOTS = ("first", "middle", "last")


def item(r):
    if r["is_gold"] or r["is_distractor"]:
        return f"- {r['body']} | TARGET={r['value']}"
    return f"- {r['body']} | NOTE=aux{r['idx']}"


def head_order(ids, recs, slot, H):
    """Non-gold lines in arrival order, gold line inserted at `slot`."""
    gi = next(i for i in ids if recs[i]["is_gold"])
    rest = [i for i in ids if i != gi]
    gpos = {"first": 0, "middle": H // 2, "last": H - 1}[slot]
    return gi, rest[:gpos] + [gi] + rest[gpos:]


def prompt_for(s, H, slot):
    recs = s["records"]
    t = H + VTAIL - 1                      # turn at which the head holds exactly H lines
    tail = list(range(t - VTAIL + 1, t + 1))
    ids = list(range(0, t - VTAIL + 1))     # older, folded into the head
    if not any(r["is_gold"] for r in recs[:len(ids)]) or t >= len(recs):
        return None
    gi, order = head_order(ids, recs, slot, H)
    head_lines = "\n".join(item(recs[i]) for i in order)
    gold_pos = order.index(gi)
    wrong_pos = [k for k, i in enumerate(order) if recs[i]["is_distractor"]]
    tail_lines = "\n".join(item(recs[i]) for i in tail)
    p = ("STATE (accumulated, append-only):\n" + head_lines + "\n"
         "RECENT TOOL RESULTS:\n" + tail_lines + "\n"
         "Using STATE and RECENT above, answer.\n"
         f"QUESTION: {s['query']}\nANSWER:")
    return {"prompt": p, "turn": t, "gold_pos": gold_pos, "wrong_pos": wrong_pos,
            "n_comp_in_ctx": len([i for i in ids + tail if recs[i]["is_distractor"]]),
            "head_lines": len(order), "tail_lines": len(tail)}


def main():
    rows = [json.loads(l) for l in open(SESS, encoding="utf-8")]
    out, skipped = [], 0
    for s in rows:
        for H in HS:
            for slot in SLOTS:
                b = prompt_for(s, H, slot)
                if b is None:
                    skipped += 1
                    continue
                out.append({"trial_id": f"P{len(out)}", "session_id": s["session_id"],
                            "H": H, "slot": slot, "gold": s["gold_val"],
                            "wrong_vals": s["wrong_vals"], "query": s["query"],
                            "turn": b["turn"], "gold_pos": b["gold_pos"],
                            "wrong_pos": b["wrong_pos"], "n_comp_in_ctx": b["n_comp_in_ctx"],
                            "head_lines": b["head_lines"], "tail_lines": b["tail_lines"],
                            "distractor_sim": s["distractor_sim"],
                            "char_len": len(b["prompt"]), "prompt": b["prompt"]})
    with open(OUT, "w", encoding="utf-8") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    per = {}
    for t in out:
        per.setdefault((t["H"], t["slot"]), []).append(t["char_len"])
    print(f"sessions={len(rows)} skipped={skipped} trials={len(out)} -> {OUT}")
    for H in HS:
        sp = {slot: max(per[(H, slot)]) - min(per[(H, slot)]) for slot in SLOTS if (H, slot) in per}
        tok = [round(sum(per[(H, s)]) / len(per[(H, s)]) / 3.1, 1)
               for s in SLOTS if (H, s) in per]
        print(f"  H={H:2d}: chars/session spread within head length = {sp}, "
              f"mean ~tokens {tok}")


if __name__ == "__main__":
    main()
