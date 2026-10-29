#!/usr/bin/env python
"""Gate (no model calls): prove the position stress manipulates ONE variable.

Assertions over position_trials.jsonl. Exit 1 on any failure so nobody spends inference on a
design that does not test what the caption says:

  1. PERMUTATION ONLY. For a session and head length, the three slots emit the identical
     MULTISET of STATE lines and the identical character length (a reordering cannot change
     either), and the tail block and question are byte-identical across slots.
  2. THE GOLD LINE MOVES WHERE IT IS CLAIMED. gold_pos == {0, H//2, H-1} and every slot level
     is realised in every session.
  3. HAZARD HELD FIXED. The same single competing TARGET=WRONG value is present in the context
     in every cell, it stays in the head, and its absolute position shifts by at most ONE line
     across slots (the unavoidable consequence of inserting the gold line ahead of it).
  4. FIDELITY FLOOR. The gold value string is present in the prompt in every cell, so any drop
     in correct recall is a reading failure, not an omission.
  5. LAYOUT FIDELITY. The rendered template equals the published stable/labelled arm
     (layout cell s1l1 = main c11) for the slot that ordering already produces, i.e. the stress
     reuses the paper's template rather than a new one.
"""
import json, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_joint_layout import Layout                     # exact published stable template
from gen_position_trials import item, VTAIL

TRIALS = os.environ.get("PS_TRIALS") or os.path.join(HERE, "position_trials.jsonl")
SESS = os.environ.get("PS_SESS") or os.path.join(HERE, "position_sessions.jsonl")
SLOTS = ("first", "middle", "last")


def state_lines(p):
    head = p.split("STATE (accumulated, append-only):\n", 1)[1].split("RECENT TOOL RESULTS:")[0]
    return [l for l in head.strip("\n").split("\n") if l]


def tail_block(p):
    return p.split("RECENT TOOL RESULTS:\n", 1)[1].split("Using STATE and RECENT")[0]


def main():
    trials = [json.loads(l) for l in open(TRIALS, encoding="utf-8")]
    sess = {s["session_id"]: s for s in (json.loads(l) for l in open(SESS, encoding="utf-8"))}
    fails = []
    by = collections.defaultdict(dict)
    for t in trials:
        by[(t["session_id"], t["H"])][t["slot"]] = t
    n_cells = 0
    wrong_shift_max = 0
    for key, cell in sorted(by.items()):
        sid, H = key
        s = sess[sid]
        if set(cell) != set(SLOTS):
            fails.append(f"{key}: missing slots {sorted(set(SLOTS) - set(cell))}")
            continue
        n_cells += 1
        base = cell["first"]
        # (1) permutation only
        l0 = sorted(state_lines(base["prompt"]))
        for sl in SLOTS:
            t = cell[sl]
            if sorted(state_lines(t["prompt"])) != l0:
                fails.append(f"{sid}/H{H}/{sl}: STATE line multiset differs from 'first'")
            if t["char_len"] != base["char_len"]:
                fails.append(f"{sid}/H{H}/{sl}: char_len {t['char_len']} != {base['char_len']}")
            if tail_block(t["prompt"]) != tail_block(base["prompt"]):
                fails.append(f"{sid}/H{H}/{sl}: tail block differs")
        # (2) gold where claimed
        want = {"first": 0, "middle": H // 2, "last": H - 1}
        for sl in SLOTS:
            if cell[sl]["gold_pos"] != want[sl]:
                fails.append(f"{sid}/H{H}/{sl}: gold_pos {cell[sl]['gold_pos']} != {want[sl]}")
        # (3) hazard fixed: one competing value, in the head, position shifts <= 1
        gold = s["gold_val"]
        wrongs = s["wrong_vals"]
        for sl in SLOTS:
            t = cell[sl]
            if t["n_comp_in_ctx"] != 1:
                fails.append(f"{sid}/H{H}/{sl}: competing values in ctx = {t['n_comp_in_ctx']} != 1")
            if len(t["wrong_pos"]) != 1:
                fails.append(f"{sid}/H{H}/{sl}: wrong lines in head = {len(t['wrong_pos'])}")
                continue
            if t["head_lines"] != H or t["tail_lines"] != VTAIL:
                fails.append(f"{sid}/H{H}/{sl}: head/tail size {t['head_lines']}/{t['tail_lines']}")
        wp = [cell[sl]["wrong_pos"][0] for sl in SLOTS if cell[sl]["wrong_pos"]]
        if wp and max(wp) - min(wp) > 1:
            fails.append(f"{sid}/H{H}: competitor position spread {max(wp)-min(wp)} > 1")
        wrong_shift_max = max(wrong_shift_max, (max(wp) - min(wp)) if wp else 0)
        # (4) fidelity floor + content checks
        for sl in SLOTS:
            p = cell[sl]["prompt"]
            if f"TARGET={gold}" not in p:
                fails.append(f"{sid}/H{H}/{sl}: gold value absent from prompt")
            if f"TARGET={wrongs[0]}" not in p:
                fails.append(f"{sid}/H{H}/{sl}: competing value absent from prompt")
            # the near-twin distractor must be the FIRST-ranked one (deep distractor d1)
            d1 = next(r for r in s["records"] if r["is_distractor"] and r["idx"] == 3)
            if f"- {d1['body']} | TARGET={d1['value']}" not in p:
                fails.append(f"{sid}/H{H}/{sl}: near-twin distractor (idx 3) not verbatim")
        # (5) template fidelity: the published stable/labelled arm (layout cell s1l1 = main c11)
        #     at the same turn must show the SAME lines under the SAME headers; the stress differs
        #     from it only by the position of the gold line.
        lay = Layout(s["records"], s["budget_v"])
        pub = lay.prompt(cell["first"]["turn"], "s1l1", s["query"])
        pub_head, pub_tail = pub.split("RECENT TOOL RESULTS:")
        for sl in SLOTS:
            ours_head, ours_tail = cell[sl]["prompt"].split("RECENT TOOL RESULTS:")
            if sorted(ours_head.split("\n")) != sorted(pub_head.split("\n")):
                fails.append(f"{sid}/H{H}/{sl}: STATE lines differ from published s1l1")
            if ours_tail != pub_tail:
                fails.append(f"{sid}/H{H}/{sl}: header/tail bytes differ from published s1l1")
    # sizes
    per = collections.Counter((t["H"], t["slot"]) for t in trials)
    tok = collections.defaultdict(list)
    for t in trials:
        tok[(t["H"], t["slot"])].append(t["char_len"])
    print(f"trials={len(trials)}  session x head-length cells={n_cells}  slots/cell={len(SLOTS)}")
    print("cell counts:", {f"H{h}_{sl}": per[(h, sl)] for h, sl in sorted(per)})
    print(f"competitor position shift across slots: max {wrong_shift_max} line (assert <= 1)")
    print("mean prompt chars per cell (approx. tokens at 3.1 chars/token):",
          {f"H{h}_{sl}": f"{sum(v)/len(v):.0f} (~{sum(v)/len(v)/3.1:.0f} tok)"
           for (h, sl), v in sorted(tok.items())})
    # prompt-size facts the manuscript quotes: length is NOT a manipulated axis, so the spread
    # across the three slots inside one (session, H) group must be zero and is printed as evidence.
    grp_spread = {key: max(cell[sl]["char_len"] for sl in SLOTS) - min(cell[sl]["char_len"] for sl in SLOTS)
                  for key, cell in by.items() if set(cell) == set(SLOTS)}
    print(f"max char spread ACROSS the three slots within a (session, H) group: "
          f"{max(grp_spread.values())} over {len(grp_spread)} groups")
    for h in sorted({h for h, _ in tok}):
        v = [x for (hh, _), vs in tok.items() if hh == h for x in vs]
        print(f"  H={h:2d}: prompt chars mean {sum(v)/len(v):.0f}, min {min(v)}, max {max(v)} "
              f"over {len(v)} trials")
    if fails:
        print(f"\nGATE FAIL ({len(fails)}):")
        for m in fails[:25]:
            print("  -", m)
        sys.exit(1)
    print("\nGATE PASS: single-factor position manipulation, matched content and budget, "
          "hazard held fixed, published template reused.")


if __name__ == "__main__":
    main()
