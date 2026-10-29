#!/usr/bin/env python
"""Shared-key gate (no model calls): the shared-key INDEX arms manipulate ONE variable on the C7 bytes.

Exit 1 on any failure, so no inference is spent on a design that does not test what it claims.
The point of this leg is that the published INDEX leg re-keyed the distractor (RELATED1) and so
removed the shared-key hazard; here the distractor keeps TARGET, exactly the C7 construction the
plain arm fails. Assertions over index_sk_trials.jsonl and the published position_trials.jsonl:

  1. SHARED-KEY STATE. Every value-bearing head line carries TARGET (gold and the one distractor);
     no RELATED key appears anywhere. This is the hazard the re-keyed leg took away.
  2. STATE AND RECENT BYTE-IDENTICAL TO C7. For each (session, H, slot) the STATE block and the
     RECENT block equal the published position cell, so plain needs no re-service and the contrast
     is internal to this leg.
  3. THE INDEX IS ONE ROW AND DERIVED. Exactly one `TARGET -> value` row, its value equal to the
     binding of a record actually in context: `index` names the gold, `index_nok` names the
     competitor, and the two arms are the C7 cell plus that row and nothing else.
  4. GOLD WHERE CLAIMED / HAZARD HELD FIXED. gold_pos == {0, H//2, H-1}; exactly one competing
     value in context, one wrong position; head/tail line counts equal H/VTAIL.
  5. FIDELITY FLOOR. Both TARGET bindings (gold and competitor) are present in the prompt, so a
     wrong answer is a reading failure, not an omission, and the index never introduces a value that
     is not already a stale line in the head or tail.
"""
import json, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_index_sk_trials import ARMS, IDX_HEAD, QUERY_KEY, VTAIL, SLOTS, HS

TRIALS = os.environ.get("IX_SK_TRIALS") or os.path.join(HERE, "index_sk_trials.jsonl")
PUB = os.path.join(HERE, "position_trials.jsonl")


def state_block(p):
    return p.split("STATE (accumulated, append-only):\n", 1)[1] \
            .split("\nRECENT TOOL RESULTS:")[0].split("\n" + IDX_HEAD)[0]


def tail_block(p):
    return p.split("RECENT TOOL RESULTS:\n", 1)[1].split("Using STATE and RECENT")[0]


def main():
    trials = [json.loads(l) for l in open(TRIALS, encoding="utf-8")]
    pub = {(t["session_id"], t["H"], t["slot"]): t
           for t in (json.loads(l) for l in open(PUB, encoding="utf-8"))}
    fails = []
    n = 0
    for t in trials:
        key = (t["session_id"], t["H"], t["slot"])
        p = pub.get(key)
        if p is None:
            fails.append(f"{key}: no published C7 counterpart")
            continue
        n += 1
        gold, wrongs = t["gold"], t["wrong_vals"]
        # (1) shared-key: no RELATED anywhere, two TARGET= bindings in the head
        if "RELATED" in t["prompt"]:
            fails.append(f"{key}/{t['arm']}: RELATED key present (not shared-key)")
        n_target = sum(1 for l in state_block(t["prompt"]).split("\n")
                       if l.startswith("- ") and f"| {QUERY_KEY}=" in l)
        if n_target != 2:
            fails.append(f"{key}/{t['arm']}: head TARGET bindings {n_target} != 2")
        # (2) STATE and RECENT identical to C7
        if state_block(t["prompt"]) != state_block(p["prompt"]):
            fails.append(f"{key}/{t['arm']}: STATE differs from published C7")
        if tail_block(t["prompt"]) != tail_block(p["prompt"]):
            fails.append(f"{key}/{t['arm']}: RECENT differs from published C7")
        # (3) one derived index row
        idx = t["prompt"].split(IDX_HEAD, 1)[1].split("RECENT TOOL RESULTS:")[0].strip("\n")
        rows = [l for l in idx.split("\n") if l]
        if len(rows) != 1 or t["index_lines"] != 1:
            fails.append(f"{key}/{t['arm']}: index rows {len(rows)} != 1")
            continue
        want = f"- {QUERY_KEY} -> {gold}" if t["arm"] == "index" else None
        if t["arm"] == "index" and rows[0] != want:
            fails.append(f"{key}: index row {rows[0]} != gold binding {want}")
        if t["arm"] == "index_nok" and rows[0] == f"- {QUERY_KEY} -> {gold}":
            fails.append(f"{key}: index_nok names the gold")
        named = rows[0].split("-> ")[1].strip()
        if named not in [gold] + wrongs:
            fails.append(f"{key}/{t['arm']}: index value {named} not a bound value in context")
        if t["arm"] == "index_nok" and named == gold:
            fails.append(f"{key}: index_nok not a competitor")
        # (4) position + hazard
        wp = {"first": 0, "middle": t["H"] // 2, "last": t["H"] - 1}[t["slot"]]
        if t["gold_pos"] != wp:
            fails.append(f"{key}/{t['arm']}: gold_pos {t['gold_pos']} != {wp}")
        if t["n_comp_in_ctx"] != 1 or len(t["wrong_pos"]) != 1:
            fails.append(f"{key}/{t['arm']}: competitor count {t['n_comp_in_ctx']}")
        if t["head_lines"] != t["H"] or t["tail_lines"] != VTAIL:
            fails.append(f"{key}/{t['arm']}: head/tail {t['head_lines']}/{t['tail_lines']}")
        # (5) fidelity floor: both bindings present in the head
        for v in (gold, named if t["arm"] == "index_nok" else wrongs[0]):
            if f"| {QUERY_KEY}={v}" not in t["prompt"]:
                fails.append(f"{key}/{t['arm']}: binding TARGET={v} absent from head")
    per = collections.Counter((t["arm"], t["H"], t["slot"]) for t in trials)
    print(f"trials={len(trials)}  cells-with-C7-counterpart={n}  arms={list(ARMS)}")
    print("per-arm counts:", {f"{a}_H{h}_{sl}": per[(a, h, sl)] for (a, h, sl) in sorted(per)})
    if fails:
        print(f"\nGATE FAIL ({len(fails)}):")
        for m in fails[:25]:
            print("  -", m)
        sys.exit(1)
    print("\nGATE PASS: shared-key STATE/RECENT are the published C7 bytes, the only addition is a "
          "single derived TARGET row (gold in index, the competitor in index_nok), the hazard and "
          "gold position are held fixed, and both bindings are present so any miss is a read failure.")


if __name__ == "__main__":
    main()
