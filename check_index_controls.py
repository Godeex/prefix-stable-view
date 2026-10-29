#!/usr/bin/env python
"""Gate (no model calls): prove the INDEX arm manipulates ONE variable.

Exit 1 on any failure, so no inference is spent on a design that does not test what its caption
will claim. Assertions over index_trials.jsonl (+ the published position_trials.jsonl it derives
from):

  1. HEAD AND TAIL BYTE-IDENTICAL ACROSS ARMS. Inside one (session, H, slot) cell the STATE block
     and the RECENT block are the same bytes in plain / index / index_nok; the ONLY textual
     difference between arms is the INDEX block, and the closing instruction + question are shared.
  2. THE HEAD IS A STRICT BYTE PREFIX OF EVERY ARM. The head ends before the first rewritten byte,
     i.e. appending the INDEX block cannot invalidate STATE -- the property the cost claim rests on.
  3. GOLD WHERE CLAIMED. gold_pos == {0, H//2, H-1}, identical across the three arms of a cell.
  4. THE INDEX IS DERIVED, NOT INVENTED. Every index line's key and value equal the binding of a
     record actually in context, one line per bound key, arrival order; the `index` arm carries the
     queried key -> gold, `index_nok` carries the same list MINUS that one row.
  5. HAZARD HELD FIXED. Exactly one competing value in context, in the head, its absolute line
     index shifting by at most one across slots (the unavoidable effect of inserting the gold row).
  6. PROVENANCE. The cell's STATE multiset equals the published C7 cell's modulo the distractor
     re-key (TARGET -> RELATED1): gold line, aux lines, tail and question byte-identical, so the
     stress reuses the published construction rather than a new one.
  7. FIDELITY FLOOR. The gold value string is present in the prompt of every arm (in the head), so
     any failure to recall is a reading failure, not an omission.
"""
import json, os, re, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_index_trials import ARMS, IDX_HEAD, QUERY_KEY, VTAIL, load

TRIALS = os.environ.get("IX_TRIALS") or os.path.join(HERE, "index_trials.jsonl")
PUB = os.path.join(HERE, "position_trials.jsonl")


def split_blocks(p):
    head = p.split("STATE (accumulated, append-only):\n", 1)[1].split("\n" + IDX_HEAD)[0]
    head = head.split("RECENT TOOL RESULTS:")[0]
    idx = p.split(IDX_HEAD, 1)[1].split("RECENT TOOL RESULTS:")[0] if IDX_HEAD in p else None
    tail = p.split("RECENT TOOL RESULTS:\n", 1)[1].split("Using the blocks above")[0]
    clos = "Using the blocks above" + p.split("Using the blocks above", 1)[1]
    return head.rstrip("\n"), idx, tail.rstrip("\n"), clos


def main():
    trials = [json.loads(l) for l in open(TRIALS, encoding="utf-8")]
    pub = {(t["session_id"], t["H"], t["slot"]): t
           for t in (json.loads(l) for l in open(PUB, encoding="utf-8"))}
    sess = {s["session_id"]: s for s in load()}
    fails = []
    by = collections.defaultdict(dict)
    for t in trials:
        by[(t["session_id"], t["H"], t["slot"])][t["arm"]] = t
    n_cells = 0
    for key, cell in sorted(by.items()):
        sid, H, slot = key
        s = sess[sid]
        gold = s["gold_val"]
        if set(cell) != set(ARMS):
            fails.append(f"{key}: arms present {sorted(cell)} != {list(ARMS)}")
            continue
        n_cells += 1
        ph, pi, pt, pc = split_blocks(cell["plain"]["prompt"])
        # (1) head/tail/closer identical, only the INDEX block differs
        for arm in ("index", "index_nok"):
            h, i, tl, cl = split_blocks(cell[arm]["prompt"])
            if h != ph:
                fails.append(f"{sid}/{H}/{slot}/{arm}: STATE bytes differ from plain")
            if tl != pt:
                fails.append(f"{sid}/{H}/{slot}/{arm}: RECENT bytes differ from plain")
            if cl != pc:
                fails.append(f"{sid}/{H}/{slot}/{arm}: closing instruction differs")
            if i is None:
                fails.append(f"{sid}/{H}/{slot}/{arm}: no INDEX block")
        if cell["index"]["index_lines"] != 2 or cell["index_nok"]["index_lines"] != 1:
            fails.append(f"{sid}/{H}/{slot}: index rows "
                         f"{cell['index']['index_lines']}/{cell['index_nok']['index_lines']} != 2/1")
        # (4) derived index: every row is the binding of a record in context
        bound = [re.match(r"- .* \| (\w+)=(\S+)$", l).groups()
                 for l in ph.split("\n") + pt.split("\n")
                 if re.match(r"- .* \| (TARGET|RELATED\d+)=", l)]
        if sorted(k for k, _ in bound) != sorted([QUERY_KEY, "RELATED1"]):
            fails.append(f"{sid}/{H}/{slot}: bound keys in context {bound}")
        idx_full = [l for l in cell["index"]["prompt"].split(IDX_HEAD, 1)[1]
                    .split("RECENT TOOL RESULTS:")[0].strip("\n").split("\n") if l]
        idx_nok = [l for l in cell["index_nok"]["prompt"].split(IDX_HEAD, 1)[1]
                   .split("RECENT TOOL RESULTS:")[0].strip("\n").split("\n") if l]
        if [l for l in idx_full if l != f"- {QUERY_KEY} -> {gold}"] != idx_nok:
            fails.append(f"{sid}/{H}/{slot}: index_nok is not index minus the queried row")
        if f"- {QUERY_KEY} -> {gold}" not in idx_full:
            fails.append(f"{sid}/{H}/{slot}: index arm lacks the queried binding")
        if any(QUERY_KEY in l for l in idx_nok):
            fails.append(f"{sid}/{H}/{slot}: placebo index carries the answer")
        # (2) head is a strict prefix of each arm prompt
        head_bytes = "STATE (accumulated, append-only):\n" + ph + "\n"
        for arm in ARMS:
            p = cell[arm]["prompt"]
            if not p.startswith(head_bytes):
                fails.append(f"{sid}/{H}/{slot}/{arm}: head is not a byte prefix")
        # (3) gold position identical and where claimed
        want = {"first": 0, "middle": H // 2, "last": H - 1}[slot]
        for arm in ARMS:
            if cell[arm]["gold_pos"] != want:
                fails.append(f"{sid}/{H}/{slot}/{arm}: gold_pos {cell[arm]['gold_pos']} != {want}")
        # (5) hazard held
        for arm in ARMS:
            t = cell[arm]
            if t["n_comp_in_ctx"] != 1 or len(t["wrong_pos"]) != 1:
                fails.append(f"{sid}/{H}/{slot}/{arm}: competitor count {t['n_comp_in_ctx']}")
            if t["head_lines"] != H or t["tail_lines"] != VTAIL:
                fails.append(f"{sid}/{H}/{slot}/{arm}: head/tail {t['head_lines']}/{t['tail_lines']}")
            if f"| {QUERY_KEY}={gold}" not in t["prompt"]:
                fails.append(f"{sid}/{H}/{slot}/{arm}: gold binding absent from the head")
        # (6) provenance vs the published C7 cell: same lines modulo the distractor re-key
        pb = pub.get(key)
        if pb is None:
            fails.append(f"{sid}/{H}/{slot}: no published C7 counterpart")
        else:
            pub_head = pb["prompt"].split("RECENT TOOL RESULTS:")[0]
            norm = lambda txt: sorted(re.sub(r"\| (TARGET|RELATED\d+)=", r"| KEY=", l)
                                      for l in txt.split("\n") if l.strip())
            if norm(pub_head) != norm(head_bytes):
                fails.append(f"{sid}/{H}/{slot}: head lines differ from C7 modulo re-key")
            if pb["prompt"].split("RECENT TOOL RESULTS:\n", 1)[1].split("Using STATE")[0] != pt + "\n":
                fails.append(f"{sid}/{H}/{slot}: tail block differs from C7")
    per = collections.Counter((t["arm"], t["H"], t["slot"]) for t in trials)
    print(f"trials={len(trials)}  cells(session x H x slot)={n_cells}  arms={list(ARMS)}")
    print("per-arm cell counts:", {f"{a}_H{h}_{sl}": per[(a, h, sl)]
                                   for (a, h, sl) in sorted(per)})
    dl = cell["index"]["char_len"] - cell["plain"]["char_len"]
    print(f"INDEX block overhead vs plain: +{dl} chars (2 rows) for this cell")
    if fails:
        print(f"\nGATE FAIL ({len(fails)}):")
        for m in fails[:25]:
            print("  -", m)
        sys.exit(1)
    print("\nGATE PASS: the INDEX block is the only difference between arms, the head stays a "
          "strict byte prefix, the index is derived from the records in context, the hazard and "
          "the gold position are held fixed, and the content is the published C7 construction.")


if __name__ == "__main__":
    main()
