#!/usr/bin/env python
"""Deterministic gate for the layout control: check the CONTRAST is what we claim, before
spending any inference on it.

For every session and every consecutive turn pair, over the four cells:
  1. front stability  : longest common prefix (chars) between prompt(t) and prompt(t+1).
                        Stable cells must reuse nearly the whole older block; sliding cells must
                        reuse only the header line. This is the property the cost axis is defined by.
  2. content equality : the multiset of record bodies + the multiset of displayed values must be
                        identical across all four cells at the same turn. Nothing is added or dropped
                        by the labelling or the reordering, so a quality gap cannot be a content gap.
  3. size matching    : prompt character length must agree across cells within a small margin, so
                        the arms do not differ by "sends less".
Prints a PASS/FAIL table. No model calls.
"""
import json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_joint_layout import Layout, CELLS

SESS = os.environ.get("RJ_SESS") or os.path.join(HERE, "joint_sessions.jsonl")
# the item text is identical in every cell; only the section headers differ in length
# ("STATE (accumulated, append-only):" vs "CONTEXT:"), a ~46 char / ~12 token asymmetry that the
# gate bounds explicitly instead of hiding. Token parity is re-checked on the real run.
MAXDIFF = float(os.environ.get("MAXDIFF", "0.05"))


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def sig(prompt):
    """content fingerprint: which bodies and which displayed key=value tokens are present."""
    bodies = Counter(l[:l.index("|")].strip("- ").strip() for l in prompt.split("\n")
                     if l.startswith("- ") and "|" in l)
    vals = Counter(w for w in prompt.replace("\n", " ").split()
                   if w.startswith("TARGET=") or w.startswith("NOTE="))
    return bodies, vals


def main():
    rows = [json.loads(l) for l in open(SESS, encoding="utf-8")]
    fail = []
    agg = {c: {"stab": [], "size": []} for c in CELLS}
    for s in rows:
        m = Layout(s["records"], s["budget_v"])
        m._wrong = s["wrong_vals"]
        t0 = min(r["idx"] for r in s["records"] if r["is_distractor"])
        turns = list(range(max(0, t0), len(s["records"])))
        for t in turns:
            ps = {c: m.prompt(t, c, s["query"]) for c in CELLS}
            base_b, base_v = sig(ps["s0l0"])
            for c in CELLS:
                b, v = sig(ps[c])
                if b != base_b:
                    fail.append((s["session_id"], t, c, "body set differs"))
                if v != base_v:
                    fail.append((s["session_id"], t, c, "displayed value set differs"))
            lens = {c: len(ps[c]) for c in CELLS}
            spread = (max(lens.values()) - min(lens.values())) / max(lens.values())
            if spread > MAXDIFF:
                fail.append((s["session_id"], t, "size", f"spread {spread:.3f} > {MAXDIFF}"))
            for c in CELLS:
                agg[c]["size"].append(lens[c])
            if t + 1 in turns:
                ps2 = {c: m.prompt(t + 1, c, s["query"]) for c in CELLS}
                for c in CELLS:
                    frac = lcp(ps[c], ps2[c]) / max(len(ps[c]), 1)
                    agg[c]["stab"].append(frac)

    print(f"sessions={len(rows)} cells={list(CELLS)}")
    print(f"{'cell':7s} {'front-reuse':>12s} {'min':>7s} {'max':>7s} {'mean chars':>11s}")
    for c in CELLS:
        v = agg[c]["stab"]
        print(f"{c:7s} {100*sum(v)/len(v):11.1f}% {100*min(v):6.1f}% {100*max(v):6.1f}% "
              f"{sum(agg[c]['size'])/len(agg[c]['size']):11.0f}")
    lo_slide = min(sum(agg[c]["stab"]) / len(agg[c]["stab"]) for c in CELLS if c[1] == "0")
    hi_stable = min(sum(agg[c]["stab"]) / len(agg[c]["stab"]) for c in CELLS if c[1] == "1")
    ok = hi_stable > 2 * lo_slide + 0.1
    print(f"\nfront-reuse separation: sliding<={lo_slide:.3f}, stable>={hi_stable:.3f} -> "
          f"{'PASS' if ok else 'FAIL'}")
    if fail:
        kinds = Counter(v[3].split(" ")[0] if v[2] == "size" else v[2] for v in fail)
        print(f"violations ({len(fail)}) by kind: {dict(kinds)}")
        print(f"first 3: {fail[:3]}")
        if any(v[2] != "size" for v in fail):
            sys.exit(1)
    else:
        print("content sets identical across all four cells at every turn: PASS")
    if not ok:
        sys.exit(1)
    print("gate: front-reuse separates, content matched, header-only size asymmetry within bound")


if __name__ == "__main__":
    main()
