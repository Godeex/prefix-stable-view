"""Diagnose the ``correct`` column of the 2x2 layout study.

The 2x2 reports substitution, recompute and a verbatim-match correctness score.
Correctness here requires the record id (e.g. ``R20``) in the answer, so it can
fall for reasons that are not fidelity losses. This script splits every answered
cell into four disjoint categories and reports the paired McNemar contrast that
the manuscript quotes. No model call: everything is read from the scored file.

Usage: python analyze_2x2_correct.py [scored.jsonl ...]
"""
import json
import os
import re
import sys
import collections
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = [os.path.join(HERE, "joint_scored_qwen3_14b.jsonl")]
CFGS = ["c00", "c10", "c01", "c11"]
BOILERPLATE = re.compile(r"^the recorded value for target[^:]*:\s*", re.I)


def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def shingles(text, n=4):
    words = norm(text).split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def load_sessions(path=None):
    path = path or os.path.join(HERE, "joint_sessions.jsonl")
    out = {}
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            out[row["session_id"]] = row["records"]
    return out


def categorize(answer, records):
    """id | content-gold | content-dist | other, on the gold record's own text."""
    gold = next(r for r in records if r["is_gold"])
    distractors = [r for r in records if r["is_distractor"]]
    if gold["value"] in answer:
        return "id"
    body = BOILERPLATE.sub("", answer)
    got = shingles(body)
    if not got:
        return "other"
    over_gold = len(got & shingles(gold["body"]))
    over_dist = max((len(got & shingles(r["body"])) for r in distractors), default=0)
    if over_gold >= 3 and over_gold > over_dist:
        return "content-gold"
    if over_dist >= 3 and over_dist > over_gold:
        return "content-dist"
    return "other"


def mcnemar(pairs):
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = lambda k: sum(comb(n, i) for i in range(k, n + 1)) * 0.5 ** n
    return b, c, min(1.0, 2 * min(tail(b), tail(c)))


def report(path, sessions):
    by = collections.defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            by[row["cfg"]][(row["session_id"], row["turn"])] = row
    print(f"== {path} ==")
    stats = {}
    for cfg in CFGS:
        cells = by[cfg]
        if not cells:
            continue
        cats = collections.Counter(
            categorize(r["answer"], sessions[r["session_id"]]) for r in cells.values()
        )
        n = len(cells)
        corr = sum(r["correct"] for r in cells.values())
        subst = sum(r["substitution"] for r in cells.values())
        stats[cfg] = (cells, corr / n, subst / n)
        print(
            f"{cfg} n={n} correct={corr / n:.3f} substitution={subst / n:.3f} "
            + " ".join(f"{k}={cats[k] / n:.3f}" for k in ("id", "content-gold", "content-dist", "other"))
        )
    for a, b in (("c01", "c11"), ("c10", "c11")):
        if a not in stats or b not in stats:
            continue
        ca, _, _ = stats[a]
        cb, _, _ = stats[b]
        shared = sorted(set(ca) & set(cb))
        m, s, p = mcnemar([(ca[k]["correct"], cb[k]["correct"]) for k in shared])
        print(f"paired correct {a}->{b}: n={len(shared)} b={m} c={s} exact McNemar p={p:.3g}")
        m2, s2, p2 = mcnemar([(ca[k]["substitution"], cb[k]["substitution"]) for k in shared])
        print(f"paired subst   {a}->{b}: b={m2} c={s2} p={p2:.3g}")


if __name__ == "__main__":
    paths = sys.argv[1:] or DEFAULT
    sessions = load_sessions()
    for p in paths:
        report(p, sessions)
