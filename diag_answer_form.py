#!/usr/bin/env python
"""POST-HOC diagnostic (added after the run, not part of the pre-declared reading).

The published position stress scores every answer into correct / substitution / other by exact
value-token matching, and the three arms of the index leg use the same buckets unchanged. Under the
re-keyed construction `substitution` is zero in every cell, so the position loss lands entirely in
`other` -- and `other` as published lumps together two different failures: the model restating the
RIGHT record in prose instead of echoing its id, and the model answering about a DIFFERENT record or
nothing at all. Which of those it is changes how the result should be read, so this script splits the
`other` bucket by text similarity to the head records it was asked about.

Method (deterministic, no model call): for each scored row whose bucket is `other`, take the record
lines of that trial's STATE head, build the set of word 5-grams of each record body, and count how
many appear in the answer. The answer is attributed to the record with the strictly largest count
when that count reaches MIN_SHARED, otherwise to "neither". Runs over any (trials, scored) pair, so
the same split is reported for the published shared-key stress.

Exposes nothing used by a pre-declared contrast: no P1/P2/P3 statistic, arm assignment or table cell
depends on it. It is a characterization of the failures, reported as exploratory.
"""
import argparse, collections, json, os, re

MIN_SHARED = 3


def head_records(prompt):
    """(value id -> (key, body)) for the record lines of the STATE head."""
    txt = prompt.split("STATE (accumulated, append-only):\n", 1)[1]
    txt = re.split(r"\nINDEX|RECENT TOOL RESULTS:", txt)[0]
    out = {}
    for line in txt.split("\n"):
        m = re.match(r"- (.*?) \| (TARGET|RELATED\d+)=(\S+)$", line)
        if m:
            out[m.group(3)] = (m.group(2), m.group(1))
    return out


def shingles(s, n=5):
    w = re.findall(r"\w+", s.lower())
    return {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def classify(row, trial):
    recs = head_records(trial["prompt"])
    ans = shingles(row["answer"])
    score = {val: len(shingles(body) & ans) for val, (key, body) in recs.items()}
    if not score:
        return "no record in head", 0
    best = max(score.values())
    if best < MIN_SHARED:
        return "neither (id echoed elsewhere / refusal / unrelated)", best
    top = [v for v, s in score.items() if s == best]
    if len(top) > 1:
        return "ambiguous (tied bodies)", best
    return ("gold record restated in prose" if top[0] == row["gold"]
            else "another record restated in prose"), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    T = {}
    for l in open(args.trials, encoding="utf-8"):
        t = json.loads(l)
        T[t["trial_id"]] = t
    rows = [json.loads(l) for l in open(args.scored, encoding="utf-8")]

    by = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    examples = collections.defaultdict(list)
    for r in rows:
        arm = r.get("arm", "position")
        if r["correct"]:
            bucket = "correct (id echoed)"
        elif r["substitution"]:
            bucket = "substitution (competing value named)"
        else:
            bucket, _ = classify(r, T[r["trial_id"]])
        by[(arm, r["H"], r["slot"])][bucket] += 1
        tot[(arm, bucket)] += 1
        if bucket.startswith(("neither", "ambiguous", "another")) and len(examples[bucket]) < 3:
            examples[bucket].append({"trial_id": r["trial_id"], "answer": r["answer"][:140]})

    print(f"== {os.path.basename(args.scored)} :: answer-form split of the non-correct answers "
          f"(POST-HOC, min {MIN_SHARED} shared word 5-grams) ==")
    for k in sorted(by):
        n = sum(by[k].values())
        line = "  ".join(f"{b.split(' (')[0]}={c}" for b, c in sorted(by[k].items(), key=lambda x: -x[1]))
        print(f"  {k[0]:10s} H={k[1]:<3d} {k[2]:7s} n={n:4d}  {line}")
    print("\n  per arm over all cells:")
    for (arm, bucket), c in sorted(tot.items()):
        n = sum(v for (a, b), v in tot.items() if a == arm)
        print(f"    {arm:10s} {bucket:52s} {c:5d} / {n:5d} = {c / n:.3f}")
    for b, ex in examples.items():
        print(f"\n  examples of {b!r}:")
        for e in ex:
            print(f"    {e['trial_id']}: {e['answer']!r}")
    if args.json_out:
        json.dump({f"{a}|H{h}|{s}": dict(c) for (a, h, s), c in by.items()},
                  open(args.json_out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
