#!/usr/bin/env python
"""Cost leg: render the rewritable current-binding INDEX on the PUBLISHED cost sessions.

The fidelity leg (gen_index_trials.py) shows what a near-end binding block buys; this leg measures
what it costs. Same sessions, turns, bodies, keys and values as the published prefix-cost study
(gen_prefix_cost.py, NSESSIONS=20 NTURNS=30 WINDOW=8 REC_SENTS=6, LongBench hotpotqa sentences,
seed 7): the script re-derives those rows and ASSERTS byte-for-byte equality with
prefix_cost_trials.jsonl before emitting anything, so the new arms are the published stream with
one block inserted, not a new stream.

Arms rendered per turn (t = 7..29, the published window; the newest record is in the tail in every
arm, so all four are answerable and accuracy stays matched):
  prefix_stable   STATE (append-only digest OBS0..OBS_{t-1}) || RECENT (newest raw)   [published]
  index_I2        STATE || INDEX (the I most recent bindings, incl. the queried key) || RECENT
  index_I8        same, I = 8
  index_I24       same, I = 24
  length_only     sliding raw window of 8 at the front                                [published]

Cost model prediction: the head bytes of index_I* are IDENTICAL to prefix_stable and a strict
prefix of the prompt, so a rewrite invalidates only the index block plus the tail:
  R(index_I) = Theta(I) + R(prefix_stable),  whereas relocating the answer inside the head costs
Theta(H) per move and a bounded fold the Theta(H_max * delta) sawtooth. The run measures the slope
in tokens per index row.

Deterministic, no model calls. Env: IC_OUT.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_prefix_cost as G

OUT = os.environ.get("IC_OUT") or os.path.join(HERE, "index_cost_trials.jsonl")
PUB = os.path.join(HERE, "prefix_cost_trials.jsonl")
IDX_HEAD = "INDEX (current bindings, rewritable):\n"
ILEVELS = (2, 8, 24)


def rows():
    """Re-derive the published sessions (same seeds, same params) -> per-turn record lists."""
    pool = G.load_sentences()
    if len(pool) < G.REC_SENTS * 4:
        pool = [f"The system recorded observation number {i} with nominal telemetry."
                for i in range(G.REC_SENTS * 4)]
    pos = [0]

    def take(n):
        body = " ".join(pool[pos[0]:pos[0] + n]); pos[0] += n
        return body

    sess = {}
    for sid in range(G.NSESSIONS):
        recs = []
        for t in range(G.NTURNS):
            recs.append({"turn": t, "key": f"OBS{t}", "val": G.WORDS[t % len(G.WORDS)] + str(sid * 100 + t),
                         "body": take(G.REC_SENTS)})
        sess[f"S{sid}"] = recs
    return sess


def render(recs, t):
    q = f"What value was recorded for {recs[t]['key']} in the latest observation?"
    new_raw = f"- {recs[t]['body']} {recs[t]['key']}={recs[t]['val']}"
    head = "\n".join(f"{r['key']}={r['val']}" for r in recs[:t])
    stable = (f"STATE (accumulated, append-only):\n{head}\n"
              f"RECENT TOOL RESULT:\n{new_raw}\n"
              f"Using STATE and RECENT above, answer.\nQUESTION: {q}\nANSWER:")
    win = recs[t - G.WINDOW + 1:t + 1]
    lo_tail = "\n".join(f"- {r['body']} {r['key']}={r['val']}" for r in win)
    length_only = (f"RECENT TOOL RESULTS:\n{lo_tail}\n"
                   f"Using the results above, answer.\nQUESTION: {q}\nANSWER:")
    out = {"prefix_stable": stable, "length_only": length_only}
    for I in ILEVELS:
        rows_idx = "\n".join(f"- {r['key']} -> {r['val']}" for r in recs[max(0, t - I + 1):t + 1])
        out[f"index_I{I}"] = (f"STATE (accumulated, append-only):\n{head}\n" + IDX_HEAD + rows_idx + "\n"
                              f"RECENT TOOL RESULT:\n{new_raw}\n"
                              f"Using the blocks above, answer.\nQUESTION: {q}\nANSWER:")
    return out


def main():
    sess = rows()
    pub = {(r["session_id"], r["turn"]): r for r in
           (json.loads(l) for l in open(PUB, encoding="utf-8"))}
    out, mismatch = [], []
    for sid, recs in sess.items():
        for t in range(G.WINDOW - 1, G.NTURNS):
            r = render(recs, t)
            p = pub.get((sid, t))
            if p is None:
                mismatch.append(f"{sid} t{t}: absent from the published trial file")
                continue
            if r["prefix_stable"] != p["prompt_prefix_stable"]:
                mismatch.append(f"{sid} t{t}: prefix_stable bytes differ from published")
            if r["length_only"] != p["prompt_length_only"]:
                mismatch.append(f"{sid} t{t}: length_only bytes differ from published")
            row = {"session_id": sid, "turn": t, "window": G.WINDOW, "key": recs[t]["key"],
                   "gold": recs[t]["val"], "new_tokens_approx": p["new_tokens_approx"]}
            row["prompt_prefix_stable"] = r["prefix_stable"]
            row["prompt_length_only"] = r["length_only"]
            # the append-only head is a SHARED byte prefix of the baseline and of every index arm,
            # and the first divergence sits exactly where the index block starts
            shared = r["prefix_stable"].split("RECENT TOOL RESULT:")[0]
            for I in ILEVELS:
                arm = f"index_I{I}"
                row[f"prompt_{arm}"] = r[arm]
                d = next(i for i in range(len(r[arm])) if r[arm][i] != r["prefix_stable"][i])
                if d != len(shared):
                    mismatch.append(f"{sid} t{t}: {arm} diverges at byte {d}, head ends at {len(shared)}")
            row["head_bytes"] = len(shared)
            out.append(row)
    if mismatch:
        print(f"PROVENANCE FAIL ({len(mismatch)}):")
        for m in mismatch[:10]:
            print("  -", m)
        sys.exit(1)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sizes = {}
    for r in out:
        for arm in ("prefix_stable", "length_only") + tuple(f"index_I{I}" for I in ILEVELS):
            sizes.setdefault(arm, []).append(len(r[f"prompt_{arm}"]))
    print(f"rows={len(out)} sessions={G.NSESSIONS} turns={G.WINDOW - 1}..{G.NTURNS - 1} -> {OUT}")
    print("provenance: prefix_stable and length_only bytes equal the published trial file (asserted)")
    base = sum(sizes["prefix_stable"]) / len(out)
    for arm, v in sizes.items():
        mean = sum(v) / len(v)
        print(f"  {arm:14s} mean chars {mean:8.1f}  ({mean - base:+7.1f} vs prefix_stable)")


if __name__ == "__main__":
    main()
