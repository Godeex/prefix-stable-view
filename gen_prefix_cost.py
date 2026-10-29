#!/usr/bin/env python
"""B-lite: build matched-quality multi-turn prefix-cost sessions.

Mechanism probe (NOT a serving throughput benchmark). Each session is a stream of tool
observations; at every turn the agent must recall the value carried by the NEWEST record.
Two context-maintenance policies are laid out for the SAME turn, carrying the SAME newest
record + question (so both are answerable -> matched quality), differing ONLY in how the
older history is placed:

  length_only   naive recency window of the last W raw records, laid at the FRONT. The
                window slides, so the leading bytes change every turn -> prefix cache miss.
  prefix_stable M-A: an APPEND-ONLY STATE head (one stable key=value digest line per prior
                turn, never rewritten) + only the newest raw record in a volatile tail.
                Prior head bytes stay a prefix -> prefix cache hit.

The cost axis (recompute tokens / prefill latency) therefore isolates prefix stability at
matched incremental content and matched correctness. Real prose for record bodies is drawn
from LongBench hotpotqa contexts; the recall slot is a controlled key=value so correctness
is machine-checkable. Mirrors the walking_skeleton prompt layout (co/assembler.py).

Env: NSESSIONS, NTURNS, WINDOW, REC_SENTS. Resumable nothing (pure derivation).
"""
import json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LB = os.environ.get("LB", "")
if not os.path.isfile(LB):
    sys.exit("LB is required: LB=/path/to/longbench/data/hotpotqa.jsonl -- see ENVIRONMENT.md section 3")
OUT = os.path.join(HERE, "prefix_cost_trials.jsonl")

NSESSIONS = int(os.environ.get("NSESSIONS", "20"))
NTURNS = int(os.environ.get("NTURNS", "30"))       # turns per session (>= WINDOW)
WINDOW = int(os.environ.get("WINDOW", "8"))        # length_only raw window size
REC_SENTS = int(os.environ.get("REC_SENTS", "6"))  # real sentences per record body

WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()


def load_sentences():
    pool = []
    for l in open(LB, encoding="utf-8"):
        d = json.loads(l)
        ctx = d.get("context") or []
        if isinstance(ctx, str):
            ctx = [ctx]
        for part in ctx:
            txt = part if isinstance(part, str) else (part[0] if part else "")
            for s in re.split(r"(?<=[.!?])\s+", txt):
                s = s.strip()
                if 60 <= len(s) <= 240:
                    pool.append(s)
        if len(pool) > 4000:
            break
    random.seed(7)
    random.shuffle(pool)
    return pool


def norm(s):
    s = str(s).lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", "", s))


def build():
    pool = load_sentences()
    if len(pool) < REC_SENTS * 4:
        pool = [f"The system recorded observation number {i} with nominal telemetry."
                for i in range(REC_SENTS * 4)]
    p = 0

    def take(n):
        nonlocal p
        body = " ".join(pool[p:p + n])
        p += n
        return body

    out = []
    for sid in range(NSESSIONS):
        recs = []
        for t in range(NTURNS):
            key = f"OBS{t}"
            val = WORDS[t % len(WORDS)] + str(sid * 100 + t)   # distinct, controlled gold
            recs.append({"turn": t, "key": key, "val": val,
                         "body": take(REC_SENTS)})
        for t in range(WINDOW - 1, NTURNS):                   # only measure once window is full
            q = f"What value was recorded for {recs[t]['key']} in the latest observation?"
            gold = recs[t]["val"]
            # length_only: last WINDOW raw records slide at the front
            win = recs[t - WINDOW + 1:t + 1]
            lo_tail = "\n".join(f"- {r['body']} {r['key']}={r['val']}" for r in win)
            lo = (f"RECENT TOOL RESULTS:\n{lo_tail}\n"
                  f"Using the results above, answer.\nQUESTION: {q}\nANSWER:")
            # prefix_stable: append-only STATE head (prior keys) + newest raw record in tail
            head = "\n".join(f"{r['key']}={r['val']}" for r in recs[:t])   # stable prefix
            new_raw = f"- {recs[t]['body']} {recs[t]['key']}={recs[t]['val']}"
            ma = (f"STATE (accumulated, append-only):\n{head}\n"
                  f"RECENT TOOL RESULT:\n{new_raw}\n"
                  f"Using STATE and RECENT above, answer.\nQUESTION: {q}\nANSWER:")
            # pinned_raw control: SAME number of raw records as length_only and newest present
            # (matched size + matched answerability), but the leading records are the FIRST W-1
            # and never change -> only difference vs length_only is front stability, not size.
            pinned = recs[:WINDOW - 1]
            pin_tail = "\n".join(f"- {r['body']} {r['key']}={r['val']}" for r in pinned)
            pr = (f"RECENT TOOL RESULTS:\n{pin_tail}\n- {recs[t]['body']} {recs[t]['key']}={recs[t]['val']}\n"
                  f"Using the results above, answer.\nQUESTION: {q}\nANSWER:")
            out.append({"session_id": f"S{sid}", "turn": t, "window": WINDOW,
                        "key": recs[t]["key"], "gold": gold,
                        "prompt_length_only": lo, "prompt_prefix_stable": ma,
                        "prompt_pinned_raw": pr,
                        "new_tokens_approx": len((new_raw + q).split())})
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lo_toks = sum(len(r["prompt_length_only"].split()) for r in out) / len(out)
    ma_toks = sum(len(r["prompt_prefix_stable"].split()) for r in out) / len(out)
    print(f"sessions={NSESSIONS} turns={NTURNS} window={WINDOW} rec_sents={REC_SENTS}")
    print(f"rows={len(out)} -> {OUT}")
    print(f"avg prompt words: length_only={lo_toks:.0f} prefix_stable={ma_toks:.0f} "
          f"(heads grow, so totals need not match; cost = recompute)")


if __name__ == "__main__":
    build()
