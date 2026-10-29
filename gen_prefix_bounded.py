#!/usr/bin/env python
"""B-lite-bounded: add the finite-head (fold/evict-within-head) arm to the prefix-cost probe.

Why this exists. The published prefix_stable arm (gen_prefix_cost.py) uses an UNBOUNDED
append-only STATE head: one stable `key=val` digest line per prior turn, never rewritten. That
maximizes prefix cache hits but its head grows linearly with session length, so a long-running
agent would eventually exhaust the KV budget. The natural question is "what if you never evict --
does the head grow without bound?" The honest answer is: a real policy folds/evicts *within* the
head under a budget. This script measures what that costs.

New arm `bounded_stable` = prefix_stable but with a head capped at H_MAX digest lines. Between
folds the head is append-only (stable prefix, low recompute -- like prefix_stable). When it would
exceed H_MAX, the oldest lines are FOLDED into a single rolled-up summary line (a "fold event"):
this REWRITES the head's leading bytes, so the prefix cache is invalidated at exactly that turn
(one recompute spike), then the head grows again until the next fold. The result is a SAWTOOTH:
low recompute on append turns, a spike on fold turns. This is the mechanism a real budgeted
history-maintenance operator pays; we quantify it rather than hand-wave it.

Kept for direct comparison at the SAME turns/records: length_only (naive, slides every turn =
recompute every turn), prefix_stable (unbounded head = the idealized ceiling reported in the
paper). All arms carry the SAME newest record + question, so all are answerable (matched quality).

Env: NSESSIONS, NTURNS, WINDOW, REC_SENTS, H_MAX, FOLD_KEEP. Resumable nothing (pure derivation).
"""
import json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LB = os.environ.get("LB", "")
if not os.path.isfile(LB):
    sys.exit("LB is required: LB=/path/to/longbench/data/hotpotqa.jsonl -- see ENVIRONMENT.md section 3")
OUT = os.path.join(HERE, "prefix_bounded_trials.jsonl")

NSESSIONS = int(os.environ.get("NSESSIONS", "12"))
NTURNS = int(os.environ.get("NTURNS", "40"))       # long enough to force several folds
WINDOW = int(os.environ.get("WINDOW", "8"))        # length_only raw window size
REC_SENTS = int(os.environ.get("REC_SENTS", "6"))  # real sentences per record body
H_MAX = int(os.environ.get("H_MAX", "10"))         # max digest lines the head may hold
FOLD_KEEP = int(os.environ.get("FOLD_KEEP", "4"))  # lines kept after a fold (rest -> 1 summary)

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
            val = WORDS[t % len(WORDS)] + str(sid * 100 + t)
            recs.append({"turn": t, "key": key, "val": val, "body": take(REC_SENTS)})

        # Replay the bounded head as a small state machine so fold events are exactly where a
        # real operator would trigger them (head would overflow), and we can tag each turn.
        head_lines = []          # current digest lines in the head
        folded_summaries = 0     # how many fold events have fired so far
        for t in range(WINDOW - 1, NTURNS):
            q = f"What value was recorded for {recs[t]['key']} in the latest observation?"
            gold = recs[t]["val"]

            # ---- length_only: last WINDOW raw records slide at the front (naive) ----
            win = recs[t - WINDOW + 1:t + 1]
            lo_tail = "\n".join(f"- {r['body']} {r['key']}={r['val']}" for r in win)
            lo = (f"RECENT TOOL RESULTS:\n{lo_tail}\n"
                  f"Using the results above, answer.\nQUESTION: {q}\nANSWER:")

            # ---- prefix_stable: UNBOUNDED append-only head (the paper's idealized ceiling) ----
            ub_head = "\n".join(f"{r['key']}={r['val']}" for r in recs[:t])
            new_raw = f"- {recs[t]['body']} {recs[t]['key']}={recs[t]['val']}"
            ma = (f"STATE (accumulated, append-only):\n{ub_head}\n"
                  f"RECENT TOOL RESULT:\n{new_raw}\n"
                  f"Using STATE and RECENT above, answer.\nQUESTION: {q}\nANSWER:")

            # ---- bounded_stable: head capped at H_MAX, folds oldest -> 1 summary line ----
            # Advance the state machine one turn (append this turn's digest line), then fold if
            # it would overflow. The fold rewrites the head -> that turn is a cache-miss spike.
            head_lines.append(f"{recs[t]['key']}={recs[t]['val']}")
            is_fold = 0
            if len(head_lines) > H_MAX:
                # fold the oldest (len - FOLD_KEEP) lines into one summary digest line
                oldest = head_lines[:len(head_lines) - FOLD_KEEP]
                keep = head_lines[len(head_lines) - FOLD_KEEP:]
                folded_summaries += 1
                summary = (f"SUMMARY{folded_summaries}: "
                           + ", ".join(l for l in oldest)
                           + f". (folded {len(oldest)} earlier observations)")
                head_lines = [summary] + keep
                is_fold = 1
            bd_head = "\n".join(head_lines)
            mb = (f"STATE (budgeted, append-only since last fold):\n{bd_head}\n"
                  f"RECENT TOOL RESULT:\n{new_raw}\n"
                  f"Using STATE and RECENT above, answer.\nQUESTION: {q}\nANSWER:")

            out.append({"session_id": f"S{sid}", "turn": t, "window": WINDOW,
                        "h_max": H_MAX, "fold_keep": FOLD_KEEP, "fold_event": is_fold,
                        "head_len": len(head_lines),
                        "key": recs[t]["key"], "gold": gold,
                        "prompt_length_only": lo, "prompt_prefix_stable": ma,
                        "prompt_bounded_stable": mb,
                        "new_tokens_approx": len((new_raw + q).split())})
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    folds = sum(r["fold_event"] for r in out)
    print(f"sessions={NSESSIONS} turns={NTURNS} window={WINDOW} rec_sents={REC_SENTS} "
          f"h_max={H_MAX} fold_keep={FOLD_KEEP}")
    print(f"rows={len(out)} -> {OUT}")
    print(f"bounded fold-events across all sessions/turns = {folds} "
          f"(turns between folds ~ {H_MAX - FOLD_KEEP + 1})")
    lo_w = sum(len(r["prompt_length_only"].split()) for r in out) / len(out)
    ma_w = sum(len(r["prompt_prefix_stable"].split()) for r in out) / len(out)
    mb_w = sum(len(r["prompt_bounded_stable"].split()) for r in out) / len(out)
    print(f"avg prompt words: length_only={lo_w:.0f} prefix_stable(unbounded)={ma_w:.0f} "
          f"bounded_stable={mb_w:.0f}  (bounded head stays ~flat while unbounded grows)")


if __name__ == "__main__":
    build()
