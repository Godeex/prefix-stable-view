#!/usr/bin/env python
"""B: build multi-turn incremental-history sessions for the DISTRACTION x PREFIXDEPTH 2x2.

Each session is a stream of "tool observation" records; a single key TARGET is revealed deep in
the history, and NEAR-TWIN distractor records carrying a COMPETING TARGET=WRONGk value are placed
BOTH deep (will have to be folded into the cacheable head) and shallow (sit in the volatile verbatim
tail). The probe query asks for TARGET's value. A competing value present in the assembled context
=> the model can substitute it (quality failure); whether the older content is a stable append-only
head or a rebuilt sliding window => prefix cache vs re-prefill (cost). Bodies are real LongBench
passages; distractor/gold/filler assignment by bge-m3 proximity (near = hazard, far = filler).

Fidelity floor is enforced by the RUNNER (every config keeps the gold TARGET value available), so
any substitution difference is purely about *carrying competing distractor values*, not gold loss.

Env: NS (sessions), NREC, VTAIL, NEAR_T. Deterministic seed.
"""
import json, os, re, sys, random, math, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", "")
if not os.path.isdir(DATA):
    sys.exit("DATA is required: DATA=/path/to/longbench/data (the hotpotqa/2wikimqa/musique .jsonl dir) -- see ENVIRONMENT.md section 3")
FILES = ["hotpotqa.jsonl", "2wikimqa.jsonl", "musique.jsonl"]
EMB = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "bge-m3:latest"
MODE = os.environ.get("RJ_MODE", "")            # e.g. "tail"; "" = base
sfx = f"_{MODE}" if MODE else ""
OUT = os.environ.get("RJ_GEN_OUT") or os.path.join(HERE, f"joint_sessions{sfx}.jsonl")
CACHE = os.path.join(HERE, "emb_cache.json")
_c = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

NS = int(os.environ.get("NS", "30"))
NREC = int(os.environ.get("NREC", "12"))
VTAIL = int(os.environ.get("VTAIL", "5"))
SPREAD = float(os.environ.get("SPREAD", "0.06"))   # fillers must sit >= this far below the 2nd distractor
TAILMODE = os.environ.get("TAILMODE", "0") == "1"  # 1 = I-03 tail-distractor stress rework
VTAIL0 = NREC - VTAIL                              # first index of the volatile tail band
# Long-record sessions (the head-position stress) have to try many pools before one passes the
# SPREAD filter, and every tried pool is embedded in full. Bound the trials and checkpoint the
# embedding cache so an interrupted run does not lose the embeddings it already paid for.
MAX_TRIED = int(os.environ.get("MAX_TRIED", "100000"))
FLUSH_EVERY = int(os.environ.get("FLUSH_EVERY", "10"))


def embed(t):
    t = str(t)
    if t in _c:
        return _c[t]
    req = urllib.request.Request(EMB, data=json.dumps({"model": EMBED_MODEL, "input": t}).encode(),
                                 headers={"Content-Type": "application/json"})
    v = json.loads(urllib.request.urlopen(req, timeout=90).read())["embeddings"][0]
    _c[t] = v
    return v


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9)


def load_passages():
    """Per question: split its LongBench context into sentence-level records (topically
    coherent pool). Returns list of pools (each a list of sentence strings)."""
    out = []
    for fn in FILES:
        p = os.path.join(DATA, fn)
        if not os.path.exists(p):
            continue
        for l in open(p, encoding="utf-8"):
            d = json.loads(l)
            ctx = d.get("context") or ""
            ctx = re.sub(r"Passage \d+:", " ", ctx if isinstance(ctx, str) else " ".join(map(str, ctx)))
            sents = [re.sub(r"\s+", " ", s).strip()
                     for s in re.split(r"(?<=[.!?])\s+", ctx)]
            sents = [s for s in sents if 70 <= len(s) <= 260]
            if len(sents) >= NREC + 4:
                out.append(sents)
            if len(out) >= NS * 2:
                break
        if len(out) >= NS * 2:
            break
    return out


def main():
    random.seed(20260923)
    pools = load_passages()
    sess = []
    tried = 0
    si = 0
    for parts in pools:
        tried += 1
        if tried > MAX_TRIED:
            print(f"[gen] stopped after MAX_TRIED={MAX_TRIED} pools", flush=True)
            break
        g = parts[0]
        gv = embed(g)
        # rank-based within the topically-coherent pool: absolute bge-m3 cos is unreliable across
        # domains (random-pair max ~0.49, same-topic pools sit higher), so "near/far" = relative.
        cands = sorted(((cos(embed(x), gv), x) for x in parts[1:]), key=lambda z: -z[0])
        if len(cands) < NREC:
            continue
        nears = cands[:2]                       # two most-confusable non-gold -> substitution hazard
        cutoff = nears[-1][0] - SPREAD          # fillers must be clearly LESS confusable
        fars = [x for c, x in cands if c <= cutoff]
        need_f = NREC - 3                        # 3 specials = gold + 2 distractors
        if len(fars) < need_f:
            continue
        right = f"R{si*13+7}"
        wrong = [f"W{si*13+1}", f"W{si*13+2}"]
        d1, d2 = nears[0][1], nears[1][1]          # two most-confusable (near-twin) distractors
        fillers = fars[:need_f]
        recs = []
        probe = None
        if TAILMODE:
            # I-03 tail-distractor stress: GOLD placed deep (idx2 -> stable head for p=1), BOTH
            # competing distractors placed INSIDE the volatile recent tail band so no layout can move
            # them away from the query. We probe only the turns where a deep gold coexists with a
            # tail distractor in the recent window (the regime where only value-folding d helps).
            pos = [None] * NREC
            pos[2] = (g, right)
            dpos = [VTAIL0, VTAIL0 + 2]            # two distractor slots inside the tail band
            pos[dpos[0]] = (d1, wrong[0])
            pos[dpos[1]] = (d2, wrong[1])
            fit = iter(fillers)
            for i in range(NREC):
                if pos[i] is None:
                    pos[i] = (next(fit), None)
            order = pos
            probe = (NREC - 5, NREC - 1)           # 5 mature turns, tail distractor present
        else:
            # ordered records: fillers, GOLD(deep), DEEP-distractor, more fillers, SHALLOW-distractor
            order = ([(x, None) for x in fillers[:2]]
                     + [(g, right), (d1, wrong[0])]
                     + [(x, None) for x in fillers[2:]]
                     + [(d2, wrong[1])])
        for i, (body, val) in enumerate(order):
            line = f"TARGET={val}" if val is not None else f"NOTE=aux{i}"
            recs.append({"idx": i, "body": body, "line": line,
                         "value": val, "is_gold": val == right,
                         "is_distractor": val is not None and val != right})
        if len(recs) < NREC:
            continue
        sess.append({"session_id": f"J{si}", "records": recs, "gold_val": right,
                     "wrong_vals": wrong, "query": "What is the recorded value for TARGET?",
                     "budget_v": VTAIL, "n_rec": len(recs), "probe_range": probe,
                     "distractor_sim": [round(nears[0][0], 3), round(nears[1][0], 3)],
                     "max_filler_sim": round(cutoff, 3)})
        si += 1
        if si % FLUSH_EVERY == 0:
            with open(OUT, "w", encoding="utf-8") as fh:
                for s in sess:
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
            json.dump(_c, open(CACHE, "w"))
            print(f"[gen] {si}/{NS} sessions from {tried} pools (checkpointed)", flush=True)
        if si >= NS:
            break
    with open(OUT, "w", encoding="utf-8") as f:
        for s in sess:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    json.dump(_c, open(CACHE, "w", encoding="utf-8"))
    gold_pos = [next(r["idx"] for r in s["records"] if r["is_gold"]) for s in sess]
    deep_d = [min((r["idx"] for r in s["records"] if r["is_distractor"]), default=-1) for s in sess]
    dsims = [d for s in sess for d in s["distractor_sim"]]
    print(f"sessions={len(sess)} (tried {tried} pools) n_rec~{NREC} v_tail={VTAIL} spread={SPREAD}")
    print(f"gold position range {min(gold_pos)}-{max(gold_pos)}; deep-distractor min idx {min(deep_d)}")
    if dsims:
        print(f"distractor-to-gold cos: min {min(dsims):.3f} med {sorted(dsims)[len(dsims)//2]:.3f} "
              f"max {max(dsims):.3f} (rank-selected near-twins; hazard strength reported honestly)")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
