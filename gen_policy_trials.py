#!/usr/bin/env python
"""E4: real-load matched-token THREE-ARM generator (natural MTU-Bench fields).

For each (base, K) we show gold + K distractors, but WHICH K distractors differs by arm:
  - length_only : first K distractors in original field order (arbitrary, non-semantic) = what a
                  budget truncation keeps when it ignores content.
  - m_a         : the K distractors with LOWEST semantic proximity to gold (the distraction-aware
                  controller drops the near ones).
  - worst       : the K distractors with HIGHEST proximity (upper bound; also spans nsim for beta refit).
All three blocks are padded/truncated to an EXACTLY equal character length => matched tokens, so the
only difference across arms is the semantic nearness of the retained distractors.

This is the main efficacy arm: it tests whether the M-A selection rule lowers contamination at
EQUAL token cost, and refits the distraction->contamination calibration on NATURAL field values.
"""
import json, os, sys, re, ast
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from co.embedder import Embedder

MTU = os.environ.get("MTU", "")
if not os.path.isfile(MTU):
    sys.exit("MTU is required: MTU=/path/to/mtu/M-S_eval.jsonl -- see ENVIRONMENT.md section 3")
KMAX = 3
MIN_FIELDS = 6
MAX_BASES = int(os.environ.get("MA_MAX_BASES", "30"))
CHAR_TARGET = 300
OUT = os.path.join(HERE, "trials_policy.jsonl")
emb = Embedder(cache_path=os.path.join(HERE, "emb_cache.json"))

FILLER_KEYS = ["id", "status", "lang", "region", "ver", "tag", "note", "rev", "code", "unit"]
FILLER_VALS = ["ok", "1", "en-US", "north", "v3", "ref", "active", "2024"]


def obs_dicts(q):
    out = []
    for m in re.finditer(r"observation:\s*:?\s*(\{[^{}]*\})", q):
        try:
            d = ast.literal_eval(m.group(1))
            if isinstance(d, dict) and len(d) >= MIN_FIELDS:
                out.append(d)
        except Exception:
            pass
    return out


def block(gold_k, gold_v, distractors, exact_len):
    fields = [(gold_k, gold_v)] + list(distractors)
    s = ", ".join(f"'{k}': {repr(v)}" for k, v in fields)
    fi = 0
    while len(s) < exact_len and fi < 60:
        s += f", '{FILLER_KEYS[fi % len(FILLER_KEYS)]}': {repr(FILLER_VALS[fi % len(FILLER_VALS)])}"
        fi += 1
    if len(s) < exact_len:
        s += f", 'filler': '{'x' * max(0, exact_len - len(s) - 15)}'"
    if len(s) > exact_len:
        s = s[:exact_len - 1]
    return "{" + s + "}"


def prompt_of(blk, gold_k):
    return ("You are given the JSON result returned by a tool call. Report the value of ONE "
            "requested field exactly as it appears. Answer with only the value, nothing else.\n\n"
            f"Tool result: {blk}\n\n"
            f"Question: What is the value of the field '{gold_k}'?")


def main():
    recs = [json.loads(l) for l in open(MTU)]
    trials, tid, nbases = [], 0, 0
    for r in recs:
        if nbases >= MAX_BASES:
            break
        obs = obs_dicts(r.get("question", ""))
        if not obs:
            continue
        D = max(obs, key=len)
        seen, items = set(), []
        for k, v in D.items():
            v = str(v).strip()
            if v and v not in seen:
                seen.add(v); items.append((k, v))
        if len(items) < MIN_FIELDS:
            continue
        # choose a gold target that has BOTH a near and a far distractor available (real contrast)
        chosen = None
        for gk, gv in items:
            others = [(k2, v2) for k2, v2 in items if k2 != gk]
            sims = [(emb.cos(emb.embed(gv), emb.embed(v2)), k2, v2) for k2, v2 in others]
            sims.sort()
            if len(sims) >= 2 and (sims[-1][0] - sims[0][0]) >= 0.10 and sims[-1][0] >= 0.35:
                chosen = (gk, gv, sims); break
        if not chosen:
            continue
        gk, gv, sims = chosen
        ordered_others = [k2 for k2, _ in items if k2 != gk]     # original field order (dict)
        simmap = {k2: s for s, k2, _ in sims}
        valmap = {k2: v2 for k2, v2 in items}
        nbases += 1
        for K in range(1, min(KMAX, len(sims)) + 1):
            by_near = [ (k2, valmap[k2]) for k2 in sorted(simmap, key=lambda x: simmap[x]) ]
            length_only = [(k2, valmap[k2]) for k2 in ordered_others[:K]]
            m_a = by_near[:K]                       # lowest proximity (drop the near ones)
            worst = by_near[-K:][::-1]             # highest proximity
            nat = len(", ".join(f"'{k}': {repr(v)}" for k, v in [(gk, gv)] + length_only))
            exact = max(CHAR_TARGET, nat)
            for arm, dis in (("length_only", length_only), ("m_a", m_a), ("worst", worst)):
                blk = block(gk, gv, dis, exact)
                dvals = [v for _, v in dis]
                dsims = [simmap[k] for k, _ in dis]
                trials.append({
                    "trial_id": f"P{tid}", "base_id": r["id"], "gold_key": gk, "gold_val": gv,
                    "K": K, "arm": arm, "shown_distractors": [k for k, _ in dis],
                    "distractor_vals": dvals,
                    "distractor_mean_sim": round(sum(dsims)/len(dsims), 4),
                    "distractor_max_sim": round(max(dsims), 4),
                    "n_fields_shown": 1 + K, "char_len": len(prompt_of(blk, gk)),
                    "prompt": prompt_of(blk, gk),
                })
                tid += 1
    # matched-token assertion: per (base,K) the three arms must be char-equal
    import collections
    grp = collections.defaultdict(list)
    for t in trials:
        grp[(t["base_id"], t["K"])].append(t["char_len"])
    worst_spread = max(max(v) - min(v) for v in grp.values())
    with open(OUT, "w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    emb.flush()
    nsim_by_arm = collections.defaultdict(list)
    for t in trials:
        nsim_by_arm[t["arm"]].append(t["distractor_mean_sim"])
    print(f"wrote {len(trials)} trials / {len(grp)} (base,K) groups; char spread={worst_spread}")
    for arm in ("length_only", "m_a", "worst"):
        a = nsim_by_arm[arm]
        print(f"  {arm:11s} mean nsim={sum(a)/len(a):.3f}")


if __name__ == "__main__":
    main()
