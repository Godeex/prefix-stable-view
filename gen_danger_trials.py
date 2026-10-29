#!/usr/bin/env python
"""(A) DANGER-SLICE generator for the conditional-benefit test.

Motivation from (1): at matched token budget, naive length_only truncation also gave ~0%
contamination on this base set, because in natural field order it rarely RETAINS a semantic
near-twin of gold. So the honest claim to test is NOT "M-A always beats truncation" but
"M-A matters exactly WHEN the naive policy happens to retain a near-twin". This generator makes
that regime explicit and measures its natural prevalence cheaply (embeddings only, no LLM call).

For each (base,K) group we compute, before any inference:
  - lo_max_sim : max semantic proximity among the distractors length_only[:K] retains
  - ma_max_sim : max proximity among the K LOWEST-proximity distractors (m_a drops the near ones)
A group is a DANGER group when lo_max_sim >= MA_DANGER_T, i.e. the naive truncation DID keep a
near-twin that a distraction-aware controller would evict. We emit the paired arms
(length_only, m_a) for those groups; both blocks padded to EQUAL char length => matched token.

Also prints a prevalence table over the whole pool so the threshold choice and slice size are
justified by data, not picked to flatter the claim.
"""
import json, os, sys, re, ast, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from co.embedder import Embedder

MTU = os.environ.get("MTU", "")
if not os.path.isfile(MTU):
    sys.exit("MTU is required: MTU=/path/to/mtu/M-S_eval.jsonl -- see ENVIRONMENT.md section 3")
KMAX = int(os.environ.get("MA_KMAX", "3"))
MIN_FIELDS = 6
MAX_BASES = int(os.environ.get("MA_MAX_BASES", "400"))
DANGER_T = float(os.environ.get("MA_DANGER_T", "0.52"))
CHAR_TARGET = 300
OUT = os.path.join(HERE, "danger_trials.jsonl")
SCAN = os.path.join(HERE, "danger_prevalence.json")
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
    groups, nbases, tid = [], 0, 0
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
        nbases += 1
        ordered_others = [k2 for k2, _ in items if k2 != gk]
        simmap = {k2: s for s, k2, _ in sims}
        valmap = {k2: v2 for k2, v2 in items}
        # each distractor carried as (key, value, sim_to_gold); length-2 prefix stays block()-compatible
        by_near = [(k2, valmap[k2], simmap[k2]) for k2 in sorted(simmap, key=lambda x: simmap[x])]
        for K in range(1, min(KMAX, len(sims)) + 1):
            length_only = [(k2, valmap[k2], simmap[k2]) for k2 in ordered_others[:K]]
            m_a = by_near[:K]
            lo_sims = [t[2] for t in length_only]
            ma_sims = [t[2] for t in m_a]
            groups.append({
                "base_id": r["id"], "gold_key": gk, "gold_val": gv, "K": K,
                "length_only": length_only, "m_a": m_a,
                "lo_max_sim": round(max(lo_sims), 4), "ma_max_sim": round(max(ma_sims), 4),
                "delta": round(max(lo_sims) - max(ma_sims), 4),
            })
    # prevalence table: how many groups reach each danger threshold
    thr = [0.45, 0.50, 0.52, 0.55, 0.58, 0.60]
    prev = {str(t): sum(1 for g in groups if g["lo_max_sim"] >= t) for t in thr}
    scan = {
        "schema": "light.ma.danger_scan.v1", "as_of": "2026-09-23",
        "n_bases": nbases, "n_groups": len(groups),
        "danger_threshold_used": DANGER_T,
        "prevalence_lo_max_sim_ge": prev,
        "note": "count of (base,K) groups where naive length_only truncation retains a distractor "
                "with max cosine to gold >= threshold; this is how often M-A has anything to prevent "
                "in natural field order. Higher threshold = rarer but more clearly dangerous.",
    }
    json.dump(scan, open(SCAN, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    danger = [g for g in groups if g["lo_max_sim"] >= DANGER_T and g["delta"] > 0.05]
    trials = []
    for g in danger:
        nat = len(", ".join(f"'{k}': {repr(v)}" for k, v, _ in
                            [(g["gold_key"], g["gold_val"], 0.0)] + g["length_only"]))
        exact = max(CHAR_TARGET, nat)
        for arm, dis in (("length_only", g["length_only"]), ("m_a", g["m_a"])):
            kv = [(k, v) for k, v, _ in dis]
            blk = block(g["gold_key"], g["gold_val"], kv, exact)
            dvals = [v for _, v, _ in dis]
            dsims = [s for _, _, s in dis]
            trials.append({
                "trial_id": f"D{tid}", "base_id": g["base_id"], "gold_key": g["gold_key"],
                "gold_val": g["gold_val"], "K": g["K"], "arm": arm,
                "shown_distractors": [k for k, _, _ in dis], "distractor_vals": dvals,
                "distractor_mean_sim": round(sum(dsims) / len(dsims), 4),
                "distractor_max_sim": round(max(dsims), 4),
                "n_fields_shown": 1 + g["K"], "char_len": len(prompt_of(blk, g["gold_key"])),
                "prompt": prompt_of(blk, g["gold_key"]),
            })
            tid += 1
    # matched-token assertion: two arms per (base,K) char-equal
    spread = max((max([t["char_len"] for t in trials if t["base_id"] == b and t["K"] == k]) -
                  min([t["char_len"] for t in trials if t["base_id"] == b and t["K"] == k]))
                 for (b, k) in {(t["base_id"], t["K"]) for t in trials})
    with open(OUT, "w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    emb.flush()
    print(f"scanned {len(groups)} (base,K) groups over {nbases} bases")
    print("prevalence lo_max_sim>=t:",
          {k: f"{v} ({v/len(groups)*100:.1f}%)" for k, v in prev.items()})
    print(f"DANGER_T={DANGER_T}: {len(danger)} danger groups -> {len(trials)} paired trials; "
          f"char spread={spread}")
    print(f"wrote {OUT} and {SCAN}")


def simmap_of(g, key):
    for k, v in g["length_only"] + g["m_a"]:
        if k == key:
            pass
    return None


if __name__ == "__main__":
    main()
