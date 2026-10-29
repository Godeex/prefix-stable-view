#!/usr/bin/env python
"""B1: passage-level DANGER-slice two-arm generator on real multi-doc QA (LongBench).

Rationale (after the (A) field-retrieval negative): single-field retrieval never exposes the
specific confusable near-duplicates, so M-A vs naive truncation is untestable there. On multi-hop
passage QA the naive policy (keep first-K passages by POSITION) can retain a semantically-adjacent
passage that carries a COMPETING answer entity -> substitution. This is where the distraction risk
has teeth (EX-2b already showed near 26% vs far 10%). Same matched-token, same gold-fidelity-floor
discipline as (A), but the unit is a passage and the outcome is substitution.

Per usable question (>=6 passages, unique gold passage):
  length_only = gold + first K distractor passages by original context position (naive truncation)
  m_a         = gold + K LOWEST-sim-to-gold distractor passages (drop the semantically nearest)
Both blocks padded to EQUAL char length (matched budget). gold always present (fidelity floor).
DANGER group = length_only retained a passage with sim >= LB_DANGER_T (the naive truncation kept a
near-adversary that a proximity-aware controller would evict).

Cheap embedding-only prevalence scan over the whole pool sizes the slice and justifies the
threshold before any LLM call. Pools hotpotqa + 2wikimqa + musique to reach ~300 usable questions.
"""
import json, os, re, sys, math, urllib.request, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", "")
if not os.path.isdir(DATA):
    sys.exit("DATA is required: DATA=/path/to/longbench/data (the hotpotqa/2wikimqa/musique .jsonl dir) -- see ENVIRONMENT.md section 3")
FILES = ["hotpotqa.jsonl", "2wikimqa.jsonl", "musique.jsonl"]
EMB = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "bge-m3:latest"
K = int(os.environ.get("LB_K", "4"))
TRUNC = 500
MIN_PASSAGES = 6
MAX_Q = int(os.environ.get("LB_MAX_Q", "300"))
DANGER_T = float(os.environ.get("LB_DANGER_T", "0.60"))
PAIRED = os.path.join(HERE, "trials_lb_paired.jsonl")
DANGER = os.path.join(HERE, "trials_lb_danger.jsonl")
SCAN = os.path.join(HERE, "lb_danger_prevalence.json")
CACHE = os.path.join(HERE, "emb_cache.json")
_c = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def embed(t):
    t = str(t)
    if t in _c:
        return _c[t]
    req = urllib.request.Request(EMB, data=json.dumps({"model": EMBED_MODEL, "input": t[:800]}).encode(),
                                 headers={"Content-Type": "application/json"})
    v = json.loads(urllib.request.urlopen(req, timeout=90).read())["embeddings"][0]
    _c[t] = v
    return v


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return d / (na * nb + 1e-9)


def split(ctx):
    return [p.strip() for p in re.split(r"(?m)^Passage \d+:\s*", ctx) if p.strip()]


def body(dsel):
    return "\n".join(f"[doc] {p[:TRUNC]}" for p, _ in dsel)


def make_prompt(gold, dsel, question, target):
    b = body(dsel)
    while len(b) < target:
        b += "\n[ref] ----"
    b = b[:target]
    return ("Answer the question using ONLY the context passages. "
            "Give the shortest answer span, nothing else.\n\n"
            f"Context:\n[doc] {gold[:TRUNC]}\n{b}\n\nQuestion: {question}")


def main():
    groups = []
    for fn in FILES:
        for line in open(os.path.join(DATA, fn)):
            r = json.loads(line)
            ps = split(r["context"])
            ans_list = r.get("answers") or []
            if len(ps) < MIN_PASSAGES or not ans_list:
                continue
            ans = str(ans_list[0]).strip()
            if not ans:
                continue
            hit = [i for i, p in enumerate(ps) if ans.lower() in p.lower()]
            if len(hit) != 1:
                continue
            gi = hit[0]; gold = ps[gi]
            # distractors in ORIGINAL position order (what naive positional truncation sees)
            dis_pos = [(p, i) for i, p in enumerate(ps) if i != gi]
            gv = embed(gold[:TRUNC])
            sims = [cos(gv, embed(p[:TRUNC])) for p, _ in dis_pos]
            # need at least one clearly-near and one clearly-far distractor
            if max(sims) - min(sims) < 0.02:
                continue
            tagged = [(p, s) for (p, _i), s in zip(dis_pos, sims)]   # (passage, sim_to_gold), position order
            order = tagged                                           # dis_pos already ascending position
            by_near = sorted(tagged, key=lambda x: x[1])             # ascending sim
            lo = order[:K]; ma = by_near[:K]
            lo_max = max(s for _, s in lo)
            ma_max = max(s for _, s in ma)
            groups.append({"base_id": r["_id"], "dataset": fn.split('.')[0], "gold": ans,
                           "gold_passage": gold, "question": r["input"],
                           "lo": lo, "ma": ma, "lo_max": round(lo_max, 4), "ma_max": round(ma_max, 4)})
            if len(groups) >= MAX_Q:
                break
        if len(groups) >= MAX_Q:
            break

    thr = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    prev = {str(t): sum(1 for g in groups if g["lo_max"] >= t) for t in thr}
    json.dump({"schema": "light.ma.lb_danger_scan.v1", "as_of": "2026-09-23",
               "n_questions": len(groups), "K": K, "trunc": TRUNC,
               "danger_threshold_used": DANGER_T, "prevalence_lo_max_ge": prev,
               "note": "count of questions where naive positional truncation retains a distractor "
                       "passage with cosine to gold >= threshold (the regime where M-A has an "
                       "adjacent-but-wrong passage to evict)."},
              open(SCAN, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def emit(gs, path):
        rows = []
        for g in gs:
            nat = len(body(g["lo"]))
            target = max(TRUNC, nat)
            for arm, dsel in (("length_only", g["lo"]), ("m_a", g["ma"])):
                dvals = [p[:TRUNC] for p, _ in dsel]
                dsims = [s for _, s in dsel]
                pr = make_prompt(g["gold_passage"], dsel, g["question"], target)
                rows.append({"trial_id": f"LB{len(rows)}", "base_id": g["base_id"], "dataset": g["dataset"],
                             "gold": g["gold"], "gold_passage": g["gold_passage"][:TRUNC], "K": K,
                             "arm": arm, "distractor_vals": dvals,
                             "distractor_mean_sim": round(sum(dsims) / len(dsims), 4),
                             "distractor_max_sim": round(max(dsims), 4),
                             "n_fields_shown": 1 + K, "char_len": len(pr), "prompt": pr})
        with open(path, "w", encoding="utf-8") as f:
            for t in rows:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        return rows

    json.dump(_c, open(CACHE, "w"))
    full = emit(groups, PAIRED)
    danger = [g for g in groups if g["lo_max"] >= DANGER_T and g["lo_max"] - g["ma_max"] > 0.03]
    drows = emit(danger, DANGER)
    # matched-token assertion within each file
    def spread(rows):
        by = collections.defaultdict(list)
        for t in rows:
            by[t["base_id"]].append(t["char_len"])
        return max((max(v) - min(v) for v in by.values()), default=0)
    print(f"usable questions={len(groups)} across {[g['dataset'] for g in groups][:3]}...")
    print("prevalence lo_max>=t:", {k: f"{v} ({v/len(groups)*100:.1f}%)" for k, v in prev.items()})
    print(f"DANGER_T={DANGER_T}: {len(danger)} danger questions -> {len(drows)} paired trials; "
          f"danger char spread={spread(drows)} (full spread={spread(full)})")
    print(f"wrote {PAIRED} ({len(full)} rows) and {DANGER} ({len(drows)} rows) and {SCAN}")


if __name__ == "__main__":
    main()
