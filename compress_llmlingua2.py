#!/usr/bin/env python
"""External competitor arm: LLMLingua-2 token pruning on the SAME cost-leg sessions.

The rolling-summary argument alone is not enough; the claim needs a real
external content rule: "no token-level compressor arm (LLMLingua-class pruning) is in the
roster". This builds one. It is the bloat fix a practitioner reaches for BEFORE they think about
prefix stability: keep the naive sliding window, but prune its tokens down to the budget our
prefix-stable view already satisfies.

Method (LLMLingua-2, Pan et al., ACL 2024): a token-classification head scores every token of
the history region and we keep the top-k by keep-probability at a fixed rate. Model:
`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`, the ungated public release
(the `-it-quality-scoring` variant is gated and was not reachable from this network). Two
departures from the vendor package, both deliberate and stated in the paper:
  * scoring + top-k selection are implemented directly (no `llmlingua` pip dependency), and
  * pruning works at WHITESPACE-TOKEN granularity on the original bytes, so every compressed
    prompt is verifiably a subsequence of the arm it compresses (no rewriting, no re-tokenised
    surface). Subword keep-probabilities are averaged onto the whitespace token they came from.

Arms produced (all share the byte-identical probe tail of `prompt_length_only`):
  comp_open50 / comp_open25 / comp_open20 / comp_open12
                                            pure pruner at keep-rate .50/.25/.20/.12. The gold
                                            record line is NOT protected: whether it survives is
                                            a MEASURED property of the content rule, counted
                                            here and reported, never assumed.
  comp_keep25 / comp_keep20                 the same pruner at those rates with the gold line
                                            force-kept (benefit-of-the-doubt, so a fidelity gap
                                            cannot be waved away as "you deleted the answer").
                                            .20 is the budget-matched rate: it predicts the
                                            append-only view's measured prompt budget (see
                                            matched_budget.py for the estimator and its error, and
                                            the disclosure in run_prefix_comp.py for why it was
                                            added mid-leg).

Env: CC_MODEL, CC_RATES, CC_KEEP. Runs on CPU (177M params) when no GPU is usable. Deterministic:
no sampling anywhere, one scoring pass per trial serves all rates (each rate is a prefix of one
rank order, so adding a rate cannot change an existing arm's bytes -- asserted by
check_comp_controls.py), and reruns must be byte-identical.
"""
import json
import os
import re
import statistics as st
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRIALS = os.path.join(HERE, "prefix_cost_trials.jsonl")
OUT = os.path.join(HERE, "prefix_comp_trials.jsonl")
REPORT = os.path.join(HERE, "compression_report.json")

CC_MODEL = os.environ.get("CC_MODEL", "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
RATES = [float(x) for x in os.environ.get("CC_RATES", "0.50,0.25,0.20,0.12").split(",")]
# One benefit-of-the-doubt arm per rate, named by its own rate (reading a fixed element
# of RATES would silently rename the arm whenever a rate is added).
KEEP_RATES = [float(x) for x in os.environ.get("CC_KEEP", "0.25,0.20").split(",")]
PROBE_MARK = "\nUsing the results above, answer."
ARM = lambda rate: f"comp_open{int(round(rate * 100))}"
KEEP_ARM = lambda rate: f"comp_keep{int(round(rate * 100))}"
# The scorer is a 512-position BERT. Every history here is ~1.4k subwords, so it is scored in
# contiguous word-aligned chunks -- the same bounded-context device the vendor package uses.
MAX_SUB = int(os.environ.get("CC_MAX_SUB", "480"))


def split_hist(prompt):
    i = prompt.index(PROBE_MARK)
    return prompt[:i], prompt[i:]


def word_spans(hist):
    """Split on whitespace keeping the separators, so a keep-set reproduces original bytes."""
    spans, pos = [], 0
    for p in re.split(r"(\s+)", hist):
        if p:
            spans.append((p, pos, pos + len(p), p.isspace()))
            pos += len(p)
    return spans


def word_lengths(tok, words):
    """Subword count each whitespace word costs, so chunks can be packed under MAX_SUB."""
    lens = [0] * len(words)
    for a in range(0, len(words), 200):
        part = words[a:a + 200]
        for w in tok(part, add_special_tokens=False, is_split_into_words=True).word_ids(batch_index=0):
            if w is not None:
                lens[a + w] += 1
    return lens


def chunks(lens, limit):
    out, cur, used = [], [], 0
    for i, n in enumerate(lens):
        if cur and used + n > limit:
            out.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += n
    if cur:
        out.append(cur)
    return out


def keep_probs(model, tok, keep_id, dev, words, ck):
    import torch
    import torch.nn.functional as Fn
    probs = {}
    for idxs in ck:
        enc = tok([words[i] for i in idxs], add_special_tokens=True,
                  is_split_into_words=True, return_tensors="pt")
        wids = enc.word_ids(batch_index=0)
        with torch.no_grad():
            logits = model(**enc.to(dev)).logits[0].float()
            pk = Fn.softmax(logits, dim=-1)[:, keep_id].cpu().tolist()
        acc = {}
        for w, p in zip(wids, pk):
            if w is not None:
                acc.setdefault(w, []).append(p)
        for local, gi in enumerate(idxs):
            v = acc.get(local)
            if v:
                probs[gi] = sum(v) / len(v)
    return probs


def prune(spans, keep):
    """Kept words emit their original bytes, dropped words collapse; runs of spaces are tidied."""
    return re.sub(r"[ \t]{2,}", " ", "".join(s[0] if (s[3] or w in keep) else ""
                                             for w, s in enumerate(spans)))


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    torch.set_grad_enabled(False)

    rows = [json.loads(l) for l in open(TRIALS, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(CC_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(CC_MODEL).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    keep_id = int(model.config.label2id.get("LABEL_1", 1))
    print(f"[compress] scorer={CC_MODEL} dev={dev} params={sum(p.numel() for p in model.parameters())/1e6:.0f}M "
          f"rows={len(rows)} rates={RATES}")

    arms = [ARM(r) for r in RATES] + [KEEP_ARM(r) for r in KEEP_RATES]
    out, t0 = [], time.time()
    for n, r in enumerate(rows):
        hist, tail = split_hist(r["prompt_length_only"])
        spans = word_spans(hist)
        widx = [w for w, s in enumerate(spans) if not s[3]]
        words = [spans[w][0] for w in widx]
        probs = keep_probs(model, tok, keep_id, dev, words, chunks(word_lengths(tok, words), MAX_SUB))
        probs = {w: probs.get(k, 0.5) for k, w in enumerate(widx)}
        gold_w = {w for w, s in enumerate(spans) if s[0] == f"{r['key']}={r['gold']}"}
        # rank once; each rate is a prefix of the same deterministic order
        order = sorted(widx, key=lambda w: (-probs.get(w, 0.5), w))

        rec = {"session_id": r["session_id"], "turn": r["turn"], "window": r["window"],
               "key": r["key"], "gold": r["gold"], "n_words": len(words),
               "stable_chars": len(r["prompt_prefix_stable"]),
               "stable_words": len(r["prompt_prefix_stable"].split()),
               "raw_chars": len(hist), "probe_tail": tail}
        for rate in RATES:
            a = ARM(rate)
            body = prune(spans, set(order[:max(1, int(round(rate * len(widx))))]))
            rec[f"prompt_{a}"] = body + tail
            rec[f"words_{a}"] = len(body.split())
            rec[f"survives_{a}"] = int(f"{r['key']}={r['gold']}" in body)
        for rate in KEEP_RATES:
            a = KEEP_ARM(rate)
            base = set(order[:max(1, int(round(rate * len(widx))))]) | gold_w
            body = prune(spans, base)
            rec[f"prompt_{a}"] = body + tail
            rec[f"words_{a}"] = len(body.split())
            rec[f"survives_{a}"] = int(f"{r['key']}={r['gold']}" in body)
        out.append(rec)
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(rows)} ({time.time()-t0:.0f}s)")

    with open(OUT, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rep = {"scorer": CC_MODEL, "dev": dev, "n_trials": len(out), "rates": RATES,
           "elapsed_s": round(time.time() - t0, 1),
           "torch": torch.__version__, "transformers": __import__("transformers").__version__,
           "median_stable_words": st.median([x["stable_words"] for x in out]),
           "arms": {}}
    for a in arms:
        rep["arms"][a] = {
            "gold_survives": sum(x[f"survives_{a}"] for x in out),
            "gold_survival_rate": round(sum(x[f"survives_{a}"] for x in out) / len(out), 4),
            "median_words": st.median([x[f"words_{a}"] for x in out]),
            "median_words_raw": st.median([x["n_words"] for x in out]),
            "median_words_stable_arm": rep["median_stable_words"],
        }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n-> {OUT}  ({len(out)} trials x {len(arms)} arms)")
    for a in arms:
        v = rep["arms"][a]
        print(f"   {a:12s} gold survives {v['gold_survives']:4d}/{len(out)} "
              f"({v['gold_survival_rate']:.3f})  median words {v['median_words']:.0f} "
              f"(raw {v['median_words_raw']:.0f})")


if __name__ == "__main__":
    main()
