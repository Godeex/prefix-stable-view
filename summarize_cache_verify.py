#!/usr/bin/env python
"""Reprint the [A]/[B] summaries of verify_cache_measure.py from the saved event file.

verify_cache_measure.py talks to the server; this script only reads cache_verify_events.json, so the
numbers that go into the paper can be re-checked without re-running any inference.
"""
import json, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EV = json.load(open(os.path.join(HERE, "cache_verify_events.json")))

d = [e["server_prompt"] - e["tok_proxy"] for e in EV]
off = sorted(set(d))
exact = sum(1 for x in d if abs(x) <= 4)
print(f"[A] proxy vs server prompt_eval_count over {len(EV)} prompts:")
print(f"    diff mean {sum(d)/len(d):+.1f} min {min(d)} max {max(d)}; "
      f"within 4 tokens {exact}/{len(EV)} = {exact/len(EV):.1%}")
print(f"    distinct offsets observed: {off}")

first = {}
for e in EV:
    k = (e["session"], e["cell"])
    first[k] = min(first.get(k, 10**9), e["turn"])
print("[B] token-level prefix reuse (turns with a measured predecessor):")
by = defaultdict(list)
for e in EV:
    if e["turn"] > first[(e["session"], e["cell"])]:
        by[e["cell"]].append(e)
for cell in sorted(by):
    sub = by[cell]
    und = [e for e in sub if e["cached"] < 0.9 * e["token_lcp_vs_prev"]]
    ovr = [e for e in sub if e["cached"] > e["token_lcp_vs_prev"] + 4]
    rat = [e["cached"] / max(e["token_lcp_vs_prev"], 1) for e in sub]
    print(f"    {cell}: n={len(sub)} mean token-LCP "
          f"{sum(e['token_lcp_vs_prev'] for e in sub)/len(sub):6.1f} mean cached "
          f"{sum(e['cached'] for e in sub)/len(sub):6.1f} ratio(cached/LCP) "
          f"{sum(rat)/len(rat):.3f} | under-reuse {len(und)} over-reuse {len(ovr)}")
print(f"\n    sessions {len(set(e['session'] for e in EV))}, "
      f"turns per stream {min(len(set(e['turn'] for e in EV if e['cell']==c)) for c in by)}")
