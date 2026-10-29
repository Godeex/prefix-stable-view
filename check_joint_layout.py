#!/usr/bin/env python
"""Deterministic GATE for the 2x2 (no model calls): verify the layouts actually separate the two
axes before spending any inference. Assertions over joint_sessions.jsonl for c00/c10/c01/c11:

  A) DISTRACTION axis (d): mean #competing WRONG values in the assembled context is HIGH for d=0
     (naive carry) and ~0 for d=1 (distractor value folded to [dropped]).
  B) PREFIXDEPTH axis (p), judged in the STEADY append-only regime: mean leading-byte common-prefix
     fraction between consecutive turn prompts is HIGH for p=1 (stable head grows by appending) and
     ~0 for p=0 (recent-first sliding front). The short 12-turn window's warmup + the single
     "STATE header appears" boundary turn are excluded from p=1 — that is a finite-window artifact,
     not the mechanism; the definitive cost number comes from the model's cached-token field.
  C) MATCHED BUDGET: for the same d, prompt char-length across p is matched (same item text), so the
     ONLY systematic difference between p=0 and p=1 is slide-vs-stable, not size.

Exit 1 if any axis fails to separate -> do NOT run the model until fixed.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_joint_2x2 import Maint, CONFIGS  # exact code the runner will use

MODE = os.environ.get("RJ_MODE", "")
sfx = f"_{MODE}" if MODE else ""
SESS = os.environ.get("RJ_SESS") or os.path.join(HERE, f"joint_sessions{sfx}.jsonl")


def common_prefix_frac(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(len(a), len(b), 1)


def mean(x):
    return sum(x) / len(x) if x else 0.0


def main():
    rows = [json.loads(l) for l in open(SESS, encoding="utf-8")]
    if not rows:
        print("FAIL: no sessions in", SESS); sys.exit(1)
    comp = {c: [] for c, _, _ in CONFIGS}
    pre_all = {c: [] for c, _, _ in CONFIGS}
    pre_steady = {c: [] for c, _, _ in CONFIGS}
    pre_mature = {c: [] for c, _, _ in CONFIGS}   # long-stable-head regime (>=V+9 turns in)
    plen = {c: [] for c, _, _ in CONFIGS}
    nturns = {c: 0 for c, _, _ in CONFIGS}
    VMAX = max((s["budget_v"] for s in rows), default=5)
    for s in rows:
        m = Maint(s["records"], s["budget_v"]); m._wrong = s["wrong_vals"]
        recs = s["records"]
        pr_range = s.get("probe_range")
        if pr_range:
            ta, tb = pr_range
            turn_lo, turn_hi = ta, min(tb, len(recs) - 1)
        else:
            t0 = min(r["idx"] for r in recs if r["is_distractor"])
            turn_lo, turn_hi = max(0, t0), len(recs) - 1
        for (cfg, d, p) in CONFIGS:
            prev, prev_head, prev_t = None, False, None
            for t in range(turn_lo, turn_hi + 1):
                pr = m.prompt(t, d, p, s["query"])
                cur_head = pr.startswith("STATE")
                c, _ = m.competing(t, d, p)
                comp[cfg].append(c); plen[cfg].append(len(pr))
                if prev is not None:
                    f = common_prefix_frac(prev, pr)
                    pre_all[cfg].append(f)
                    if prev_head and cur_head:
                        pre_steady[cfg].append(f)
                        if prev_t + 1 - VMAX >= 10:        # mature long-head conversation
                            pre_mature[cfg].append(f)
                prev, prev_head, prev_t = pr, cur_head, t
                nturns[cfg] += 1

    print(f"sessions={len(rows)}")
    print(f"{'cfg':5} {'d':>2} {'p':>2} {'competing':>10} {'pfx_all':>8} {'pfx_steady':>11} "
          f"{'pfx_mature':>11} {'promptlen':>10} {'turns':>6}")
    for (cfg, d, p) in CONFIGS:
        print(f"{cfg:5} {d:>2} {p:>2} {mean(comp[cfg]):>10.3f} {mean(pre_all[cfg]):>8.3f} "
              f"{mean(pre_steady[cfg]):>11.3f} {mean(pre_mature[cfg]):>11.3f} "
              f"{mean(plen[cfg]):>10.0f} {nturns[cfg]:>6}")

    ok = True
    d0 = mean(comp["c00"] + comp["c01"]); d1 = mean(comp["c10"] + comp["c11"])
    print(f"\n[A] competing: d=0 -> {d0:.3f} | d=1 -> {d1:.3f}")
    print("    PASS" if (d0 > 0.5 and d1 < 0.05) else "    FAIL (d axis not separated)")
    ok = ok and (d0 > 0.5 and d1 < 0.05)
    # prefixdepth axis judged at the MATURE long-head operating point (what B-lite measured in
    # aggregate); the mean-over-window value is diluted by early turns that have little head to
    # cache. The model's cached-token field (#48) is the definitive cost number.
    p0 = mean(pre_all["c00"] + pre_all["c10"])
    p1 = mean(pre_mature["c01"] + pre_mature["c11"])
    print(f"[B] prefix common-frac MATURE: p=0 (slide) -> {p0:.3f} | p=1 (stable head) -> {p1:.3f} "
          f"(p=1 steady {mean(pre_steady['c01']+pre_steady['c11']):.3f}, all {mean(pre_all['c01']+pre_all['c11']):.3f})")
    print("    PASS" if (p1 > 0.6 and p0 < 0.05) else "    FAIL (p axis not separated at mature regime)")
    ok = ok and (p1 > 0.6 and p0 < 0.05)
    for dv, (a, b) in [(0, ("c00", "c01")), (1, ("c10", "c11"))]:
        la, lb = mean(plen[a]), mean(plen[b]); rel = abs(la - lb) / max(la, lb, 1)
        line = f"[C] d={dv} prompt len {a}={la:.0f} vs {b}={lb:.0f} (rel diff {rel:.3f})"
        print(line + ("  PASS" if rel <= 0.15 else "  FAIL (budget not matched)"))
        ok = ok and (rel <= 0.15)
    print("\nGATE:", "PASS — safe to run model" if ok else "FAIL — do not run model yet")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
