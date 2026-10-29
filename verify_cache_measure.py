#!/usr/bin/env python
"""Is the cached-token field actually measuring a TOKEN prefix?

The cost metric is recompute = prompt_eval_count - prompt_eval_cached_count. One may object
that (i) llama.cpp keys its radix tree on token ids, so a shared STRING prefix does not guarantee a
shared TOKEN prefix (BPE merges across the join), and (ii) the server's own cache can evict, so a
low hit rate might be eviction rather than layout. This script answers both with the server's own
numbers plus a calibrated proxy tokenizer. No new claims are made beyond what is measured.

A  proxy fidelity    : tokenize every prompt with a local Qwen BPE and compare the count to the
                      server's prompt_eval_count. What matters is whether the gap is a CONSTANT
                      (chat-template wrapper) or noise; only then is the proxy usable for B.
B  token-prefix reuse : for consecutive turns, compare the longest common TOKEN prefix length of
                      prompt(t), prompt(t+1) to the cached count reported at t+1. Under-reuse
                      (cached << token-LCP) would mean BPE boundaries or eviction are breaking the
                      claimed mechanism; over-reuse (cached > token-LCP) would mean stale nodes from
                      an earlier run are leaking into the measurement (the nonce should make this 0).
C  smoke calibration  : build two prompts sharing an exact k-token prefix, warm then re-ask with a
                      different suffix, and report cached vs k.
D  eviction probe     : re-ask a warmed prefix after N unrelated sessions have run in between, and
                      report whether the hit survives at the real load scale.
E  server facts       : versions, GPU, context length, what the harness does and does not pin.

PB_PARTS selects sections (default ABCDE). Sections A/B write cache_verify_events.json (per-prompt
rows; summarize_cache_verify.py recomputes their summary from that file without any model call);
every section is persisted to cache_verify_report.json.
"""
import json, os, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_joint_layout import Layout, CELLS

CHAT = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("PB_MODEL", "qwen3:14b")
NUMCTX = int(os.environ.get("PB_NUMCTX", "8192"))
SESS = os.path.join(HERE, "joint_sessions.jsonl")
NONCE = "verify" + str(int(time.time()))
PROXY = os.environ.get("PB_PROXY", "Qwen/Qwen2.5-VL-7B-Instruct")
# Which sections to run: A proxy fidelity, B token-prefix reuse, C smoke calibration,
# D eviction probe, E server facts. PB_PARTS=CE runs the cheap persistent ones only.
PARTS = set((os.environ.get("PB_PARTS") or "ABCDE").strip().upper())


def call(prompt, npred=1):
    body = json.dumps({"model": MODEL, "stream": False, "think": False,
                       "options": {"temperature": 0.0, "seed": 1234, "num_predict": npred,
                                   "num_ctx": NUMCTX},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(CHAT, data=body, headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return (int(d.get("prompt_eval_count") or 0), int(d.get("prompt_eval_cached_count") or 0),
            round(int(d.get("prompt_eval_duration") or 0) / 1e6, 1))


def load_report():
    p = os.path.join(HERE, "cache_verify_report.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def main():
    rep = load_report()
    rep["meta"] = {"model": MODEL, "num_ctx": NUMCTX, "proxy_tokenizer": PROXY,
                   "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "parts": sorted(PARTS)}

    need_tok = PARTS & {"A", "B", "C"}
    tok = None
    if need_tok:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(PROXY, use_fast=True)
    print(f"proxy tokenizer={PROXY}  server model={MODEL}  parts={''.join(sorted(PARTS))}")

    ev = []
    if {"A", "B"} & PARTS:
        rows = [json.loads(l) for l in open(SESS, encoding="utf-8")][:8]
        for s in rows:
            m = Layout(s["records"], s["budget_v"])
            m._wrong = s["wrong_vals"]
            t0 = min(r["idx"] for r in s["records"] if r["is_distractor"])
            turns = list(range(max(0, t0), len(s["records"])))
            for cell in ("s0l0", "s1l1"):
                prev_ids = None
                for t in turns:
                    p = f"#run={NONCE}_{cell}_{s['session_id']}\n" + m.prompt(t, cell, s["query"])
                    ids = tok(p, add_special_tokens=False)["input_ids"]
                    cnt, cac, ms = call(p)
                    lcp = 0
                    if prev_ids is not None:
                        for a, b in zip(prev_ids, ids):
                            if a != b:
                                break
                            lcp += 1
                    ev.append({"session": s["session_id"], "cell": cell, "turn": t,
                               "tok_proxy": len(ids), "server_prompt": cnt,
                               "token_lcp_vs_prev": lcp, "cached": cac})
                    prev_ids = ids

        # A) proxy fidelity
        d = [e["server_prompt"] - e["tok_proxy"] for e in ev]
        off = sorted(set(d))
        print(f"\n[A] proxy vs server prompt_eval_count over {len(ev)} prompts: "
              f"diff mean {sum(d)/len(d):+.1f} min {min(d)} max {max(d)}; "
              f"distinct offsets {off}")
        rep["A_proxy_fidelity"] = {"n": len(ev), "mean_diff": sum(d) / len(d),
                                   "min_diff": min(d), "max_diff": max(d),
                                   "offsets": off,
                                   "constant_offset": len(off) == 1}

        # B) token-prefix reuse, per cell, excluding the first turn of each stream
        print("\n[B] token-level prefix reuse (turns with a measured predecessor only)")
        first = {}
        for e in ev:
            key = (e["session"], e["cell"])
            first[key] = min(first.get(key, 10**9), e["turn"])
        B = {}
        for cell in ("s0l0", "s1l1"):
            sub = [e for e in ev if e["cell"] == cell and e["turn"] > first[(e["session"], cell)]]
            und = [e for e in sub if e["cached"] < 0.9 * e["token_lcp_vs_prev"]]
            ovr = [e for e in sub if e["cached"] > e["token_lcp_vs_prev"] + 4]
            rat = [e["cached"] / max(e["token_lcp_vs_prev"], 1) for e in sub]
            lcp = sum(e["token_lcp_vs_prev"] for e in sub) / len(sub)
            cac = sum(e["cached"] for e in sub) / len(sub)
            print(f"    {cell}: n={len(sub)} mean token-LCP {lcp:6.1f} mean cached {cac:6.1f} "
                  f"ratio(cached/LCP) mean {sum(rat)/len(rat):.3f} "
                  f"| under-reuse {len(und)}  over-reuse(stale) {len(ovr)}")
            B[cell] = {"n": len(sub), "mean_token_lcp": lcp, "mean_cached": cac,
                       "mean_ratio": sum(rat) / len(rat), "under_reuse": len(und),
                       "over_reuse": len(ovr)}
        rep["B_token_prefix_reuse"] = B

    # C) smoke calibration on a constructed shared prefix
    base = "\n".join(f"FACT {i}: the sensor reading for zone {i} nominal value is {i*7%97} units."
                     for i in range(80))
    if "C" in PARTS:
        print("\n[C] smoke: prompt pair with an exactly k-token shared prefix")
        C = []
        for extra in (0, 1, 5, 20):
            b2 = base + ("\n" + " ".join(["padding word"] * (extra * 4)) if extra else "")
            pre = f"#run={NONCE}_smoke_{extra}\n{b2}"
            k = len(tok(pre, add_special_tokens=False)["input_ids"])
            warm = call(pre + "\nQUESTION: what is FACT 3?\nANSWER:")
            c2 = call(pre + "\nQUESTION: what is FACT 44?\nANSWER:")
            print(f"    k={k:4d} shared tokens -> second call cached={c2[1]:4d} "
                  f"(ratio {c2[1]/k:.3f}) recompute={c2[0]-c2[1]:3d} prompt={c2[0]} "
                  f"warm prompt={warm[0]}")
            C.append({"k_proxy": k, "cached": c2[1], "prompt": c2[0], "recompute": c2[0] - c2[1],
                      "warm_prompt": warm[0], "ratio": c2[1] / k})
        rep["C_smoke_calibration"] = C

    # D) eviction probe: warm one prefix, run unrelated load, re-ask
    if "D" in PARTS:
        print("\n[D] eviction: same nonce-prefixed prompt re-asked after unrelated interleaved load")
        keep = f"#run={NONCE}_ev\n{base}\nQUESTION: what is FACT 9?\nANSWER:"
        warm = call(keep)
        D = [{"gap": None, "cached": warm[1], "prompt": warm[0]}]
        for gap in (0, 4, 16, 40):
            for i in range(gap):
                filler = (f"#run={NONCE}_fill{i}\n" +
                          "\n".join(f"NOTE {j}: value {j*13%91}" for j in range(30)) +
                          "\nQUESTION: notes?\nANSWER:")
                call(filler)
            c = call(keep)
            print(f"    after {gap:2d} interleaved requests: cached={c[1]:4d}/{c[0]} "
                  f"recompute={c[0]-c[1]}")
            D.append({"gap": gap, "cached": c[1], "prompt": c[0], "recompute": c[0] - c[1]})
        rep["D_eviction_probe"] = D

    # E) server facts actually observed
    print("\n[E] server facts")
    E = {}
    E["ollama_version"] = subprocess.run(["ollama", "-v"], capture_output=True,
                                              text=True).stdout.strip()
    print("    ollama:", E["ollama_version"])
    gl = subprocess.run(["strings", "/usr/bin/ollama"], capture_output=True,
                        text=True).stdout.count("ggml")
    print(f"    llama.cpp/ggml build string: {gl} occurrences in the ollama binary; "
          "exact llama.cpp version not exposed by the CLI -> unknown")
    E["ggml_string_count"] = gl
    E["llama_cpp_version"] = "unknown"
    sh = subprocess.run(["ollama", "show", MODEL], capture_output=True, text=True).stdout
    E["ollama_show"] = [ln.strip() for ln in sh.split("\n")
                        if any(w in ln for w in ("quantization", "context length", "parameters"))]
    for ln in E["ollama_show"]:
        print("   ", ln)
    E["gpu"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print("    GPU:", E["gpu"].replace("\n", " | "))
    E["pinned"] = (f"num_ctx={NUMCTX}, seed=1234, temperature=0, num_predict<=40; "
                   "keep_alive and cache TTL NOT set by the client (server defaults apply)")
    print(f"    harness pins {E['pinned']}")
    rep["E_server_facts"] = E

    if ev:
        json.dump(ev, open(os.path.join(HERE, "cache_verify_events.json"), "w"), indent=1)
        print(f"\n-> per-prompt events: cache_verify_events.json ({len(ev)} rows)")
    json.dump(rep, open(os.path.join(HERE, "cache_verify_report.json"), "w"), indent=1)
    print("-> all sections: cache_verify_report.json")


if __name__ == "__main__":
    main()
