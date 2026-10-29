#!/usr/bin/env python
"""B-lite probe 0: does ollama expose a usable prefix-cache signal, and does a
byte-stable prefix get cached while a mid-shifting prefix does not?

Pure mechanism validation before we build the real two-arm cost experiment.
Keeps prompts long enough that cache hits are meaningful; temp=0; num_predict=1 so
we measure PREFILL only (recompute is a prefill-side cost). Sequential single requests.
"""
import json, os, time, urllib.request

CHAT = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("PB_MODEL", "qwen3:8b")


def call(prompt):
    body = json.dumps({"model": MODEL, "stream": False, "think": False,
                       "options": {"temperature": 0.0, "num_predict": 1},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(CHAT, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=200) as r:
        d = json.loads(r.read())
    wall = time.perf_counter() - t0
    keys = {k: d.get(k) for k in ("prompt_eval_count", "prompt_eval_cached_count",
                                  "prompt_eval_duration", "eval_duration", "total_duration")}
    return keys, wall


def facts(n):
    # deterministic filler lines -> stable multi-hundred-token prefix
    return "\n".join(f"FACT {i}: the sensor reading for zone {i} nominal value is {i*7%97} units."
                     for i in range(n))


def main():
    base = facts(120)  # long stable prefix
    print(f"model={MODEL} prefix_chars={len(base)}")

    print("\n== warm call (cold cache) ==")
    k, w = call(base + "\nQUESTION: what is FACT 3?\nANSWER:")
    print("cold ", k, f"wall={w:.2f}s")

    print("\n== same stable prefix, DIFFERENT suffix (prefix should be cached) ==")
    for q in (5, 40, 90, 110):
        k, w = call(base + f"\nQUESTION: what is FACT {q}?\nANSWER:")
        print(f"q={q:<4}", k, f"wall={w:.2f}s")

    print("\n== control: SHIFTING prefix (drop a leading line each call) ==")
    for shift in range(1, 5):
        shifted = "\n".join(base.split("\n")[shift:])  # leading tokens change each call
        k, w = call(shifted + "\nQUESTION: last?\nANSWER:")
        print(f"shift={shift}", k, f"wall={w:.2f}s")

    print("\n== same exact prompt twice (max cache reuse) ==")
    p = base + "\nQUESTION: what is FACT 7?\nANSWER:"
    call(p)
    k, w = call(p)
    print("again", k, f"wall={w:.2f}s")


if __name__ == "__main__":
    main()
