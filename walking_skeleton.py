#!/usr/bin/env python
"""M-A walking skeleton: end-to-end run on a synthetic >=5-turn matched-token session.

Runs the full thin path: HistoryLog.append -> SummaryView.fold (projection pushdown)
-> DistractionCost -> AdmissionEvictor.decide -> ContextAssembler.assemble -> ollama chat
-> RoundMetric.judge -> EvalSink, then writes a raw run bundle (manifest + metrics + stdout).

This is a WALKING SKELETON: proves the seams connect and the three invariants hold on
a live model. It is NOT the main comparison matrix (no length-only arm, no A/B, no
dose-response). No research design is changed here.
"""
import hashlib
import json
import os
import sys
import time
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from co.history_log import HistoryLog, TurnRecord                    # noqa: E402
from co.summary_view import SummaryView, is_byte_prefix              # noqa: E402
from co.embedder import Embedder                                     # noqa: E402
from co.distraction_cost import DistractionCost, Candidate          # noqa: E402
from co.admission import AdmissionEvictor, VERBATIM, EVICT          # noqa: E402
from co.assembler import ContextAssembler                           # noqa: E402
from co.eval_sink import RoundMetric                                # noqa: E402

MODEL = os.environ.get("MA_MODEL", "qwen3:8b")
BUDGET_T = int(os.environ.get("MA_BUDGET_T", "40"))
RUNS_DIR = os.path.join(HERE, "runs")
EMB_CACHE = os.path.join(HERE, "emb_cache.json")


def _sha(path):
    return "sha256:" + hashlib.sha256(open(path, "rb").read()).hexdigest()


def make_session():
    """5 turns; each tool result = 1 gold field + near distractor(s) that are semantically
    adjacent to the gold, matched so a naive 'keep everything' context would risk contamination."""
    turns = []
    for i in range(5):
        gold = {"city": ["Lima", "Hanoi", "Quito", "Rabat", "Sucre"][i]}
        near = [("country", ["China", "Peru", "Morocco", "Ecuador", "Bolivia"][i])]
        fields = list(gold.items()) + near
        turns.append({"turn_id": i, "ask_key": "city",
                      "gold_key": "city", "gold_val": gold["city"],
                      "fields": fields})
    return turns


def run():
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir = os.path.join(RUNS_DIR, f"walk-{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_lines = []

    def echo(msg):
        print(msg)
        log_lines.append(msg)

    emb = Embedder(cache_path=EMB_CACHE)
    params = json.load(open(os.path.join(HERE, "configs", "calibration_params.json")))
    assert params.get("fit_scope") == "pilot" and params.get("provisional"), \
        "walking skeleton must run on PROVISIONAL pilot calibration, not a claimed fit"
    cost = DistractionCost(embedder=emb, params=params["params"])
    asm = ContextAssembler(model=MODEL)

    hist = HistoryLog()
    view = SummaryView(embedder=emb)
    ev = AdmissionEvictor(embedder=emb, cost=cost, budget_T=BUDGET_T)
    metrics = []
    prev_prompt = ""
    probe_prefix_ok = True
    probe_fidelity_ok = True

    for t in make_session():
        hist.append(TurnRecord(turn_id=t["turn_id"], role="tool_result",
                               fields=dict(t["fields"]), raw_text=json.dumps(dict(t["fields"])),
                               tokens=sum(len(str(v).split()) for _, v in t["fields"])))
        # projection pushdown: only the gold-bearing binding is materialized into the cacheable prefix
        view.fold({t["gold_key"]: t["gold_val"]})

        cands = [Candidate(unit_id=f"{t['turn_id']}:{k}", text=f"{k}={v}",
                           is_gold_bearing=(k == t["gold_key"]), turn_id=t["turn_id"],
                           tokens=max(1, len(str(v).split())))
                 for k, v in t["fields"]]
        decision = ev.decide(cands, gold_values=[t["gold_val"]])
        # fidelity floor probe: gold must never be EVICT
        gold_state = decision[f"{t['turn_id']}:{t['gold_key']}"]
        if gold_state == EVICT:
            probe_fidelity_ok = False
        verbatim = [c for c in cands if decision[c.unit_id] == VERBATIM]

        prompt = asm.assemble(view, verbatim,
                              question=f"What is the value of '{t['ask_key']}'?")
        if prev_prompt and not is_byte_prefix(
                view_prefix_of(prev_prompt), view_prefix_of(prompt)):
            probe_prefix_ok = False
        prev_prompt = prompt

        d0 = time.perf_counter()
        answer, ptec = asm.chat(prompt)
        decision_ms = (time.perf_counter() - d0) * 1000
        rm = RoundMetric(turn=t["turn_id"], prompt_tokens=ptec, answer=answer,
                         gold=t["gold_val"],
                         distractor_vals=[v for k, v in t["fields"] if k != t["gold_key"]],
                         degraded_embed=False).judge()
        metrics.append(rm.to_dict())
        echo(f"turn {t['turn_id']}: decision={decision} prompt_tokens={ptec} "
             f"answer={answer!r} gold_present={rm.gold_present} "
             f"contam={rm.hit_distractor_vals} subst={rm.substitution}")

    echo(f"PROBE prefix_stability={'PASS' if probe_prefix_ok else 'FAIL'}")
    echo(f"PROBE fidelity_floor={'PASS' if probe_fidelity_ok else 'FAIL'}")
    if not (probe_prefix_ok and probe_fidelity_ok):
        echo("WALKING SKELETON FAILED INVARIANTS")

    # persist bundle (raw evidence for result-analysis to recompute)
    with open(os.path.join(run_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    open(os.path.join(run_dir, "stdout.log"), "w", encoding="utf-8").write("\n".join(log_lines))
    emb.flush()
    manifest = {
        "schema": "light.run_manifest.v3-lite",
        "run_id": f"walk-{stamp}",
        "purpose": "M-A walking skeleton (thin E2E, not the main matrix)",
        "authorized_action_ids": ["mod-history-append-log", "mod-prefix-stable-summary-view",
                                   "mod-distraction-cost-model", "mod-admission-controller",
                                   "mod-context-assembler-ollama", "mod-eval-metric-emitter"],
        "config": {"model": MODEL, "budget_T": BUDGET_T,
                   "calibration": params["params"], "fit_scope": params["fit_scope"],
                   "provisional": True, "temperature": asm.temperature},
        "started_at": stamp, "ended_at": dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z"),
        "invariants": {"prefix_stability": probe_prefix_ok, "fidelity_floor": probe_fidelity_ok},
        "emb_cache_sha256": _sha(EMB_CACHE) if os.path.exists(EMB_CACHE) else None,
        "n_turns": len(metrics),
        "exit_state": "COMPLETED" if (probe_prefix_ok and probe_fidelity_ok) else "FAILED",
    }
    json.dump(manifest, open(os.path.join(run_dir, "run_manifest.json"), "w"),
              ensure_ascii=False, indent=2)
    echo(f"RUN DIR: {run_dir}  exit={manifest['exit_state']}")
    return 0 if manifest["exit_state"] == "COMPLETED" else 1


def view_prefix_of(prompt):
    # bytes up to the first 'RECENT' marker == the append-only cacheable STATE head
    idx = prompt.find("RECENT")
    return prompt[:idx].encode("utf-8") if idx >= 0 else b""


if __name__ == "__main__":
    raise SystemExit(run())
