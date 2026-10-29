"""Walking-skeleton failure probes (TDD red-first). No pytest installed -> plain asserts.

Run:  python tests/test_probes.py
These assert the three invariants the design froze (ADR-0002 / interface-contract / ADR-0003)
plus a couple of gold/property checks. They MUST fail before co.* exists.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from co.history_log import HistoryLog, TurnRecord           # noqa: E402
from co.summary_view import SummaryView, is_byte_prefix     # noqa: E402
from co.distraction_cost import DistractionCost, Candidate  # noqa: E402
from co.admission import AdmissionEvictor, VERBATIM, FOLD, EVICT  # noqa: E402
from co.assembler import ContextAssembler                    # noqa: E402


_DIM = 257


class FakeEmbedder:
    """Deterministic token-bag embedder (dim-consistent) so probes need no live ollama;
    cosine == normalized shared-token overlap, so proximity is monotone in token sharing."""
    def __init__(self): self.calls = 0
    def embed(self, text):
        self.calls += 1
        vec = [0.0] * _DIM
        for tok in str(text).lower().split():
            vec[hash(tok) % _DIM] += 1.0
        return vec


def _cand(uid, text, gold=False, turn=0, tokens=10):
    return Candidate(unit_id=uid, text=text, is_gold_bearing=gold, turn_id=turn, tokens=tokens)


def probe_prefix_stability():
    v = SummaryView(embedder=FakeEmbedder())
    log = HistoryLog()
    for i in range(5):
        rec = TurnRecord(turn_id=i, role="tool_result", fields={"f%d" % i: "v%d" % i},
                         raw_text="row %d" % i, tokens=8)
        log.append(rec)
        v.fold({("k%d" % i): ("val%d" % i)})
        assert is_byte_prefix(v.prev_serialization_bytes(), v.serialize_bytes()), \
            "summary prefix rewritten in place (breaks KV cache)"
    # assembled prompt must always start with the current cacheable prefix P
    asm = ContextAssembler(model="stub")
    prompt = asm.assemble(v, verbatim=[_cand("a", "gold-alpha", gold=True, turn=4)])
    assert prompt.startswith(v.prefix_text()), "verbatim tail leaked before cached prefix"


def probe_fidelity_floor():
    ev = AdmissionEvictor(embedder=FakeEmbedder(),
                          cost=DistractionCost(embedder=FakeEmbedder(), params=_params()),
                          budget_T=30)
    units = [_cand("g", "the gold value", gold=True, turn=3, tokens=25),
             _cand("d", "distractor near", gold=False, turn=1, tokens=20)]
    decision = ev.decide(units, gold_values=["the gold value"])
    assert decision["g"] != EVICT, "gold-bearing unit was evicted (fidelity floor breached)"
    # under any budget, gold must be VERBATIM or FOLD, never EVICT
    ev2 = AdmissionEvictor(embedder=FakeEmbedder(),
                           cost=DistractionCost(embedder=FakeEmbedder(), params=_params()),
                           budget_T=0)
    d2 = ev2.decide([_cand("g2", "gold", gold=True, turn=0, tokens=99)], gold_values=["gold"])
    assert d2["g2"] in (VERBATIM, FOLD), "gold evicted under zero budget"


def probe_embedding_degradation():
    class Boom(FakeEmbedder):
        def embed(self, text):  # simulate ollama embed endpoint down
            raise RuntimeError("embed endpoint unavailable")
    cost = DistractionCost(embedder=Boom(), params=_params())
    p, degraded = cost.score_with_status(
        _cand("d", "unrelated distractor text", gold=False, turn=1),
        gold_values=["alpha bravo value"],
        co_present=[_cand("d", "unrelated distractor text", gold=False, turn=1)],
    )
    assert 0.0 <= p <= 1.0, "degraded path returned out-of-range score"
    assert degraded is True, "degraded embedding not flagged"


def probe_score_monotonic_in_nsim():
    """property: higher semantic proximity of co-present distractors -> higher predicted contamination."""
    em = FakeEmbedder()
    cost = DistractionCost(embedder=em, params={"beta0": -2.0, "beta1": 4.0, "beta2": 0.3,
                                                 "tau_near": 0.2, "fit_scope": "pilot"})
    def score(near):
        d = _cand("d", near, gold=False, turn=1)
        return cost.score_with_status(d, gold_values=[near.split()[0] + " x"],
                                       co_present=[d])[0]
    assert score("alpha x rest") >= score("zzz yyy rest"), "contamination not monotone in proximity"


def _params():
    return json.loads(json.dumps({"beta0": -2.0, "beta1": 4.0, "beta2": 0.3,
                                   "tau_near": 0.2, "fit_scope": "pilot"}))


def main():
    probes = [probe_prefix_stability, probe_fidelity_floor,
              probe_embedding_degradation, probe_score_monotonic_in_nsim]
    for p in probes:
        p()
        print("PASS", p.__name__)
    print("ALL PROBES PASS")


if __name__ == "__main__":
    main()
