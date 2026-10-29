"""mod-distraction-cost-model: the CBO-style pre-execution cost term (ADR-0003).

p(contamination) = sigmoid(beta0 + beta1*nsim + beta2*K)
  nsim = mean over co-present distractor units of max cosine to any gold value
  K    = count of co-present units whose proximity > tau_near
NO token/length term here by design: M-B showed the surrogate is orthogonal to
length (VIF~1); length is constrained separately by the token budget in admission.
"""
import math


class Candidate:
    __slots__ = ("unit_id", "text", "is_gold_bearing", "turn_id", "tokens")

    def __init__(self, unit_id, text, is_gold_bearing=False, turn_id=0, tokens=0):
        self.unit_id = unit_id
        self.text = text
        self.is_gold_bearing = is_gold_bearing
        self.turn_id = turn_id
        self.tokens = tokens


def _cos(a, b):
    n = max(len(a), len(b))
    a = list(a) + [0.0] * (n - len(a))
    b = list(b) + [0.0] * (n - len(b))
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def _overlap(a, b):
    ta, tb = set(str(a).lower().split()), set(str(b).lower().split())
    return len(ta & tb) / (len(ta | tb) + 1e-9)


class DistractionCost:
    def __init__(self, embedder, params):
        self.embedder = embedder
        self.params = params

    def score_with_status(self, cand: Candidate, gold_values, co_present):
        """Return (prob_contamination, degraded: bool)."""
        gold_values = list(gold_values or [])
        distractors = [c for c in co_present if not c.is_gold_bearing]
        if not gold_values or not distractors:
            return 0.0, False
        degraded = False
        tau = self.params.get("tau_near", 0.2)
        try:
            gv = [self.embedder.embed(g) for g in gold_values]
            sims = []
            for d in distractors:
                dv = self.embedder.embed(d.text)
                sims.append(max(_cos(dv, g) for g in gv))
        except Exception:
            # ADR-0003 degrade path: fall back to lexical overlap, still emit a score.
            degraded = True
            sims = [max(_overlap(d.text, g) for g in gold_values) for d in distractors]
        nsim = sum(sims) / len(sims)
        k = sum(1 for s in sims if s > tau)
        p = _sigmoid(self.params["beta0"] + self.params["beta1"] * nsim
                     + self.params["beta2"] * k)
        return p, degraded

    def score(self, cand, gold_values, co_present):
        return self.score_with_status(cand, gold_values, co_present)[0]
