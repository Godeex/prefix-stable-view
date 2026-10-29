"""mod-admission-controller: utility - prefix-depth - distraction, under token budget.

score(u) = future_utility(u) - lam_recompute*prefix_invalidate_depth(u)
                       - distraction_cost(u)
Fidelity floor (hard): is_gold_bearing units are NEVER evicted (VERBATIM or FOLD only).
Selection greedily fills the budget by score; the budget alone may still cut a distractor.
"""
from .distraction_cost import Candidate  # noqa: F401  (re-export for type clarity)

VERBATIM, FOLD, EVICT = "VERBATIM", "FOLD", "EVICT"


class AdmissionEvictor:
    def __init__(self, embedder, cost, budget_T, lam_recompute=0.5):
        self.embedder = embedder
        self.cost = cost
        self.budget_T = budget_T
        self.lam_recompute = lam_recompute

    def _utility(self, u, newest_turn):
        # heuristic (NOT a learned model): recency dominates utility for the skeleton.
        return 1.0 / (1.0 + max(0, newest_turn - u.turn_id))

    def _prefix_depth(self, u):
        # older units sit deeper in the prefix -> evicting/folding them costs more recompute
        return 1.0 / (1.0 + u.turn_id)

    def decide(self, units, gold_values):
        newest = max((u.turn_id for u in units), default=0)
        scored = []
        for u in units:
            p, _degraded = self.cost.score_with_status(u, gold_values, units)
            s = self._utility(u, newest) - self.lam_recompute * self._prefix_depth(u) - p
            scored.append((s, u))
        scored.sort(key=lambda x: -x[0])
        out, used = {}, 0
        for s, u in scored:
            if u.is_gold_bearing:
                # fidelity floor: gold always kept, at least FOLD, VERBATIM if budget allows
                out[u.unit_id] = VERBATIM if used + u.tokens <= self.budget_T else FOLD
                used += min(u.tokens, self.budget_T)
                continue
            if used + u.tokens <= self.budget_T:
                out[u.unit_id] = VERBATIM
                used += u.tokens
            else:
                out[u.unit_id] = EVICT
        return out
