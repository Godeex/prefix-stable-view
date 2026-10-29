"""mod-eval-metric-emitter: per-round metrics for the main analysis (same judgement rules as pilot)."""
from dataclasses import dataclass, asdict, field


@dataclass
class RoundMetric:
    turn: int
    prompt_tokens: int = 0
    ttft_ms: float = 0.0
    decision_ms: float = 0.0
    answer: str = ""
    gold: str = ""
    distractor_vals: list = field(default_factory=list)
    hit_distractor_vals: list = field(default_factory=list)   # contamination: values present ONLY in distractors
    gold_present: bool = False
    substitution: bool = False
    tool_reused: bool | None = None
    task_success: bool | None = None
    degraded_embed: bool = False

    def judge(self):
        """Contamination + substitution labels (mirrors pilot run_infer rules)."""
        low = self.answer.lower()
        self.hit_distractor_vals = [v for v in self.distractor_vals if str(v).lower() in low]
        self.gold_present = str(self.gold).lower() in low
        if self.hit_distractor_vals and not self.gold_present:
            self.substitution = True          # gave a distractor value, dropped gold
        if self.substitution or self.hit_distractor_vals:
            self.task_success = False
        elif self.gold_present:
            self.task_success = True
        return self

    def to_dict(self):
        return asdict(self)
