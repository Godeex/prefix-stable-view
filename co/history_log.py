"""mod-history-append-log: append-only turn log (fidelity floor: history never rewritten)."""
from dataclasses import dataclass, field


@dataclass
class TurnRecord:
    turn_id: int
    role: str                       # user | assistant | tool_result
    fields: dict = field(default_factory=dict)   # name -> value from a tool observation
    raw_text: str = ""
    tokens: int = 0


class HistoryLog:
    def __init__(self):
        self._turns = []

    def append(self, rec: TurnRecord) -> None:
        # append-only: reject reuse of a turn_id, never mutate prior entries
        if any(t.turn_id == rec.turn_id for t in self._turns):
            raise ValueError(f"duplicate turn_id {rec.turn_id} (log is append-only)")
        self._turns.append(rec)

    def __iter__(self):
        return iter(self._turns)

    def __len__(self):
        return len(self._turns)

    def last(self):
        return self._turns[-1] if self._turns else None
