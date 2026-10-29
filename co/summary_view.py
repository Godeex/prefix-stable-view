"""mod-prefix-stable-summary-view: the streaming materialized view.

Invariant (ADR-0002): the serialized view is APPEND-ONLY so the KV-cacheable prefix
never changes bytes. fold() may only extend; it must never rewrite earlier bytes.
"""


def is_byte_prefix(a: bytes, b: bytes) -> bool:
    return b.startswith(a)


class SummaryView:
    def __init__(self, embedder=None):
        self.embedder = embedder
        self._bindings = []          # ordered list of (key, value) appended, never mutated
        self._ser = b""              # current serialized bytes
        self._prev_ser = b""         # serialized bytes as of the previous fold (for prefix assertions)

    def fold(self, new_bindings: dict) -> None:
        # append-only: add new (key,value) rows; a re-bound key appends a later row,
        # it does not rewrite the earlier one (keeps the cached prefix byte-stable).
        self._prev_ser = self._ser
        for k, val in new_bindings.items():
            self._bindings.append((k, str(val)))
        self._ser = self.serialize_bytes()

    def _render(self) -> str:
        return "\n".join(f"{k}={v}" for k, v in self._bindings)

    def serialize_bytes(self) -> bytes:
        return self._render().encode("utf-8")

    def prev_serialization_bytes(self) -> bytes:
        return self._prev_ser   # snapshot from before the most recent fold

    def prefix_text(self) -> str:
        return self._render()

    def gold_value(self, key: str):
        for k, v in reversed(self._bindings):   # latest binding wins for lookups
            if k == key:
                return v
        return None

    def size_tokens(self) -> int:
        return len(self.prefix_text())          # coarse token proxy for the skeleton
