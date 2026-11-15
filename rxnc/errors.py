"""Diagnostics with source positions.

Every compiler stage raises :class:`RxncError` (or a subclass) carrying the
line/column where the problem is, so the CLI can print a caret under the
offending token instead of a bare message.
"""

from __future__ import annotations


class RxncError(Exception):
    """A compile-time error with a source position."""

    stage = "error"

    def __init__(self, message: str, line: int = 0, col: int = 0, source: str | None = None):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col
        self.source = source

    def with_source(self, source: str) -> RxncError:
        self.source = source
        return self

    def render(self) -> str:
        """Message, location and a caret under the offending column."""
        head = f"{self.stage}: {self.message}"
        if not self.line:
            return head
        head += f"\n  --> line {self.line}, column {self.col}"
        if self.source:
            lines = self.source.splitlines()
            if 0 < self.line <= len(lines):
                text = lines[self.line - 1]
                head += f"\n   |\n   | {text}\n   | {' ' * max(self.col - 1, 0)}^"
        return head

    def __str__(self) -> str:  # pragma: no cover - exercised through render()
        return self.render()


class SyntaxErrorRxnc(RxncError):
    stage = "syntax error"


class SemanticError(RxncError):
    stage = "semantic error"


class BalanceError(SemanticError):
    stage = "balance error"
