"""Hand-written tokenizer for the reaction DSL.

Token kinds: NUMBER, IDENT, KEYWORD, STRING, OP, ARROW, REV, NEWLINE, EOF.
Newlines are significant (they terminate statements); everything else is
free-form whitespace.  ``#`` starts a comment that runs to end of line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .errors import SyntaxErrorRxnc

KEYWORDS = {"let", "species"}
OPERATORS = {"+", "-", "*", "/", "^", "(", ")", "[", "]", ",", ":", "="}


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    col: int
    value: Optional[float] = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Token({self.kind}, {self.text!r}, {self.line}:{self.col})"


def _is_ident_start(ch: str) -> bool:
    return ch.isalpha() or ch == "_"


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def tokenize(src: str) -> List[Token]:
    """Turn source text into a token list ending with a single EOF token."""
    toks: List[Token] = []
    i, line, line_start = 0, 1, 0
    n = len(src)

    def col(at: int) -> int:
        return at - line_start + 1

    while i < n:
        ch = src[i]
        if ch == "\n":
            toks.append(Token("NEWLINE", "\\n", line, col(i)))
            i += 1
            line += 1
            line_start = i
            continue
        if ch in " \t\r":
            i += 1
            continue
        if ch == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if ch == "\\" and i + 1 < n and src[i + 1] == "\n":  # line continuation
            i += 2
            line += 1
            line_start = i
            continue
        start = i
        # multi-character operators first
        if src.startswith("<=>", i) or src.startswith("<->", i):
            toks.append(Token("REV", src[i : i + 3], line, col(i)))
            i += 3
            continue
        if src.startswith("->", i):
            toks.append(Token("ARROW", "->", line, col(i)))
            i += 2
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and src[i + 1].isdigit()):
            i += 1
            while i < n and (src[i].isdigit() or src[i] == "."):
                i += 1
            if i < n and src[i] in "eE":
                j = i + 1
                if j < n and src[j] in "+-":
                    j += 1
                if j < n and src[j].isdigit():
                    i = j
                    while i < n and src[i].isdigit():
                        i += 1
            text = src[start:i]
            try:
                val = float(text)
            except ValueError as exc:  # pragma: no cover - lexer shape prevents this
                raise SyntaxErrorRxnc(f"malformed number {text!r}", line, col(start)) from exc
            toks.append(Token("NUMBER", text, line, col(start), val))
            continue
        if _is_ident_start(ch):
            i += 1
            while i < n and _is_ident_char(src[i]):
                i += 1
            text = src[start:i]
            kind = "KEYWORD" if text in KEYWORDS else "IDENT"
            toks.append(Token(kind, text, line, col(start)))
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            buf = []
            while i < n and src[i] != quote:
                if src[i] == "\n":
                    raise SyntaxErrorRxnc("unterminated string", line, col(start))
                buf.append(src[i])
                i += 1
            if i >= n:
                raise SyntaxErrorRxnc("unterminated string", line, col(start))
            i += 1
            toks.append(Token("STRING", "".join(buf), line, col(start)))
            continue
        if ch in OPERATORS:
            toks.append(Token("OP", ch, line, col(i)))
            i += 1
            continue
        raise SyntaxErrorRxnc(f"unexpected character {ch!r}", line, col(i))

    toks.append(Token("NEWLINE", "\\n", line, col(n)))
    toks.append(Token("EOF", "", line, col(n)))
    return toks
