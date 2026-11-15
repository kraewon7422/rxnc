"""AST node types plus a precedence-aware unparser.

Expressions are immutable dataclasses.  ``key()`` returns a canonical string
used as the hash key for common-subexpression elimination and for structural
equality during simplification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

Pos = Tuple[int, int]

# operator -> (precedence, right_associative)
BINARY_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 4}
RIGHT_ASSOC = {"^"}
FUNCTIONS = {"exp": 1, "log": 1, "log10": 1, "sqrt": 1, "abs": 1, "pow": 2, "max": 2, "min": 2}


class Expr:
    """Base class for expression nodes."""

    pos: Pos = (0, 0)

    def key(self) -> str:  # pragma: no cover - overridden everywhere
        raise NotImplementedError


@dataclass(frozen=True)
class Num(Expr):
    value: float
    pos: Pos = (0, 0)

    def key(self) -> str:
        return repr(self.value)


@dataclass(frozen=True)
class Var(Expr):
    """A parameter, the temperature ``T``, or the gas constant ``R``."""

    name: str
    pos: Pos = (0, 0)

    def key(self) -> str:
        return f"@{self.name}"


@dataclass(frozen=True)
class Conc(Expr):
    """A species concentration, written ``[A]`` in source."""

    species: str
    pos: Pos = (0, 0)

    def key(self) -> str:
        return f"[{self.species}]"


@dataclass(frozen=True)
class Unary(Expr):
    op: str
    operand: Expr
    pos: Pos = (0, 0)

    def key(self) -> str:
        return f"(u{self.op} {self.operand.key()})"


@dataclass(frozen=True)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr
    pos: Pos = (0, 0)

    def key(self) -> str:
        return f"({self.op} {self.left.key()} {self.right.key()})"


@dataclass(frozen=True)
class Call(Expr):
    func: str
    args: Tuple[Expr, ...]
    pos: Pos = (0, 0)

    def key(self) -> str:
        return f"({self.func} " + " ".join(a.key() for a in self.args) + ")"


# --------------------------------------------------------------------------- #
# statements
# --------------------------------------------------------------------------- #
@dataclass
class ParamDef:
    name: str
    expr: Expr
    pos: Pos = (0, 0)


@dataclass
class SpeciesDef:
    name: str
    formula: Optional[str]
    pos: Pos = (0, 0)


@dataclass
class ReactionNode:
    reactants: List[Tuple[float, str]]
    products: List[Tuple[float, str]]
    reversible: bool
    clauses: Dict[str, Union[Expr, str]] = field(default_factory=dict)
    clause_pos: Dict[str, Pos] = field(default_factory=dict)
    pos: Pos = (0, 0)


@dataclass
class Program:
    statements: List[Union[ParamDef, SpeciesDef, ReactionNode]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# unparser
# --------------------------------------------------------------------------- #
def unparse(e: Expr) -> str:
    """Render an expression with the minimum number of parentheses."""
    return _unparse(e, 0)


def _unparse(e: Expr, parent_prec: int) -> str:
    if isinstance(e, Num):
        v = e.value
        return repr(int(v)) if v == int(v) and abs(v) < 1e16 else repr(v)
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Conc):
        return f"[{e.species}]"
    if isinstance(e, Call):
        return f"{e.func}(" + ", ".join(_unparse(a, 0) for a in e.args) + ")"
    if isinstance(e, Unary):
        inner = _unparse(e.operand, 3)
        return f"-{inner}" if parent_prec <= 3 else f"(-{inner})"
    if isinstance(e, Binary):
        prec = BINARY_PRECEDENCE[e.op]
        right_assoc = e.op in RIGHT_ASSOC
        left = _unparse(e.left, prec + (1 if right_assoc else 0))
        right = _unparse(e.right, prec + (0 if right_assoc else 1))
        text = f"{left} {e.op} {right}" if e.op in "+-" else f"{left}{e.op}{right}"
        return f"({text})" if prec < parent_prec else text
    raise TypeError(f"cannot unparse {e!r}")  # pragma: no cover
