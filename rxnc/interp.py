"""Tree-walking evaluator: the reference backend.

Slow but obviously correct, so the generated code can be differential-tested
against it on random states.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from .analyze import Mechanism
from .errors import SemanticError
from .nodes import Binary, Call, Conc, Expr, Num, Unary, Var

_UNARY = {
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
}
_BINARY = {"pow": math.pow, "max": max, "min": min}


def evaluate(expr: Expr, conc: Dict[str, float], T: float) -> float:
    """Evaluate one expression at a given state."""
    if isinstance(expr, Num):
        return expr.value
    if isinstance(expr, Var):
        if expr.name == "T":
            return T
        raise SemanticError(f"unresolved name {expr.name!r} at evaluation time")
    if isinstance(expr, Conc):
        return conc[expr.species]
    if isinstance(expr, Unary):
        return -evaluate(expr.operand, conc, T)
    if isinstance(expr, Binary):
        a = evaluate(expr.left, conc, T)
        b = evaluate(expr.right, conc, T)
        if expr.op == "+":
            return a + b
        if expr.op == "-":
            return a - b
        if expr.op == "*":
            return a * b
        if expr.op == "/":
            return a / b
        return float(a**b)
    if isinstance(expr, Call):
        args = [evaluate(a, conc, T) for a in expr.args]
        if expr.func in _UNARY:
            return float(_UNARY[expr.func](args[0]))
        return float(_BINARY[expr.func](*args))
    raise TypeError(f"cannot evaluate {expr!r}")  # pragma: no cover


class InterpreterKernel:
    """Backend with the same call contract as the generated code."""

    name = "interp"

    def __init__(self, mech: Mechanism):
        self.species = list(mech.species)
        self.rates: List[Expr] = mech.rate_exprs()
        dC = mech.drate_dC()
        dT = mech.drate_dT()
        self.dC_pattern: Tuple[Tuple[int, int], ...] = tuple(sorted(dC))
        self.dT_pattern: Tuple[int, ...] = tuple(sorted(dT))
        self._dC = [dC[key] for key in self.dC_pattern]
        self._dT = [dT[key] for key in self.dT_pattern]

    def __call__(self, C: Sequence[float], T: float):
        conc = {s: float(C[i]) for i, s in enumerate(self.species)}
        rates = [evaluate(e, conc, T) for e in self.rates]
        dC = [evaluate(e, conc, T) for e in self._dC]
        dT = [evaluate(e, conc, T) for e in self._dT]
        return rates, dC, dT
