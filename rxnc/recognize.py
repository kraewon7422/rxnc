"""Recognise modified-Arrhenius rate coefficients in an arbitrary expression.

The DSL lets ``k:`` hold any algebra over ``T``.  Exporting to a solver that
only understands ``A (T/T_ref)^b exp(-Ea/(R T))`` therefore needs a pass that
decides whether a given expression *is* that form, however it was spelled.

Identification is numeric -- fit the three parameters from three sample
temperatures -- followed by verification at four further temperatures.  An
expression that does not reproduce itself to 1e-9 relative error is rejected,
so the pass can never silently mis-export a non-Arrhenius rate law.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

from .interp import evaluate
from .nodes import Expr
from .symbolic import depends_on, species_used

R_GAS = 8.314462618
FIT_T = (300.0, 400.0, 500.0)
CHECK_T = (280.0, 350.0, 450.0, 600.0)
RTOL = 1e-9


def recognize_arrhenius(
    k: Expr, T_ref: float = 298.15, fit_T: Sequence[float] = FIT_T
) -> Optional[Dict[str, float]]:
    """Return ``{A, Ea, b, T_ref}`` if ``k`` is modified-Arrhenius, else ``None``."""
    if species_used(k):
        return None
    if not depends_on(k, "T"):
        try:
            value = evaluate(k, {}, 300.0)
        except Exception:
            return None
        if value <= 0.0 or not math.isfinite(value):
            return None
        return {"A": value, "Ea": 0.0, "b": 0.0, "T_ref": T_ref}

    try:
        rows = []
        rhs = []
        for T in fit_T:
            value = evaluate(k, {}, T)
            if value <= 0.0 or not math.isfinite(value):
                return None
            rows.append([1.0, math.log(T / T_ref), -1.0 / (R_GAS * T)])
            rhs.append(math.log(value))
        solution = _solve3(rows, rhs)
        if solution is None:
            return None
        ln_A, b, Ea = solution
        if not all(math.isfinite(v) for v in (ln_A, b, Ea)):
            return None
        A = math.exp(ln_A)
        if abs(b) < 1e-10:
            b = 0.0
        if abs(Ea) < 1e-7:
            Ea = 0.0
        for T in CHECK_T:
            want = evaluate(k, {}, T)
            got = A * (T / T_ref) ** b * math.exp(-Ea / (R_GAS * T))
            if not math.isfinite(got) or abs(got - want) > RTOL * max(abs(want), 1e-300):
                return None
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    return {"A": A, "Ea": Ea, "b": b, "T_ref": T_ref}


def _solve3(rows, rhs):
    """Gaussian elimination with partial pivoting on a 3x3 system."""
    m = [list(r) + [v] for r, v in zip(rows, rhs, strict=True)]
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    x = [0.0] * n
    for r in reversed(range(n)):
        acc = m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))
        x[r] = acc / m[r][r]
    return x
