"""Simplification and symbolic differentiation over the expression AST.

``simplify`` is a bottom-up rewrite: constant folding plus the usual algebraic
identities.  ``diff`` differentiates with respect to a temperature or a species
concentration and simplifies as it goes, so the emitted Jacobian entries stay
readable (and cheap) rather than exploding into ``1*x + 0`` chains.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .errors import SemanticError
from .nodes import Binary, Call, Conc, Expr, Num, Unary, Var

ZERO = Num(0.0)
ONE = Num(1.0)

_UNARY_FUNCS = {
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
}
_BINARY_FUNCS = {"pow": math.pow, "max": max, "min": min}


def is_num(e: Expr, value: Optional[float] = None) -> bool:
    return isinstance(e, Num) and (value is None or e.value == value)


def simplify(e: Expr) -> Expr:
    """Fold constants and apply algebraic identities bottom-up."""
    if isinstance(e, (Num, Var, Conc)):
        return e
    if isinstance(e, Unary):
        x = simplify(e.operand)
        if isinstance(x, Num):
            return Num(-x.value)
        if isinstance(x, Unary) and x.op == "-":
            return x.operand
        return Unary("-", x)
    if isinstance(e, Call):
        args = tuple(simplify(a) for a in e.args)
        if all(isinstance(a, Num) for a in args):
            vals = [a.value for a in args]  # type: ignore[attr-defined]
            if e.func in _UNARY_FUNCS and len(vals) == 1:
                return Num(float(_UNARY_FUNCS[e.func](vals[0])))
            if e.func in _BINARY_FUNCS and len(vals) == 2:
                return Num(float(_BINARY_FUNCS[e.func](vals[0], vals[1])))
        if e.func == "exp" and is_num(args[0], 0.0):
            return ONE
        if e.func == "pow":
            return simplify(Binary("^", args[0], args[1]))
        return Call(e.func, args)
    if isinstance(e, Binary):
        a, b = simplify(e.left), simplify(e.right)
        op = e.op
        if isinstance(a, Num) and isinstance(b, Num):
            return Num(_apply(op, a.value, b.value))
        if op == "+":
            if is_num(a, 0.0):
                return b
            if is_num(b, 0.0):
                return a
            if isinstance(b, Unary) and b.op == "-":
                return simplify(Binary("-", a, b.operand))
        elif op == "-":
            if is_num(b, 0.0):
                return a
            if is_num(a, 0.0):
                return simplify(Unary("-", b))
            if a.key() == b.key():
                return ZERO
        elif op == "*":
            if is_num(a, 0.0) or is_num(b, 0.0):
                return ZERO
            if is_num(a, 1.0):
                return b
            if is_num(b, 1.0):
                return a
            if isinstance(b, Num) and not isinstance(a, Num):
                a, b = b, a  # keep numeric factors on the left
            # (c1*x)*c2 -> (c1*c2)*x
            if (
                isinstance(a, Num)
                and isinstance(b, Binary)
                and b.op == "*"
                and isinstance(b.left, Num)
            ):
                return simplify(Binary("*", Num(a.value * b.left.value), b.right))
        elif op == "/":
            if is_num(a, 0.0):
                return ZERO
            if is_num(b, 1.0):
                return a
            if a.key() == b.key():
                return ONE
        elif op == "^":
            if is_num(b, 1.0):
                return a
            if is_num(b, 0.0):
                return ONE
            if is_num(a, 1.0):
                return ONE
        return Binary(op, a, b)
    raise TypeError(f"cannot simplify {e!r}")  # pragma: no cover


def _apply(op: str, x: float, y: float) -> float:
    if op == "+":
        return x + y
    if op == "-":
        return x - y
    if op == "*":
        return x * y
    if op == "/":
        if y == 0.0:
            raise SemanticError("division by zero while folding constants")
        return x / y
    if op == "^":
        return float(x**y)
    raise SemanticError(f"unknown operator {op!r}")  # pragma: no cover


def substitute(e: Expr, params: Dict[str, float]) -> Expr:
    """Replace parameter names by their numeric values, then simplify."""
    return simplify(_subst(e, params))


def _subst(e: Expr, params: Dict[str, float]) -> Expr:
    if isinstance(e, Var):
        if e.name in params:
            return Num(params[e.name], e.pos)
        return e
    if isinstance(e, (Num, Conc)):
        return e
    if isinstance(e, Unary):
        return Unary(e.op, _subst(e.operand, params), e.pos)
    if isinstance(e, Binary):
        return Binary(e.op, _subst(e.left, params), _subst(e.right, params), e.pos)
    if isinstance(e, Call):
        return Call(e.func, tuple(_subst(a, params) for a in e.args), e.pos)
    raise TypeError(f"cannot substitute into {e!r}")  # pragma: no cover


def diff(e: Expr, wrt: str) -> Expr:
    """d(expr)/d(wrt), where ``wrt`` is ``"T"`` or ``"[A]"`` for a species.

    The result is simplified; a derivative that vanishes comes back as ``Num(0)``.
    """
    return simplify(_diff(e, wrt))


def _diff(e: Expr, wrt: str) -> Expr:
    if isinstance(e, Num):
        return ZERO
    if isinstance(e, Var):
        return ONE if f"@{e.name}" == wrt or e.name == wrt else ZERO
    if isinstance(e, Conc):
        return ONE if f"[{e.species}]" == wrt else ZERO
    if isinstance(e, Unary):
        return Unary("-", _diff(e.operand, wrt))
    if isinstance(e, Binary):
        a, b = e.left, e.right
        da, db = simplify(_diff(a, wrt)), simplify(_diff(b, wrt))
        if e.op in "+-":
            return Binary(e.op, da, db)
        if e.op == "*":
            return Binary("+", Binary("*", da, b), Binary("*", a, db))
        if e.op == "/":
            if is_num(db, 0.0):
                return Binary("/", da, b)
            num = Binary("-", Binary("*", da, b), Binary("*", a, db))
            return Binary("/", num, Binary("^", b, Num(2.0)))
        if e.op == "^":
            if isinstance(b, Num):  # power rule
                outer = Binary("*", Num(b.value), Binary("^", a, Num(b.value - 1.0)))
                return Binary("*", outer, da)
            # general: u^v * (v' ln u + v u' / u)
            inner = Binary(
                "+",
                Binary("*", db, Call("log", (a,))),
                Binary("/", Binary("*", b, da), a),
            )
            return Binary("*", Binary("^", a, b), inner)
        raise SemanticError(f"cannot differentiate operator {e.op!r}")  # pragma: no cover
    if isinstance(e, Call):
        if e.func in ("max", "min"):
            raise SemanticError(
                f"{e.func}() is not differentiable; the analytic Jacobian cannot be built "
                "for this rate law"
            )
        if e.func == "pow":
            return _diff(Binary("^", e.args[0], e.args[1]), wrt)
        x = e.args[0]
        dx = _diff(x, wrt)
        if e.func == "exp":
            return Binary("*", Call("exp", (x,)), dx)
        if e.func == "log":
            return Binary("/", dx, x)
        if e.func == "log10":
            return Binary("/", dx, Binary("*", x, Num(math.log(10.0))))
        if e.func == "sqrt":
            return Binary("/", dx, Binary("*", Num(2.0), Call("sqrt", (x,))))
        if e.func == "abs":
            return Binary("*", dx, Binary("/", x, Call("abs", (x,))))
        raise SemanticError(f"unknown function {e.func!r}")  # pragma: no cover
    raise TypeError(f"cannot differentiate {e!r}")  # pragma: no cover


def depends_on(e: Expr, wrt: str) -> bool:
    """Whether ``wrt`` (``"T"`` or ``"[A]"``) appears anywhere in the tree."""
    if isinstance(e, Num):
        return False
    if isinstance(e, Var):
        return e.name == wrt
    if isinstance(e, Conc):
        return f"[{e.species}]" == wrt
    if isinstance(e, Unary):
        return depends_on(e.operand, wrt)
    if isinstance(e, Binary):
        return depends_on(e.left, wrt) or depends_on(e.right, wrt)
    if isinstance(e, Call):
        return any(depends_on(a, wrt) for a in e.args)
    return False  # pragma: no cover


def species_used(e: Expr) -> set:
    """Set of species names appearing as concentrations in the expression."""
    out: set = set()
    _walk_species(e, out)
    return out


def _walk_species(e: Expr, out: set) -> None:
    if isinstance(e, Conc):
        out.add(e.species)
    elif isinstance(e, Unary):
        _walk_species(e.operand, out)
    elif isinstance(e, Binary):
        _walk_species(e.left, out)
        _walk_species(e.right, out)
    elif isinstance(e, Call):
        for a in e.args:
            _walk_species(a, out)
