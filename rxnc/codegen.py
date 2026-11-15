"""Python backend: emit a straight-line kernel with common subexpressions shared.

One function evaluates the reaction rates, the nonzero d(rate)/d(C) entries and
the nonzero d(rate)/dT entries together, so temperature factors and
concentration products computed for the rates are reused by the derivatives
instead of being recomputed.

Parenthesisation is driven by an explicit atomic/compound flag on every emitted
fragment rather than by inspecting the generated text.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Sequence, Tuple

from .analyze import Mechanism
from .nodes import Binary, Call, Conc, Expr, Num, Unary, Var

HEADER = "from math import exp, log, log10, sqrt\n\n"
_PY_FUNC = {"exp": "exp", "log": "log", "log10": "log10", "sqrt": "sqrt", "abs": "abs"}


class Frag(NamedTuple):
    """A generated code fragment and whether it can stand without parentheses."""

    code: str
    atomic: bool

    def wrapped(self) -> str:
        return self.code if self.atomic else f"({self.code})"


class Emitter:
    """Builds straight-line code, naming every repeated subexpression once."""

    def __init__(self, species: Sequence[str]):
        self.index = {s: i for i, s in enumerate(species)}
        self.counts: Dict[str, int] = {}
        self.cache: Dict[str, Frag] = {}
        self.lines: List[str] = []
        self.n_temp = 0
        self.reused = 0

    # -- pass 1: occurrence counts ------------------------------------------ #
    def count(self, e: Expr) -> None:
        key = e.key()
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] > 1:
            return  # children were counted with the first occurrence
        for child in _children(e):
            self.count(child)

    # -- pass 2: emit -------------------------------------------------------- #
    def emit(self, e: Expr) -> Frag:
        key = e.key()
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        frag = self._build(e)
        if self.counts.get(key, 0) > 1 and not isinstance(e, (Num, Var, Conc)):
            name = f"_t{self.n_temp}"
            self.n_temp += 1
            self.reused += self.counts[key] - 1
            self.lines.append(f"    {name} = {frag.code}")
            frag = Frag(name, True)
        self.cache[key] = frag
        return frag

    def _build(self, e: Expr) -> Frag:
        if isinstance(e, Num):
            return Frag(repr(e.value), e.value >= 0)
        if isinstance(e, Var):
            return Frag("T" if e.name == "T" else e.name, True)
        if isinstance(e, Conc):
            return Frag(f"_c{self.index[e.species]}", True)
        if isinstance(e, Unary):
            return Frag(f"-{self.emit(e.operand).wrapped()}", False)
        if isinstance(e, Binary):
            a, b = self.emit(e.left), self.emit(e.right)
            if e.op == "^":
                return Frag(f"{a.wrapped()}**{b.wrapped()}", False)
            return Frag(f"{a.wrapped()} {e.op} {b.wrapped()}", False)
        if isinstance(e, Call):
            if e.func == "pow":
                a, b = self.emit(e.args[0]), self.emit(e.args[1])
                return Frag(f"{a.wrapped()}**{b.wrapped()}", False)
            args = ", ".join(self.emit(a).code for a in e.args)
            return Frag(f"{_PY_FUNC.get(e.func, e.func)}({args})", True)
        raise TypeError(f"cannot emit {e!r}")  # pragma: no cover


def _children(e: Expr) -> Tuple[Expr, ...]:
    if isinstance(e, Unary):
        return (e.operand,)
    if isinstance(e, Binary):
        return (e.left, e.right)
    if isinstance(e, Call):
        return tuple(e.args)
    return ()


def generate(
    mech: Mechanism,
) -> Tuple[str, Tuple[Tuple[int, int], ...], Tuple[int, ...], Dict[str, int]]:
    """Generate kernel source.

    Returns ``(source, dC_pattern, dT_pattern, stats)``; the patterns give the
    meaning of the kernel's second and third return values.
    """
    from .symbolic import species_used

    rates = mech.rate_exprs()
    dC = mech.drate_dC()
    dT = mech.drate_dT()
    dC_pattern = tuple(sorted(dC))
    dT_pattern = tuple(sorted(dT))

    em = Emitter(mech.species)
    todo: List[Expr] = list(rates) + [dC[k] for k in dC_pattern] + [dT[k] for k in dT_pattern]
    for e in todo:
        em.count(e)

    used = sorted({em.index[s] for e in todo for s in species_used(e)})
    prologue = [f"    _c{i} = C[{i}]" for i in used]

    rate_code = [em.emit(e).code for e in rates]
    dC_code = [em.emit(dC[k]).code for k in dC_pattern]
    dT_code = [em.emit(dT[k]).code for k in dT_pattern]

    body = "\n".join(prologue + em.lines)
    src = HEADER + "def kernel(C, T):\n"
    src += '    """Return (rates, nonzero d(rate)/dC, nonzero d(rate)/dT)."""\n'
    if body:
        src += body + "\n"
    src += "    return (\n"
    for fragments in (rate_code, dC_code, dT_code):
        inner = ", ".join(fragments) + ("," if len(fragments) == 1 else "")
        src += f"        ({inner}),\n"
    src += "    )\n"
    stats = {
        "temporaries": em.n_temp,
        "reused_subexpressions": em.reused,
        "rate_entries": len(rate_code),
        "jacobian_entries": len(dC_code),
    }
    return src, dC_pattern, dT_pattern, stats


def load(source: str, name: str = "rxnc_generated"):
    """Execute generated source and return its ``kernel`` function."""
    namespace: Dict[str, object] = {"__name__": name}
    exec(compile(source, f"<{name}>", "exec"), namespace)  # noqa: S102 - self-generated
    return namespace["kernel"]


def generate_standalone(mech: Mechanism, origin: str = "<string>") -> str:
    """Generated kernel plus the metadata needed to use it without rxnc."""
    src, dC_pattern, dT_pattern, _stats = generate(mech)
    head = (
        '"""Generated by rxnc -- do not edit.\n\n'
        f"Source: {origin}\n"
        f"{mech.n_reactions} reactions, {mech.n_species} species.\n"
        "The kernel returns (rates, dC_values, dT_values); DC_PATTERN[i] gives the\n"
        "(reaction, species) index pair of dC_values[i] and DT_PATTERN[i] the\n"
        'reaction index of dT_values[i].\n"""\n\n'
    )
    meta = [
        "",
        f"SPECIES = {mech.species!r}",
        f"NU = {mech.nu_matrix()!r}",
        f"DH = {[r.dH for r in mech.reactions]!r}",
        f"DC_PATTERN = {[tuple(k) for k in dC_pattern]!r}",
        f"DT_PATTERN = {list(dT_pattern)!r}",
        "",
    ]
    return head + src + "\n".join(meta)
