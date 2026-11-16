"""Chemical formula parsing and element balance checking."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

from .errors import SemanticError

ELEMENTS = set(
    """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge
    As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu
    Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu
    Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og D""".split()
)


def parse_formula(text: str, line: int = 0, col: int = 0) -> Dict[str, float]:
    """Parse ``C6H12O6`` / ``Ca(OH)2`` into an element -> count mapping.

    Supports nested parentheses and fractional subscripts.  Raises
    :class:`SemanticError` on an unknown element symbol or bad syntax.
    """
    counts: Counter = Counter()
    stack: List[Counter] = [counts]
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            stack.append(Counter())
            i += 1
            continue
        if ch == ")":
            if len(stack) == 1:
                raise SemanticError(f"unbalanced ')' in formula {text!r}", line, col)
            group = stack.pop()
            i += 1
            mult, i = _read_number(text, i)
            for el, c in group.items():
                stack[-1][el] += c * mult
            continue
        if ch.isalpha():
            j = i + 1
            if j < n and text[j].islower():
                j += 1
            symbol = text[i:j]
            if symbol not in ELEMENTS:
                raise SemanticError(
                    f"unknown element symbol {symbol!r} in formula {text!r}", line, col
                )
            i = j
            mult, i = _read_number(text, i)
            stack[-1][symbol] += mult
            continue
        raise SemanticError(f"unexpected character {ch!r} in formula {text!r}", line, col)
    if len(stack) != 1:
        raise SemanticError(f"unbalanced '(' in formula {text!r}", line, col)
    return dict(counts)


def _read_number(text: str, i: int) -> Tuple[float, int]:
    start = i
    while i < len(text) and (text[i].isdigit() or text[i] == "."):
        i += 1
    if start == i:
        return 1.0, i
    return float(text[start:i]), i


def looks_like_formula(name: str) -> bool:
    """True when a species name parses as a formula AND is unambiguous enough.

    A bare ``A`` or ``B`` would parse as argon-free nonsense or boron, so a
    name only counts as self-describing when it contains a digit or more than
    one element symbol.  Used for *suggestions*, never for silent balancing.
    """
    try:
        counts = parse_formula(name)
    except SemanticError:
        return False
    if not counts or any(v <= 0 for v in counts.values()):
        return False  # e.g. "S0" parses as zero sulfur atoms: a label, not a formula
    return any(ch.isdigit() for ch in name) or len(counts) > 1


def balance_residual(
    terms: Iterable[Tuple[float, Dict[str, float]]],
) -> Dict[str, float]:
    """Signed element totals for ``(coefficient, formula_counts)`` pairs."""
    total: Counter = Counter()
    for coeff, counts in terms:
        for el, c in counts.items():
            total[el] += coeff * c
    return {el: v for el, v in total.items() if abs(v) > 1e-9}
