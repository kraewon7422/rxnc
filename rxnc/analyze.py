"""Semantic analysis: name resolution, mass-action expansion, element balance.

Output is the :class:`Mechanism` IR, in which every rate law is a single
expression tree over ``T`` and species concentrations only -- all parameters
have been folded to numbers -- plus the stoichiometry matrix and the symbolic
derivatives the backends need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import symbolic as sym
from .errors import BalanceError, SemanticError
from .formula import balance_residual, looks_like_formula, parse_formula
from .nodes import (
    FUNCTIONS,
    Binary,
    Call,
    Conc,
    Expr,
    Num,
    ParamDef,
    Program,
    ReactionNode,
    SpeciesDef,
    Unary,
    Var,
    unparse,
)
from .parser import parse

R_GAS = 8.314462618  # J/(mol K)
BUILTIN_PARAMS = {"R": R_GAS}
RESERVED = {"T", "R"}


@dataclass
class ReactionIR:
    """One reaction with its net rate expression and stoichiometry."""

    index: int
    nu: Dict[str, float]
    rate: Expr
    dH: float = 0.0
    label: Optional[str] = None
    reversible: bool = False
    line: int = 0
    equation: str = ""
    # mass-action provenance, kept so the ReactionNetwork adapter can recognise
    # classic Arrhenius reactions; None when the user wrote an explicit rate law
    ma_forward: Optional[Tuple[Expr, Dict[str, float]]] = None
    ma_reverse: Optional[Tuple[Expr, Dict[str, float]]] = None
    balance_checked: bool = False

    @property
    def name(self) -> str:
        return self.label or f"R{self.index + 1}"


@dataclass
class Mechanism:
    """Analysed mechanism: the compiler's intermediate representation."""

    species: List[str]
    reactions: List[ReactionIR]
    params: Dict[str, float] = field(default_factory=dict)
    formulas: Dict[str, Optional[str]] = field(default_factory=dict)
    source: str = ""
    unbalanced_note: List[str] = field(default_factory=list)

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        return len(self.reactions)

    def nu_matrix(self) -> List[List[float]]:
        return [[r.nu.get(s, 0.0) for s in self.species] for r in self.reactions]

    def rate_exprs(self) -> List[Expr]:
        return [r.rate for r in self.reactions]

    def drate_dC(self) -> Dict[Tuple[int, int], Expr]:
        """Nonzero d(rate_j)/d(C_i) entries, keyed by (reaction, species)."""
        out: Dict[Tuple[int, int], Expr] = {}
        for j, rxn in enumerate(self.reactions):
            used = sym.species_used(rxn.rate)
            for i, s in enumerate(self.species):
                if s not in used:
                    continue
                d = sym.diff(rxn.rate, f"[{s}]")
                if not sym.is_num(d, 0.0):
                    out[(j, i)] = d
        return out

    def drate_dT(self) -> Dict[int, Expr]:
        """Nonzero d(rate_j)/dT entries, keyed by reaction index."""
        out: Dict[int, Expr] = {}
        for j, rxn in enumerate(self.reactions):
            if not sym.depends_on(rxn.rate, "T"):
                continue
            d = sym.diff(rxn.rate, "T")
            if not sym.is_num(d, 0.0):
                out[j] = d
        return out

    def summary(self) -> str:
        lines = [
            f"{self.n_reactions} reactions, {self.n_species} species: " + ", ".join(self.species)
        ]
        for r in self.reactions:
            lines.append(f"  {r.name}: {r.equation}")
            lines.append(f"      rate = {unparse(r.rate)}")
            if r.dH:
                lines.append(f"      dH   = {r.dH:g} J/mol")
        lines.extend("  note: " + n for n in self.unbalanced_note)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def analyze(program: Program, source: str = "", strict_balance: bool = False) -> Mechanism:
    """Resolve names, expand mass-action shorthand and check element balance."""
    params: Dict[str, float] = dict(BUILTIN_PARAMS)
    declared: Dict[str, Optional[str]] = {}
    order: List[str] = []

    def note_species(name: str, line: int, col: int) -> None:
        if name in RESERVED:
            raise SemanticError(
                f"{name!r} is reserved and cannot name a species", line, col, source
            )
        if name not in declared:
            declared[name] = None
            order.append(name)

    # pass 1: parameters and species declarations ---------------------------- #
    for st in program.statements:
        if isinstance(st, ParamDef):
            if st.name in RESERVED:
                raise SemanticError(
                    f"{st.name!r} is a builtin and cannot be redefined", *st.pos, source
                )
            if st.name in params:
                raise SemanticError(f"parameter {st.name!r} is already defined", *st.pos, source)
            _check_symbols(st.expr, params, set(), source, allow_T=False, context="a parameter")
            folded = sym.substitute(st.expr, params)
            if not isinstance(folded, Num):
                raise SemanticError(
                    f"parameter {st.name!r} must be a constant expression", *st.pos, source
                )
            params[st.name] = folded.value
        elif isinstance(st, SpeciesDef):
            note_species(st.name, *st.pos)
            if st.formula is not None:
                parse_formula(st.formula, *st.pos)  # validate eagerly
                declared[st.name] = st.formula

    # pass 2: species appearing in reactions, in order of first appearance ---- #
    for st in program.statements:
        if isinstance(st, ReactionNode):
            for _coeff, name in list(st.reactants) + list(st.products):
                note_species(name, *st.pos)

    species = list(order)
    species_set = set(species)

    # pass 3: build each reaction -------------------------------------------- #
    reactions: List[ReactionIR] = []
    notes: List[str] = []
    for idx, st in enumerate(s for s in program.statements if isinstance(s, ReactionNode)):
        reactions.append(
            _build_reaction(idx, st, params, species_set, declared, source, strict_balance, notes)
        )

    formulas = {s: _formula_for(s, declared) for s in species}
    return Mechanism(species, reactions, params, formulas, source, notes)


def _formula_for(name: str, declared: Dict[str, Optional[str]]) -> Optional[str]:
    if declared.get(name):
        return declared[name]
    return name if looks_like_formula(name) else None


def _build_reaction(
    idx: int,
    st: ReactionNode,
    params: Dict[str, float],
    species_set: set,
    declared: Dict[str, Optional[str]],
    source: str,
    strict_balance: bool,
    notes: List[str],
) -> ReactionIR:
    line = st.pos[0]
    nu: Dict[str, float] = {}
    for coeff, name in st.reactants:
        nu[name] = nu.get(name, 0.0) - coeff
    for coeff, name in st.products:
        nu[name] = nu.get(name, 0.0) + coeff
    nu = {k: v for k, v in nu.items() if v != 0.0}

    clauses = st.clauses
    has_fwd = "rate" in clauses or "k" in clauses
    if not has_fwd:
        raise SemanticError(
            "reaction needs a rate law: add ', k: <expression>' for mass action "
            "or ', rate: <expression>' for an explicit rate",
            *st.pos,
            source,
        )
    if "rate" in clauses and "k" in clauses:
        raise SemanticError("give either k: or rate:, not both", *st.clause_pos["k"], source)
    if st.reversible and not ("krev" in clauses or "raterev" in clauses):
        raise SemanticError("a reversible reaction needs krev: or raterev:", *st.pos, source)
    for key in ("krev", "raterev"):
        if key in clauses and not st.reversible:
            raise SemanticError(
                f"{key}: given for an irreversible reaction; use '<=>' to make it reversible",
                *st.clause_pos[key],
                source,
            )

    forward, ma_fwd = _rate_from_clauses(
        clauses, "rate", "k", st.reactants, params, species_set, source, st
    )
    rate = forward
    ma_rev = None
    if st.reversible:
        reverse, ma_rev = _rate_from_clauses(
            clauses, "raterev", "krev", st.products, params, species_set, source, st
        )
        rate = sym.simplify(Binary("-", forward, reverse))

    dH = 0.0
    if "dh" in clauses:
        expr = clauses["dh"]
        _check_symbols(expr, params, species_set, source, allow_T=False, context="dH")
        folded = sym.substitute(expr, params)
        if not isinstance(folded, Num):
            raise SemanticError("dH must be a constant", *st.clause_pos["dh"], source)
        dH = folded.value

    label = clauses.get("name") if isinstance(clauses.get("name"), str) else None
    equation = _equation_text(st)
    checked = _check_balance(st, nu, declared, source, strict_balance, notes, equation)

    return ReactionIR(
        index=idx,
        nu=nu,
        rate=rate,
        dH=dH,
        label=label,
        reversible=st.reversible,
        line=line,
        equation=equation,
        ma_forward=ma_fwd,
        ma_reverse=ma_rev,
        balance_checked=checked,
    )


def _rate_from_clauses(clauses, rate_key: str, k_key: str, side, params, species_set, source, st):
    """Return (rate_expression, mass_action_provenance)."""
    if rate_key in clauses:
        expr = clauses[rate_key]
        _check_symbols(expr, params, species_set, source, allow_T=True, context=f"{rate_key}:")
        return sym.substitute(expr, params), None
    expr = clauses[k_key]
    _check_symbols(expr, params, species_set, source, allow_T=True, context=f"{k_key}:")
    k = sym.substitute(expr, params)
    orders: Dict[str, float] = {}
    for coeff, name in side:
        orders[name] = orders.get(name, 0.0) + coeff
    rate: Expr = k
    for name, order in orders.items():
        term: Expr = Conc(name) if order == 1.0 else Binary("^", Conc(name), Num(order))
        rate = Binary("*", rate, term)
    return sym.simplify(rate), (k, orders)


def _check_symbols(
    expr: Expr, params: Dict[str, float], species_set: set, source: str, allow_T: bool, context: str
) -> None:
    """Reject unknown names, unknown functions and wrong arities."""
    if isinstance(expr, Num):
        return
    if isinstance(expr, Var):
        if expr.name == "T":
            if not allow_T:
                raise SemanticError(f"temperature T cannot appear in {context}", *expr.pos, source)
            return
        if expr.name not in params:
            hint = ""
            if expr.name in species_set:
                hint = f"; write [{expr.name}] for its concentration"
            elif params:
                hint = "; known parameters: " + ", ".join(sorted(p for p in params if p != "R"))
            raise SemanticError(f"unknown name {expr.name!r}{hint}", *expr.pos, source)
        return
    if isinstance(expr, Conc):
        if expr.species not in species_set:
            raise SemanticError(
                f"unknown species [{expr.species}]; it appears in no reaction -- "
                f"declare it with 'species {expr.species}' if it is a catalyst or inert",
                *expr.pos,
                source,
            )
        return
    if isinstance(expr, Unary):
        _check_symbols(expr.operand, params, species_set, source, allow_T, context)
        return
    if isinstance(expr, Binary):
        _check_symbols(expr.left, params, species_set, source, allow_T, context)
        _check_symbols(expr.right, params, species_set, source, allow_T, context)
        return
    if isinstance(expr, Call):
        if expr.func not in FUNCTIONS:
            raise SemanticError(
                f"unknown function {expr.func!r}; available: " + ", ".join(sorted(FUNCTIONS)),
                *expr.pos,
                source,
            )
        want = FUNCTIONS[expr.func]
        if len(expr.args) != want:
            raise SemanticError(
                f"{expr.func}() takes {want} argument(s), got {len(expr.args)}",
                *expr.pos,
                source,
            )
        for a in expr.args:
            _check_symbols(a, params, species_set, source, allow_T, context)
        return
    raise TypeError(f"unexpected node {expr!r}")  # pragma: no cover


def _equation_text(st: ReactionNode) -> str:
    def side(terms: Sequence[Tuple[float, str]]) -> str:
        if not terms:
            return "0"
        parts = []
        for coeff, name in terms:
            parts.append(name if coeff == 1.0 else f"{coeff:g} {name}")
        return " + ".join(parts)

    arrow = "<=>" if st.reversible else "->"
    return f"{side(st.reactants)} {arrow} {side(st.products)}"


def _check_balance(
    st: ReactionNode,
    nu: Dict[str, float],
    declared: Dict[str, Optional[str]],
    source: str,
    strict: bool,
    notes: List[str],
    equation: str,
) -> bool:
    """Check element balance when every species in the reaction has a formula.

    Formulas come from ``species X = ...`` declarations, or from the species
    name itself when it is unambiguously a formula (contains a digit or more
    than one element symbol) -- so ``2 NO + O2 -> 2 NO2`` is checked while
    ``A -> B`` is skipped rather than silently read as boron.
    """
    terms = []
    for name, coeff in nu.items():
        formula = _formula_for(name, declared)
        if formula is None:
            if strict:
                raise BalanceError(
                    f"species {name!r} has no formula, so '{equation}' cannot be balance-checked; "
                    f"declare it with 'species {name} = <formula>'",
                    *st.pos,
                    source,
                )
            notes.append(f"'{equation}' not balance-checked ({name} has no formula)")
            return False
        terms.append((coeff, parse_formula(formula, *st.pos)))
    residual = balance_residual(terms)
    if residual:
        detail = ", ".join(f"{el}: {v:+g}" for el, v in sorted(residual.items()))
        raise BalanceError(f"'{equation}' does not balance ({detail})", *st.pos, source)
    return True


def analyze_source(src: str, strict_balance: bool = False) -> Mechanism:
    """Parse and analyse DSL source in one step."""
    return analyze(parse(src), src, strict_balance=strict_balance)
