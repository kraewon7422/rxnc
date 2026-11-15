"""Simplification and symbolic differentiation."""

import math
import random

import pytest

from rxnc.interp import evaluate
from rxnc.nodes import Num, unparse
from rxnc.parser import Parser
from rxnc.symbolic import depends_on, diff, is_num, simplify, species_used, substitute

EXPRESSIONS = [
    "1.2e9*exp(-72747/(8.314*T))*[A]^2",
    "[A]*[B]/(1 + 0.02*[C])",
    "sqrt([A])*log([B]) + T^0.7",
    "exp(-1000/T)*[A]^1.5*[B]^0.5",
    "1/(1 + exp(-(T - 400)/20))*[A]",
    "log10([A] + 1)*[B]",
    "([A] - [B])^3",
]


def parse_expr(text):
    return Parser(text).parse_expr()


def numeric_derivative(expr, conc, T, wrt):
    if wrt == "T":
        h = 1e-5 * T
        return (evaluate(expr, conc, T + h) - evaluate(expr, conc, T - h)) / (2 * h)
    species = wrt[1:-1]
    h = 1e-6 * max(conc[species], 1.0)
    up = dict(conc, **{species: conc[species] + h})
    dn = dict(conc, **{species: conc[species] - h})
    return (evaluate(expr, up, T) - evaluate(expr, dn, T)) / (2 * h)


@pytest.mark.parametrize("text", EXPRESSIONS)
def test_derivatives_match_finite_differences(text):
    expr = parse_expr(text)
    rng = random.Random(7)
    for _ in range(8):
        conc = {s: rng.uniform(10.0, 500.0) for s in ("A", "B", "C")}
        T = rng.uniform(300.0, 700.0)
        for wrt in ["T"] + [f"[{s}]" for s in sorted(species_used(expr))]:
            analytic = evaluate(diff(expr, wrt), conc, T)
            numeric = numeric_derivative(expr, conc, T, wrt)
            scale = max(abs(numeric), abs(analytic), 1e-12)
            assert analytic == pytest.approx(numeric, rel=2e-5, abs=1e-8 * scale)


@pytest.mark.parametrize(
    "text, want",
    [
        ("[A] + 0", "[A]"),
        ("0 + [A]", "[A]"),
        ("1*[A]", "[A]"),
        ("[A]*0", "0"),
        ("[A]^1", "[A]"),
        ("[A]^0", "1"),
        ("[A]/[A]", "1"),
        ("[A] - [A]", "0"),
        ("2*3*[A]", "6*[A]"),
        ("exp(0)*[A]", "[A]"),
    ],
)
def test_simplify_identities(text, want):
    assert unparse(simplify(parse_expr(text))) == want


def test_constant_folding_of_parameters():
    expr = substitute(parse_expr("A1*exp(-Ea/(R*T))"), {"A1": 2.0, "Ea": 100.0, "R": 8.0})
    assert unparse(expr) == "2*exp(-12.5/T)"


def test_zero_derivative_detected():
    expr = parse_expr("1e3*[A]")
    assert is_num(diff(expr, "T"), 0.0)
    assert is_num(diff(expr, "[B]"), 0.0)
    assert not depends_on(expr, "T")


def test_power_rule_is_exact():
    d = diff(parse_expr("[A]^3"), "[A]")
    assert evaluate(d, {"A": 2.0}, 300.0) == pytest.approx(12.0)


def test_non_differentiable_function_is_reported():
    from rxnc.errors import SemanticError

    with pytest.raises(SemanticError, match="not differentiable"):
        diff(parse_expr("max([A], 1)"), "[A]")


def test_simplify_folds_math_calls():
    assert simplify(parse_expr("exp(1)")) == Num(math.exp(1.0))
