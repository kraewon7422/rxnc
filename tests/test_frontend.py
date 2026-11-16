"""Lexer, parser and formula tests."""

import pytest

from rxnc import parse, unparse
from rxnc.errors import SemanticError, SyntaxErrorRxnc
from rxnc.formula import balance_residual, looks_like_formula, parse_formula
from rxnc.lexer import tokenize
from rxnc.nodes import ReactionNode
from rxnc.parser import Parser


def expr(text):
    return Parser(text).parse_expr()


CONC = {"A": 2.0, "B": 3.0, "C": 5.0}


def value(text):
    from rxnc.interp import evaluate

    return evaluate(expr(text), CONC, 300.0)


# -- lexer ------------------------------------------------------------------ #
def test_tokenize_numbers_and_comments():
    toks = [t for t in tokenize("1.2e9 + .5  # comment\n") if t.kind not in ("NEWLINE", "EOF")]
    assert [t.kind for t in toks] == ["NUMBER", "OP", "NUMBER"]
    assert toks[0].value == 1.2e9 and toks[2].value == 0.5


def test_tokenize_reports_position():
    with pytest.raises(SyntaxErrorRxnc) as err:
        tokenize("A -> B\nA ->? B\n")
    assert err.value.line == 2 and err.value.col == 5


def test_unterminated_string():
    with pytest.raises(SyntaxErrorRxnc):
        tokenize('A -> B, name: "oops\n')


# -- expression grammar ------------------------------------------------------ #
@pytest.mark.parametrize(
    "text, want",
    [
        ("2 + 3 * 4", 14.0),
        ("(2 + 3) * 4", 20.0),
        ("2 ^ 3 ^ 2", 512.0),  # right associative
        ("-2 ^ 2", -4.0),  # unary minus binds looser than ^
        ("(-2) ^ 2", 4.0),
        ("8 / 4 / 2", 1.0),  # left associative
        ("exp(0) + sqrt(9)", 4.0),
        ("--3", 3.0),
        ("2 * T", 600.0),
    ],
)
def test_precedence(text, want):
    assert value(text) == pytest.approx(want)


def test_unparse_round_trip():
    for text in ["2 + 3*4", "(2 + 3)*4", "2^3^2", "-2^2", "exp(-1.0/(8.0*T))*[A]^2"]:
        once = unparse(expr(text))
        assert value(once) == pytest.approx(value(text), rel=1e-12, abs=1e-300)
        assert unparse(expr(once)) == once


def test_parse_error_message_points_at_token():
    with pytest.raises(SyntaxErrorRxnc) as err:
        parse("A + -> B, k: 1\n")
    assert "expected a species name" in err.value.message
    assert "^" in err.value.render()


# -- reaction syntax --------------------------------------------------------- #
def test_reaction_sides_and_clauses():
    prog = parse('2 NO + O2 -> 2 NO2, k: 1.3e4, dH: -114e3, name: "oxidation"\n')
    rxn = prog.statements[0]
    assert isinstance(rxn, ReactionNode)
    assert rxn.reactants == [(2.0, "NO"), (1.0, "O2")]
    assert rxn.products == [(2.0, "NO2")]
    assert rxn.reversible is False
    assert rxn.clauses["name"] == "oxidation"


def test_empty_side_and_reversible():
    prog = parse("0 -> A, k: 1\nA <=> B, k: 1, krev: 2\n")
    assert prog.statements[0].reactants == []
    assert prog.statements[1].reversible is True


def test_unknown_clause_is_rejected():
    with pytest.raises(SyntaxErrorRxnc) as err:
        parse("A -> B, kk: 1\n")
    assert "unknown clause" in err.value.message


def test_duplicate_clause_is_rejected():
    with pytest.raises(SyntaxErrorRxnc):
        parse("A -> B, k: 1, k: 2\n")


# -- formulas ---------------------------------------------------------------- #
def test_parse_formula_nested():
    assert parse_formula("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}
    assert parse_formula("C6H12O6") == {"C": 6, "H": 12, "O": 6}


def test_parse_formula_rejects_unknown_element():
    with pytest.raises(SemanticError):
        parse_formula("Xx2")


def test_balance_residual():
    terms = [(-2.0, parse_formula("NO")), (-1.0, parse_formula("O2")), (2.0, parse_formula("NO2"))]
    assert balance_residual(terms) == {}


def test_looks_like_formula_is_conservative():
    assert looks_like_formula("NO2")
    assert looks_like_formula("H2O")
    assert not looks_like_formula("A")  # a bare placeholder is not a formula
    assert not looks_like_formula("B")  # even though B is boron
    assert looks_like_formula("S4")  # but S4 genuinely is tetrasulfur
    assert not looks_like_formula("S0")  # a zero subscript means no atoms
    assert not looks_like_formula("X0")  # X is not an element symbol
