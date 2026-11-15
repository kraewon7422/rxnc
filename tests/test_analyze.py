"""Semantic analysis: name resolution, mass-action expansion, balance checks."""

import pytest

from rxnc import analyze_source, unparse
from rxnc.errors import BalanceError, SemanticError


def test_species_order_follows_first_appearance():
    mech = analyze_source("B -> C, k: 1\nA -> B, k: 1\n")
    assert mech.species == ["B", "C", "A"]


def test_declared_species_come_first_and_keep_formulas():
    mech = analyze_source("species W = H2O\nA -> B, k: 1\n")
    assert mech.species == ["W", "A", "B"]
    assert mech.formulas["W"] == "H2O"


def test_mass_action_expansion_uses_reactant_coefficients():
    mech = analyze_source("2 NO + O2 -> 2 NO2, k: 3.0\n")
    assert unparse(mech.reactions[0].rate) == "3*[NO]^2*[O2]"
    assert mech.reactions[0].nu == {"NO": -2.0, "O2": -1.0, "NO2": 2.0}


def test_catalyst_cancels_in_stoichiometry_but_not_in_the_rate():
    mech = analyze_source("A + K -> K + B, k: 2.0\n")
    rxn = mech.reactions[0]
    assert rxn.nu == {"A": -1.0, "B": 1.0}  # K cancels
    assert "[K]" in unparse(rxn.rate)  # but still drives the rate


def test_parameters_are_folded_to_numbers():
    mech = analyze_source("let A0 = 2e3\nlet Ea = 50e3\nA -> B, k: A0*exp(-Ea/(R*T))\n")
    text = unparse(mech.reactions[0].rate)
    assert "A0" not in text and "Ea" not in text and text.startswith("2000*exp(")


def test_explicit_rate_law_is_kept_verbatim():
    mech = analyze_source("A + B -> C, rate: 3.0*[A]*[B]/(1 + 0.5*[C])\n")
    assert mech.reactions[0].ma_forward is None
    assert unparse(mech.reactions[0].rate) == "3*[A]*[B]/(1 + 0.5*[C])"


def test_reversible_rate_is_forward_minus_reverse():
    mech = analyze_source("A <=> B, k: 2.0, krev: 0.5\n")
    assert unparse(mech.reactions[0].rate) == "2*[A] - 0.5*[B]"


def test_dH_defaults_to_zero_and_accepts_expressions():
    mech = analyze_source("let q = 25e3\nA -> B, k: 1\nB -> C, k: 1, dH: -2*q\n")
    assert mech.reactions[0].dH == 0.0
    assert mech.reactions[1].dH == -50e3


# -- diagnostics ------------------------------------------------------------- #
def test_unknown_parameter_is_reported_with_position():
    with pytest.raises(SemanticError) as err:
        analyze_source("A -> B, k: kk*exp(-1/T)\n")
    assert "unknown name 'kk'" in err.value.message
    assert err.value.line == 1


def test_bare_species_name_suggests_bracket_syntax():
    with pytest.raises(SemanticError, match=r"write \[A\] for its concentration"):
        analyze_source("A -> B, rate: 2.0*A\n")


def test_unknown_species_in_rate_is_reported():
    with pytest.raises(SemanticError, match=r"unknown species \[Z\]"):
        analyze_source("A -> B, rate: 2.0*[A]*[Z]\n")


def test_missing_rate_law_is_reported():
    with pytest.raises(SemanticError, match="needs a rate law"):
        analyze_source("A -> B\n")


def test_both_k_and_rate_is_rejected():
    with pytest.raises(SemanticError, match="not both"):
        analyze_source("A -> B, k: 1, rate: 2*[A]\n")


def test_reverse_coefficient_requires_a_reversible_arrow():
    with pytest.raises(SemanticError, match="use '<=>'"):
        analyze_source("A -> B, k: 1, krev: 1\n")


def test_reversible_without_reverse_coefficient_is_rejected():
    with pytest.raises(SemanticError, match="needs krev"):
        analyze_source("A <=> B, k: 1\n")


def test_temperature_cannot_appear_in_a_parameter():
    with pytest.raises(SemanticError, match="temperature T cannot appear"):
        analyze_source("let bad = 2*T\nA -> B, k: 1\n")


def test_builtin_names_are_protected():
    with pytest.raises(SemanticError, match="cannot be redefined"):
        analyze_source("let R = 1.0\nA -> B, k: 1\n")
    with pytest.raises(SemanticError, match="reserved"):
        analyze_source("T -> B, k: 1\n")


def test_unknown_function_lists_the_available_ones():
    with pytest.raises(SemanticError, match="unknown function 'tanh'"):
        analyze_source("A -> B, rate: tanh([A])\n")


def test_wrong_arity_is_reported():
    with pytest.raises(SemanticError, match=r"exp\(\) takes 1 argument"):
        analyze_source("A -> B, rate: exp([A], 2)\n")


# -- element balance --------------------------------------------------------- #
def test_balanced_reaction_passes():
    mech = analyze_source("2 NO + O2 -> 2 NO2, k: 1\n")
    assert mech.reactions[0].balance_checked
    assert mech.unbalanced_note == []


def test_unbalanced_reaction_is_rejected():
    with pytest.raises(BalanceError) as err:
        analyze_source("H2 + O2 -> H2O, k: 1\n")
    assert "does not balance" in err.value.message and "O: " in err.value.message


def test_placeholder_species_are_skipped_not_guessed():
    mech = analyze_source("A -> B, k: 1\n")  # B is boron only to a careless reader
    assert not mech.reactions[0].balance_checked
    assert "not balance-checked" in mech.unbalanced_note[0]


def test_strict_balance_demands_formulas():
    with pytest.raises(BalanceError, match="declare it with"):
        analyze_source("A -> B, k: 1\n", strict_balance=True)


def test_declared_formulas_enable_checking_of_placeholder_names():
    mech = analyze_source(
        "species A = CH4, B = CH3, H = H\nA -> B + H, k: 1\n", strict_balance=True
    )
    assert mech.reactions[0].balance_checked
