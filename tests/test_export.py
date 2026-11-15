"""Arrhenius recognition and export to reactor_api.ReactionNetwork."""

import random

import numpy as np
import pytest
from mechanisms import MASS_ACTION_ONLY, random_mechanism

from rxnc import compile_mechanism, recognize_arrhenius
from rxnc.errors import SemanticError
from rxnc.parser import Parser
from rxnc.symbolic import substitute

R_GAS = 8.314462618

reactor_api = pytest.importorskip("reactor_api", reason="reactor_api not installed")


def k_expr(text):
    return substitute(Parser(text).parse_expr(), {"R": R_GAS})


def test_recognizes_plain_arrhenius():
    got = recognize_arrhenius(k_expr("1.2e9*exp(-72747/(R*T))"))
    assert got["A"] == pytest.approx(1.2e9, rel=1e-9)
    assert got["Ea"] == pytest.approx(72747.0, rel=1e-9)
    assert got["b"] == 0.0


def test_recognizes_modified_arrhenius():
    got = recognize_arrhenius(k_expr("1e9*(T/298.15)^0.5*exp(-111e3/(R*T))"))
    assert got["b"] == pytest.approx(0.5, rel=1e-9)
    assert got["Ea"] == pytest.approx(111e3, rel=1e-9)


def test_recognizes_an_algebraically_disguised_spelling():
    """Identification is numeric, so any equivalent spelling is accepted."""
    got = recognize_arrhenius(k_expr("exp(log(1e5) - 50e3/(R*T))"))
    assert got["A"] == pytest.approx(1e5, rel=1e-8)
    assert got["Ea"] == pytest.approx(50e3, rel=1e-8)


def test_constant_rate_coefficient():
    got = recognize_arrhenius(k_expr("2.5"))
    assert got == {"A": 2.5, "Ea": 0.0, "b": 0.0, "T_ref": 298.15}


@pytest.mark.parametrize("text", ["1e3 + T", "1e3*exp(-50e3/(R*T)) + 1", "1e3/(1 + exp(-T/20))"])
def test_rejects_non_arrhenius(text):
    assert recognize_arrhenius(k_expr(text)) is None


def test_rejects_rate_coefficients_that_depend_on_concentration():
    assert recognize_arrhenius(k_expr("1e3*[A]")) is None


@pytest.mark.parametrize("name", sorted(MASS_ACTION_ONLY))
def test_export_reproduces_reactor_api(name):
    mech = compile_mechanism(MASS_ACTION_ONLY[name])
    network = mech.to_reaction_network()
    assert network.species == mech.species
    rng = random.Random(3)
    for _ in range(5):
        C = np.array([rng.uniform(1.0, 500.0) for _ in mech.species])
        T = rng.uniform(300.0, 700.0)
        np.testing.assert_allclose(mech.rates(C, T), network.rates(C, T), rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            mech.production(C, T), network.production(C, T), rtol=1e-12, atol=0
        )
        assert mech.heat_release(C, T) == pytest.approx(network.heat_release(C, T), rel=1e-12)


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_export_reproduces_reactor_api_on_random_mechanisms(seed):
    mech = compile_mechanism(random_mechanism(seed))
    network = mech.to_reaction_network()
    rng = random.Random(seed)
    for _ in range(4):
        C = np.array([rng.uniform(1.0, 500.0) for _ in mech.species])
        T = rng.uniform(300.0, 700.0)
        np.testing.assert_allclose(mech.rates(C, T), network.rates(C, T), rtol=1e-12, atol=0)


def test_catalytic_orders_survive_the_round_trip():
    """K cancels from the stoichiometry but must stay in the rate law."""
    mech = compile_mechanism("A + K -> B + K, k: 2.0e3*exp(-30e3/(R*T))\n")
    network = mech.to_reaction_network()
    C = np.array([100.0, 5.0, 0.0])
    np.testing.assert_allclose(mech.rates(C, 400.0), network.rates(C, 400.0), rtol=1e-12)
    np.testing.assert_allclose(mech.production(C, 400.0), network.production(C, 400.0), rtol=1e-12)


def test_explicit_rate_law_refuses_to_export():
    mech = compile_mechanism("A + B -> C, rate: 3.0*[A]*[B]/(1 + 0.5*[C])\n")
    with pytest.raises(SemanticError, match="cannot represent"):
        mech.to_reaction_network()


def test_non_arrhenius_coefficient_refuses_to_export():
    mech = compile_mechanism("A -> B, k: 1e3/(1 + exp(-(T - 400)/20))\n")
    with pytest.raises(SemanticError, match="is not of the form"):
        mech.to_reaction_network()
