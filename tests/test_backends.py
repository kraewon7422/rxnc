"""Differential tests: generated code vs the reference interpreter, and
analytic Jacobians vs finite differences."""

import random

import numpy as np
import pytest
from mechanisms import ALL, random_mechanism

from rxnc import compile_mechanism
from rxnc.analyze import analyze_source
from rxnc.codegen import generate_standalone


def states(mech, seed, n=6, lo=10.0, hi=800.0):
    rng = random.Random(seed)
    for _ in range(n):
        C = np.array([rng.uniform(lo, hi) for _ in mech.species])
        T = rng.uniform(280.0, 750.0)
        yield C, T


@pytest.mark.parametrize("name", sorted(ALL))
def test_generated_code_matches_the_interpreter(name):
    fast = compile_mechanism(ALL[name], backend="python")
    slow = compile_mechanism(ALL[name], backend="interp")
    for C, T in states(fast, seed=hash(name) % 1000):
        np.testing.assert_allclose(fast.rates(C, T), slow.rates(C, T), rtol=1e-13, atol=0)
        np.testing.assert_allclose(
            fast.rate_jacobian(C, T), slow.rate_jacobian(C, T), rtol=1e-13, atol=0
        )
        np.testing.assert_allclose(fast.drate_dT(C, T), slow.drate_dT(C, T), rtol=1e-13, atol=0)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_mechanisms_agree_across_backends(seed):
    src = random_mechanism(seed)
    fast = compile_mechanism(src, backend="python")
    slow = compile_mechanism(src, backend="interp")
    for C, T in states(fast, seed=seed, n=4):
        np.testing.assert_allclose(fast.rates(C, T), slow.rates(C, T), rtol=1e-13, atol=0)
        np.testing.assert_allclose(
            fast.production_jacobian(C, T), slow.production_jacobian(C, T), rtol=1e-13, atol=0
        )


@pytest.mark.parametrize("name", sorted(ALL))
def test_production_jacobian_matches_finite_differences(name):
    mech = compile_mechanism(ALL[name])
    for C, T in states(mech, seed=hash(name) % 997, n=3, lo=50.0, hi=400.0):
        analytic = mech.production_jacobian(C, T)
        numeric = np.zeros_like(analytic)
        for col in range(mech.n_species):
            h = 1e-6 * max(C[col], 1.0)
            up, dn = C.copy(), C.copy()
            up[col] += h
            dn[col] -= h
            numeric[:, col] = (mech.production(up, T) - mech.production(dn, T)) / (2 * h)
        scale = max(np.abs(numeric).max(), 1e-12)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=1e-6 * scale)


@pytest.mark.parametrize("name", sorted(ALL))
def test_temperature_derivatives_match_finite_differences(name):
    mech = compile_mechanism(ALL[name])
    for C, T in states(mech, seed=hash(name) % 991, n=3, lo=50.0, hi=400.0):
        h = 1e-4 * T
        numeric = (mech.production(C, T + h) - mech.production(C, T - h)) / (2 * h)
        scale = max(np.abs(numeric).max(), 1e-12)
        np.testing.assert_allclose(mech.dproduction_dT(C, T), numeric, rtol=2e-5, atol=1e-6 * scale)
        numeric_q = (mech.heat_release(C, T + h) - mech.heat_release(C, T - h)) / (2 * h)
        assert mech.dheat_dT(C, T) == pytest.approx(numeric_q, rel=2e-5, abs=1e-6 * abs(numeric_q))


def test_full_jacobian_assembles_the_same_blocks():
    mech = compile_mechanism(ALL["gas_phase"])
    C, T = next(iter(states(mech, seed=11, n=1)))
    full = mech.full_jacobian(C, T)
    n = mech.n_species
    np.testing.assert_allclose(full[:n, :n], mech.production_jacobian(C, T), rtol=1e-14)
    np.testing.assert_allclose(full[:n, n], mech.dproduction_dT(C, T), rtol=1e-14)
    np.testing.assert_allclose(full[n, :n], -mech.dH @ mech.rate_jacobian(C, T), rtol=1e-14)
    assert full[n, n] == pytest.approx(mech.dheat_dT(C, T), rel=1e-14)


def test_jacobian_sparsity_skips_structural_zeros():
    mech = compile_mechanism(ALL["series"])  # A -> B -> C
    jac = mech.rate_jacobian([1.0, 1.0, 1.0], 400.0)
    assert jac[0, 1] == 0.0 and jac[1, 0] == 0.0  # each rate sees one species
    assert mech.stats["jacobian_entries"] == 2  # and only those two are emitted


def test_common_subexpressions_are_shared():
    mech = compile_mechanism(ALL["gas_phase"])
    assert mech.stats["reused_subexpressions"] > 0
    assert mech.source.count("exp(") < 2 * (mech.n_reactions + mech.stats["jacobian_entries"])


def test_rates_are_zero_without_reactants():
    mech = compile_mechanism(ALL["series"])
    np.testing.assert_allclose(mech.rates([0.0, 0.0, 0.0], 500.0), [0.0, 0.0])


def test_standalone_module_is_self_contained(tmp_path):
    src = generate_standalone(analyze_source(ALL["reversible"]), origin="reversible.rxn")
    path = tmp_path / "generated_kernel.py"
    path.write_text(src, encoding="utf-8")
    namespace = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    mech = compile_mechanism(ALL["reversible"])
    assert namespace["SPECIES"] == mech.species
    assert namespace["NU"] == mech.nu.tolist()
    C, T = [300.0, 200.0, 100.0], 420.0
    rates, _, _ = namespace["kernel"](C, T)
    np.testing.assert_allclose(np.array(rates), mech.rates(C, T), rtol=1e-14)


def test_backend_name_is_validated():
    with pytest.raises(ValueError, match="unknown backend"):
        compile_mechanism(ALL["simple"], backend="fortran")


def test_vector_rejects_unknown_species():
    mech = compile_mechanism(ALL["simple"])
    assert list(mech.vector({"B": 2.0})) == [0.0, 2.0]
    with pytest.raises(KeyError):
        mech.vector({"Q": 1.0})
