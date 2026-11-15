"""Mechanism sources shared by the backend tests."""

import random

SIMPLE = """
A -> B, k: 1.2e9*exp(-72747/(R*T)), dH: -50e3
"""

SERIES = """
A -> B, k: 1.0e8*exp(-60e3/(R*T)), dH: -40e3
B -> C, k: 5.0e7*exp(-70e3/(R*T)), dH: -25e3
"""

REVERSIBLE = """
let Ea_f = 45e3
A + B <=> C, k: 2.0e6*exp(-Ea_f/(R*T)), krev: 8.0e9*exp(-95e3/(R*T)), dH: -60e3
"""

NONLINEAR = """
# Langmuir-Hinshelwood surface rate: not expressible as mass action
species S
A + B -> C, rate: 4.0e5*exp(-25e3/(R*T))*[A]*[B]/(1 + 0.02*[C] + 0.5*[S])^2, dH: -80e3
C -> A + B, k: 1.0e3*exp(-40e3/(R*T)), dH: 80e3
"""

GAS_PHASE = """
species NO, O2, NO2 = NO2, N2
2 NO + O2 -> 2 NO2, k: 1.3e4*exp(1050/(R*T)), dH: -114e3
NO2 + NO2 -> 2 NO + O2, k: 1.0e9*(T/298.15)^0.5*exp(-111e3/(R*T)), dH: 114e3
"""

FRACTIONAL = """
A + 2 B -> C, rate: 1.0e4*exp(-30e3/(R*T))*[A]^0.5*[B]^1.5, dH: -20e3
"""

ALL = {
    "simple": SIMPLE,
    "series": SERIES,
    "reversible": REVERSIBLE,
    "nonlinear": NONLINEAR,
    "gas_phase": GAS_PHASE,
    "fractional": FRACTIONAL,
}

MASS_ACTION_ONLY = {k: ALL[k] for k in ("simple", "series", "reversible", "gas_phase")}


def random_mechanism(seed: int, n_species: int = 5, n_reactions: int = 6) -> str:
    """Generate a random mass-action mechanism with integer orders."""
    rng = random.Random(seed)
    names = [f"S{i}" for i in range(n_species)]
    lines = []
    for _ in range(n_reactions):
        reactants = rng.sample(names, rng.randint(1, 2))
        products = rng.sample([n for n in names if n not in reactants], rng.randint(1, 2))
        left = " + ".join(
            (name if (c := rng.choice([1, 1, 2])) == 1 else f"{c} {name}") for name in reactants
        )
        right = " + ".join(products)
        A = rng.uniform(1e3, 1e9)
        Ea = rng.uniform(10e3, 90e3)
        dH = rng.uniform(-100e3, 100e3)
        reversible = rng.random() < 0.3
        if reversible:
            lines.append(
                f"{left} <=> {right}, k: {A:.6g}*exp(-{Ea:.6g}/(R*T)), "
                f"krev: {A * 10:.6g}*exp(-{Ea + 20e3:.6g}/(R*T)), dH: {dH:.6g}"
            )
        else:
            lines.append(f"{left} -> {right}, k: {A:.6g}*exp(-{Ea:.6g}/(R*T)), dH: {dH:.6g}")
    return "\n".join(lines) + "\n"
