"""Backend comparison: tree-walking interpreter vs generated code.

Run with ``python examples/benchmark.py``.  Reports the cost of one rate
evaluation and of one rate-plus-Jacobian evaluation for each backend, and --
when ``reactor_api`` is importable -- the cost of the equivalent NumPy
ReactionNetwork, which has no analytic Jacobian at all.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rxnc import compile_mechanism  # noqa: E402

MECHANISM = """
let Ea1 = 60e3
let Ea2 = 70e3
let Ea3 = 45e3

A -> B,       k: 1.0e8*exp(-Ea1/(R*T)), dH: -40e3
B -> C,       k: 5.0e7*exp(-Ea2/(R*T)), dH: -25e3
A + B <=> D,  k: 2.0e6*exp(-Ea3/(R*T)), krev: 8.0e9*exp(-95e3/(R*T)), dH: -60e3
C + D -> E,   k: 3.0e5*exp(-30e3/(R*T)), dH: -15e3
A + E -> 2 C, k: 7.0e6*exp(-55e3/(R*T)), dH: -35e3
"""


def timeit(fn, n: int) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n


def main(n: int = 20000) -> None:
    fast = compile_mechanism(MECHANISM, backend="python")
    slow = compile_mechanism(MECHANISM, backend="interp")
    C = np.array([500.0, 300.0, 120.0, 60.0, 25.0])
    T = 430.0

    rows = [
        ("interp  rates", timeit(lambda: slow.rates(C, T), n)),
        ("codegen rates", timeit(lambda: fast.rates(C, T), n)),
        ("interp  rates+Jacobian", timeit(lambda: slow.full_jacobian(C, T), n)),
        ("codegen rates+Jacobian", timeit(lambda: fast.full_jacobian(C, T), n)),
    ]
    try:
        network = fast.to_reaction_network()
        rows.append(("reactor_api rates (NumPy)", timeit(lambda: network.rates(C, T), n)))
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"(reactor_api comparison skipped: {exc})")

    width = max(len(label) for label, _ in rows)
    print(f"{fast.n_reactions} reactions, {fast.n_species} species, {n} repeats")
    print(f"generated code: {fast.stats}")
    for label, seconds in rows:
        print(f"  {label:<{width}}  {seconds * 1e6:8.2f} us")


if __name__ == "__main__":
    main()
