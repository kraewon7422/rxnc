"""The compiled artefact: rates, Jacobians and the ReactionNetwork adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from . import codegen
from .analyze import Mechanism, analyze_source
from .errors import RxncError, SemanticError
from .interp import InterpreterKernel
from .nodes import unparse
from .recognize import recognize_arrhenius

BACKENDS = ("python", "interp")


class CompiledMechanism:
    """Evaluates a mechanism's rates and analytic derivatives.

    ``backend="python"`` generates and executes straight-line code;
    ``backend="interp"`` walks the expression trees and serves as the
    reference implementation.
    """

    def __init__(self, mech: Mechanism, backend: str = "python"):
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
        self.mech = mech
        self.backend = backend
        self.species = list(mech.species)
        self.nu = np.array(mech.nu_matrix(), dtype=float).reshape(mech.n_reactions, mech.n_species)
        self.dH = np.array([r.dH for r in mech.reactions], dtype=float)
        self.source: Optional[str] = None
        self.stats: Dict[str, int] = {}
        if backend == "python":
            src, dC_pattern, dT_pattern, stats = codegen.generate(mech)
            self.source = src
            self.stats = stats
            self._kernel = codegen.load(src)
            self._dC_pattern = dC_pattern
            self._dT_pattern = dT_pattern
        else:
            kernel = InterpreterKernel(mech)
            self._kernel = kernel
            self._dC_pattern = kernel.dC_pattern
            self._dT_pattern = kernel.dT_pattern

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_source(
        cls, src: str, backend: str = "python", strict_balance: bool = False
    ) -> CompiledMechanism:
        return cls(analyze_source(src, strict_balance=strict_balance), backend)

    @classmethod
    def from_file(
        cls, path: str | Path, backend: str = "python", strict_balance: bool = False
    ) -> CompiledMechanism:
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_source(text, backend, strict_balance)

    # -- sizes -------------------------------------------------------------- #
    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        return self.nu.shape[0]

    def index(self, name: str) -> int:
        return self.species.index(name)

    def vector(self, mapping: Dict[str, float], default: float = 0.0) -> np.ndarray:
        """Turn ``{species: value}`` into an array ordered like ``self.species``."""
        unknown = set(mapping) - set(self.species)
        if unknown:
            raise KeyError(f"unknown species: {sorted(unknown)}")
        return np.array([mapping.get(s, default) for s in self.species], dtype=float)

    # -- evaluation --------------------------------------------------------- #
    def _call(self, C: Sequence[float], T: float):
        return self._kernel(C, float(T))

    def rates(self, C: Sequence[float], T: float) -> np.ndarray:
        """Net rate of every reaction, mol/(m^3 s)."""
        rates, _, _ = self._call(C, T)
        return np.asarray(rates, dtype=float)

    def production(self, C: Sequence[float], T: float) -> np.ndarray:
        """Net formation rate of every species, mol/(m^3 s)."""
        return self.nu.T @ self.rates(C, T)

    def heat_release(self, C: Sequence[float], T: float) -> float:
        """Volumetric heat release, W/m^3 (positive heats the fluid)."""
        return float(-self.dH @ self.rates(C, T))

    def rate_jacobian(self, C: Sequence[float], T: float) -> np.ndarray:
        """d(rate_j)/d(C_i), shape (n_reactions, n_species)."""
        _, dC, _ = self._call(C, T)
        out = np.zeros((self.n_reactions, self.n_species))
        for (j, i), value in zip(self._dC_pattern, dC, strict=True):
            out[j, i] = value
        return out

    def production_jacobian(self, C: Sequence[float], T: float) -> np.ndarray:
        """d(production_i)/d(C_l), shape (n_species, n_species)."""
        return self.nu.T @ self.rate_jacobian(C, T)

    def drate_dT(self, C: Sequence[float], T: float) -> np.ndarray:
        """d(rate_j)/dT, shape (n_reactions,)."""
        _, _, dT = self._call(C, T)
        out = np.zeros(self.n_reactions)
        for j, value in zip(self._dT_pattern, dT, strict=True):
            out[j] = value
        return out

    def dproduction_dT(self, C: Sequence[float], T: float) -> np.ndarray:
        """d(production_i)/dT, shape (n_species,)."""
        return self.nu.T @ self.drate_dT(C, T)

    def dheat_dT(self, C: Sequence[float], T: float) -> float:
        """d(heat release)/dT, W/(m^3 K)."""
        return float(-self.dH @ self.drate_dT(C, T))

    def full_jacobian(self, C: Sequence[float], T: float) -> np.ndarray:
        """Jacobian of ``[production, heat_release]`` w.r.t. ``[C, T]``.

        Shape ``(n_species + 1, n_species + 1)`` -- the block a reactor energy
        balance needs, assembled from the same kernel call.
        """
        rates, dC, dT = self._call(C, T)
        drdC = np.zeros((self.n_reactions, self.n_species))
        for (j, i), value in zip(self._dC_pattern, dC, strict=True):
            drdC[j, i] = value
        drdT = np.zeros(self.n_reactions)
        for j, value in zip(self._dT_pattern, dT, strict=True):
            drdT[j] = value
        out = np.zeros((self.n_species + 1, self.n_species + 1))
        out[: self.n_species, : self.n_species] = self.nu.T @ drdC
        out[: self.n_species, self.n_species] = self.nu.T @ drdT
        out[self.n_species, : self.n_species] = -self.dH @ drdC
        out[self.n_species, self.n_species] = -self.dH @ drdT
        return out

    # -- reporting ---------------------------------------------------------- #
    def listing(self) -> str:
        """Human-readable dump of the analysed mechanism."""
        return self.mech.summary()

    def jacobian_report(self) -> str:
        lines = []
        for (j, i), expr in sorted(self.mech.drate_dC().items()):
            lines.append(
                f"d(rate[{self.mech.reactions[j].name}])/d[{self.species[i]}] = {unparse(expr)}"
            )
        for j, expr in sorted(self.mech.drate_dT().items()):
            lines.append(f"d(rate[{self.mech.reactions[j].name}])/dT = {unparse(expr)}")
        return "\n".join(lines)

    # -- export ------------------------------------------------------------- #
    def to_reaction_network(self, T_ref: float = 298.15):
        """Export to ``reactor_api.kinetics.ReactionNetwork``.

        Only mass-action reactions whose ``k:`` is a modified-Arrhenius
        expression can be exported; anything else raises, naming the reaction,
        rather than approximating it.
        """
        try:
            from reactor_api.kinetics import Reaction, build_network
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RxncError(
                "reactor_api is not installed; `pip install reactor-api` or point "
                "PYTHONPATH at a checkout to use to_reaction_network()"
            ) from exc

        reactions = []
        for rxn in self.mech.reactions:
            if rxn.ma_forward is None:
                raise SemanticError(
                    f"reaction {rxn.name} ({rxn.equation}) was written with an explicit "
                    "rate: law, which ReactionNetwork cannot represent; use the rxnc "
                    "kernel directly for this mechanism",
                    rxn.line,
                    1,
                    self.mech.source,
                )
            k_expr, orders = rxn.ma_forward
            fwd = recognize_arrhenius(k_expr, T_ref)
            if fwd is None:
                raise SemanticError(
                    f"reaction {rxn.name}: k = {unparse(k_expr)} is not of the form "
                    "A (T/T_ref)^b exp(-Ea/(R T)), so it cannot be exported",
                    rxn.line,
                    1,
                    self.mech.source,
                )
            kwargs = dict(
                stoichiometry=dict(rxn.nu),
                A=fwd["A"],
                Ea=fwd["Ea"],
                b=fwd["b"],
                T_ref=T_ref,
                orders=dict(orders),
                dH=rxn.dH,
                name=rxn.name,
            )
            if rxn.reversible:
                if rxn.ma_reverse is None:
                    raise SemanticError(
                        f"reaction {rxn.name}: the reverse step uses raterev:, which "
                        "ReactionNetwork cannot represent",
                        rxn.line,
                        1,
                        self.mech.source,
                    )
                k_rev, orders_rev = rxn.ma_reverse
                rev = recognize_arrhenius(k_rev, T_ref)
                if rev is None:
                    raise SemanticError(
                        f"reaction {rxn.name}: krev = {unparse(k_rev)} is not "
                        "modified-Arrhenius, so it cannot be exported",
                        rxn.line,
                        1,
                        self.mech.source,
                    )
                kwargs.update(
                    A_rev=rev["A"],
                    Ea_rev=rev["Ea"],
                    b_rev=rev["b"],
                    orders_rev=dict(orders_rev),
                )
            reactions.append(Reaction(**kwargs))
        return build_network(self.species, reactions)


def compile_mechanism(
    src: str, backend: str = "python", strict_balance: bool = False
) -> CompiledMechanism:
    """Compile DSL source into a :class:`CompiledMechanism`."""
    return CompiledMechanism.from_source(src, backend, strict_balance)


def compile_file(
    path: str | Path, backend: str = "python", strict_balance: bool = False
) -> CompiledMechanism:
    """Compile a ``.rxn`` file into a :class:`CompiledMechanism`."""
    return CompiledMechanism.from_file(path, backend, strict_balance)
