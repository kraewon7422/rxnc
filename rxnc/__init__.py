"""rxnc -- a compiler from a reaction-mechanism DSL to fast rate kernels.

Typical use::

    from rxnc import compile_mechanism

    mech = compile_mechanism("A -> B, k: 1.2e9*exp(-72747/(R*T)), dH: -50e3")
    mech.rates([1000.0], 350.0)
    mech.production_jacobian([1000.0], 350.0)   # analytic, not finite-difference
"""

from .analyze import Mechanism, ReactionIR, analyze, analyze_source
from .errors import BalanceError, RxncError, SemanticError, SyntaxErrorRxnc
from .mechanism import CompiledMechanism, compile_file, compile_mechanism
from .nodes import unparse
from .parser import parse
from .recognize import recognize_arrhenius

__version__ = "0.1.0"
__all__ = [
    "BalanceError",
    "CompiledMechanism",
    "Mechanism",
    "ReactionIR",
    "RxncError",
    "SemanticError",
    "SyntaxErrorRxnc",
    "analyze",
    "analyze_source",
    "compile_file",
    "compile_mechanism",
    "parse",
    "recognize_arrhenius",
    "unparse",
]
