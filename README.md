# rxnc — a reaction-mechanism compiler

[![CI](https://github.com/kraewon7422/rxnc/actions/workflows/ci.yml/badge.svg)](https://github.com/kraewon7422/rxnc/actions/workflows/ci.yml)

`rxnc` reads a mechanism written in a small domain-specific language, checks it,
differentiates it symbolically, and generates a straight-line Python kernel that
returns reaction rates **and their analytic Jacobian** in one call.

```
A + B -> C, rate: 4.0e5*exp(-25e3/(R*T))*[A]*[B]/(1 + 0.02*[C]), dH: -80e3
```

The pipeline is the usual one: lexer → recursive-descent parser → AST →
semantic analysis → IR → symbolic differentiation → code generation, with a
tree-walking interpreter kept as a reference backend so the generated code can
be differential-tested against it.

## Quick start

```bash
pip install -e .

rxnc check     examples/no_oxidation.rxn      # parse, resolve, balance-check
rxnc jacobian  examples/no_oxidation.rxn      # symbolic rate derivatives
rxnc emit      examples/no_oxidation.rxn -o kernel.py   # standalone module
rxnc eval      examples/no_oxidation.rxn --at "NO=100,O2=50" -T 600
```

```python
from rxnc import compile_mechanism

mech = compile_mechanism(open("examples/multiplicity.rxn").read())
mech.rates([1000.0, 0.0], 350.0)  # mol/(m3 s)
mech.production_jacobian([1000.0, 0.0], 350.0)  # analytic, not finite-difference
mech.full_jacobian([1000.0, 0.0], 350.0)  # [production, heat] vs [C, T]
print(mech.source)  # the generated kernel
```

## Language

A mechanism file is a sequence of newline-terminated statements; `#` starts a
comment.

| statement | meaning |
| --- | --- |
| `let NAME = <constant expression>` | a parameter, folded into the rate laws at compile time |
| `species NAME [= FORMULA], ...` | declares a species (and optionally its formula) ahead of use |
| `<reactants> -> <products>, <clauses>` | an irreversible reaction |
| `<reactants> <=> <products>, <clauses>` | a reversible reaction |

Stoichiometry is written `2 NO + O2 -> 2 NO2`; a bare `0` on either side means
"nothing" (a source or a sink term). Clauses, separated by commas:

| clause | meaning |
| --- | --- |
| `k: <expr>` | rate **coefficient** — the compiler multiplies in the mass-action product of the reactants, `k·∏[Rᵢ]^νᵢ` |
| `rate: <expr>` | the complete rate law, used exactly as written |
| `krev:` / `raterev:` | the same two choices for the reverse step of a `<=>` reaction |
| `dH: <expr>` | heat of reaction, J per mol of extent (negative = exothermic) |
| `name: "..."` | a label used in diagnostics and reports |

Rate expressions are ordinary algebra over `+ - * / ^` with the functions
`exp`, `log`, `log10`, `sqrt`, `abs`, `pow`, `max`, `min`. `T` is the
temperature in K, `R` the gas constant, `[A]` the concentration of species `A`,
and any parameter defined with `let` may be used by name. `^` is
right-associative and unary minus binds looser than it, so `-2^2` is `-4`.

A species that appears only in a rate law — a catalyst, a site blocker, an
inert that carries heat capacity — must be declared with `species`, which also
pins it into the state-vector ordering.

## Element balance

When every species in a reaction has a formula, `rxnc` checks that the reaction
balances and reports the offending elements if it does not:

```
balance error: 'H2 + O2 -> H2O' does not balance (O: -1)
  --> line 1, column 1
   |
   | H2 + O2 -> H2O, k: 1
   | ^
```

A formula comes from a `species X = <formula>` declaration, or from the species
name itself when the name is unambiguously a formula — it contains a digit or
more than one element symbol. So `2 NO + O2 -> 2 NO2` is checked, while
`A -> B` is skipped rather than silently read as boron. `--strict-balance`
turns an unbalanceable reaction into an error instead of a note.

## What gets generated

The code generator emits one kernel that computes the rates, the nonzero
`∂rate/∂C` entries and the nonzero `∂rate/∂T` entries together, so the shared
Arrhenius factors and concentration products are computed once. Structural
zeros are never emitted at all: a rate that does not mention a species
contributes no Jacobian entry.

```python
def kernel(C, T):
    _c0 = C[0]
    _t0 = 8.314462618 * T
    _t1 = exp((-72747.0) / _t0)
    _t2 = 1200000000.0 * _t1
    ...
```

`rxnc emit` writes that kernel to a standalone module together with `SPECIES`,
`NU`, `DH` and the index patterns that give meaning to the kernel's return
values — no `rxnc` import needed at run time.

## Exporting to reactor_api

`CompiledMechanism.to_reaction_network()` converts a mechanism into a
[`reactor_api`](https://github.com/kraewon7422/reactor-api)
`ReactionNetwork`, which understands mass action with modified-Arrhenius
coefficients and nothing else. Deciding whether a given `k:` expression *is*
`A·(T/T_ref)^b·exp(−Ea/RT)` is done numerically — fit the three parameters from
three temperatures, then verify at four more to a relative tolerance of 1e-9 —
so an equivalent-but-odd spelling such as `exp(log(1e5) - 50e3/(R*T))` is
recognised, while anything that is not Arrhenius is *rejected* rather than
approximated. A reaction written with `rate:` refuses to export, naming itself
in the error.

## Layout

| file | role |
| --- | --- |
| `rxnc/lexer.py` | tokenizer with source positions |
| `rxnc/nodes.py` | AST nodes, canonical keys, precedence-aware unparser |
| `rxnc/parser.py` | recursive descent + precedence climbing |
| `rxnc/formula.py` | chemical formula parsing, element balance |
| `rxnc/symbolic.py` | simplification and symbolic differentiation |
| `rxnc/analyze.py` | name resolution, mass-action expansion, the IR |
| `rxnc/interp.py` | reference backend (tree walking) |
| `rxnc/codegen.py` | Python backend with common-subexpression elimination |
| `rxnc/recognize.py` | Arrhenius identification with numeric verification |
| `rxnc/mechanism.py` | compiled artefact, Jacobian assembly, export |
| `rxnc/cli.py` | `check` / `emit` / `jacobian` / `eval` |

## Verification

Nothing is tested against stored outputs. Each layer has an independent oracle:

1. the parser is checked against the arithmetic value of the expression it builds, and against unparse round-trips;
2. symbolic derivatives are checked against central finite differences on random states;
3. the generated code is checked against the tree-walking interpreter on random states, including randomly generated mechanisms;
4. assembled production Jacobians are checked against finite differences of the production rates;
5. the `ReactionNetwork` export is checked against `reactor_api` itself, which is an independent implementation of the same mass-action kinetics.

```bash
pytest -q
```

Tests that need `reactor_api` skip when it is not installed.

## Limitations

Rate laws must be differentiable: `max` and `min` compile and evaluate, but
asking for a Jacobian through them is an error rather than a one-sided
derivative. Concentrations are used as given — there is no clipping of negative
trial states, which is the integrator's job. Expressions are scalar; there is no
vectorised-over-states evaluation yet, and no third-party-backend (NumPy, JAX,
C) code generator — the emitter is structured so that adding one means
replacing `codegen.Emitter`, not the pipeline.
