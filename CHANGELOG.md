# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-15

### Added
- Reaction-mechanism DSL: `let` parameters, `species` declarations with chemical
  formulas, irreversible (`->`) and reversible (`<=>`) reactions, and the
  `k:` / `rate:` / `krev:` / `raterev:` / `dH:` / `name:` clauses.
- Hand-written lexer and recursive-descent parser (precedence climbing for
  expressions) with source positions and caret diagnostics.
- Semantic analysis: name resolution, parameter folding, mass-action expansion
  from reactant coefficients, and element-balance checking with a conservative
  formula-inference rule.
- Symbolic differentiation with simplification, producing analytic
  `d(rate)/dC` and `d(rate)/dT`.
- Two backends with an identical call contract: a tree-walking interpreter
  (reference) and a Python code generator that shares common subexpressions
  across rates and derivatives and emits only structurally nonzero Jacobian
  entries.
- `CompiledMechanism` with `rates`, `production`, `heat_release`,
  `rate_jacobian`, `production_jacobian`, `dproduction_dT`, `dheat_dT` and
  `full_jacobian`.
- Export to [reactor-api](https://github.com/kraewon7422/reactor-api)
  `ReactionNetwork`, with numeric Arrhenius identification verified at four
  temperatures to 1e-9 relative; non-Arrhenius rate laws are rejected, never
  approximated.
- CLI: `rxnc check | emit | jacobian | eval`, plus `python -m rxnc`.
- Example mechanisms and a backend benchmark script.

[0.1.0]: https://github.com/kraewon7422/rxnc/releases/tag/v0.1.0
