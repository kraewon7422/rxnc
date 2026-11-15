"""Command line interface: ``rxnc <command> <file.rxn>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from .analyze import analyze_source
from .codegen import generate_standalone
from .errors import RxncError
from .mechanism import CompiledMechanism


def _parse_state(text: Optional[str]) -> Dict[str, float]:
    state: Dict[str, float] = {}
    if not text:
        return state
    for part in text.split(","):
        if "=" not in part:
            raise SystemExit(f"bad --at entry {part!r}; expected SPECIES=VALUE")
        name, value = part.split("=", 1)
        state[name.strip()] = float(value)
    return state


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rxnc", description="Reaction-mechanism compiler")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("file", help="mechanism source (.rxn)")
    common.add_argument(
        "--strict-balance",
        action="store_true",
        help="fail when a reaction cannot be element-balance checked",
    )

    sub.add_parser("check", parents=[common], help="parse, analyse and report")
    emit = sub.add_parser("emit", parents=[common], help="write the generated Python kernel")
    emit.add_argument("-o", "--output", help="output file (default: stdout)")
    sub.add_parser("jacobian", parents=[common], help="print the symbolic rate derivatives")
    ev = sub.add_parser("eval", parents=[common], help="evaluate rates at one state")
    ev.add_argument("--at", required=True, help="state, e.g. 'A=1000,B=0'")
    ev.add_argument("--temperature", "-T", type=float, required=True, help="temperature, K")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.file)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "check":
            mech = analyze_source(text, strict_balance=args.strict_balance)
            print(mech.summary())
            checked = sum(1 for r in mech.reactions if r.balance_checked)
            print(f"ok: {checked}/{mech.n_reactions} reactions element-balanced")
            return 0

        if args.command == "emit":
            mech = analyze_source(text, strict_balance=args.strict_balance)
            out = generate_standalone(mech, origin=str(path))
            if args.output:
                Path(args.output).write_text(out, encoding="utf-8")
                print(f"wrote {args.output}")
            else:
                print(out)
            return 0

        compiled = CompiledMechanism.from_source(
            text, backend="python", strict_balance=args.strict_balance
        )
        if args.command == "jacobian":
            print(compiled.jacobian_report())
            return 0

        state = _parse_state(args.at)
        C = compiled.vector(state)
        T = args.temperature
        rates = compiled.rates(C, T)
        production = compiled.production(C, T)
        print(f"T = {T:g} K")
        for rxn, r in zip(compiled.mech.reactions, rates, strict=True):
            print(f"  rate[{rxn.name}] = {r:.6g} mol/(m3 s)   ({rxn.equation})")
        for name, value in zip(compiled.species, production, strict=True):
            print(f"  d[{name}]/dt = {value:.6g} mol/(m3 s)")
        print(f"  heat release = {compiled.heat_release(C, T):.6g} W/m3")
        return 0
    except RxncError as exc:
        print(exc.with_source(text).render(), file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
