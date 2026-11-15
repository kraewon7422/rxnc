"""Recursive-descent parser with precedence climbing for the expression grammar.

Grammar (EBNF-ish)::

    program    := statement*
    statement  := param_def | species_def | reaction
    param_def  := 'let' IDENT '=' expr NEWLINE
    species_def:= 'species' spec (',' spec)* NEWLINE
    spec       := IDENT ['=' (IDENT | STRING)]
    reaction   := side ('->' | '<=>') side (',' clause)* NEWLINE
    side       := term ('+' term)* | '0'
    term       := [NUMBER] IDENT
    clause     := IDENT ':' (expr | STRING)
    expr       := precedence climbing over + - * / ^ with unary minus
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .errors import SyntaxErrorRxnc
from .lexer import Token, tokenize
from .nodes import (
    BINARY_PRECEDENCE,
    RIGHT_ASSOC,
    Binary,
    Call,
    Conc,
    Expr,
    Num,
    ParamDef,
    Program,
    ReactionNode,
    SpeciesDef,
    Unary,
    Var,
)

CLAUSE_KEYS = {"k", "rate", "krev", "raterev", "dh", "name"}
EXPR_CLAUSES = {"k", "rate", "krev", "raterev", "dh"}


class Parser:
    def __init__(self, src: str):
        self.src = src
        self.toks: List[Token] = tokenize(src)
        self.i = 0

    # -- token helpers ----------------------------------------------------- #
    @property
    def cur(self) -> Token:
        return self.toks[self.i]

    def advance(self) -> Token:
        tok = self.toks[self.i]
        if tok.kind != "EOF":
            self.i += 1
        return tok

    def at_op(self, text: str) -> bool:
        return self.cur.kind == "OP" and self.cur.text == text

    def accept_op(self, text: str) -> bool:
        if self.at_op(text):
            self.advance()
            return True
        return False

    def expect_op(self, text: str) -> Token:
        if not self.at_op(text):
            raise self.error(f"expected {text!r}, found {self.describe(self.cur)}")
        return self.advance()

    def expect(self, kind: str, what: str) -> Token:
        if self.cur.kind != kind:
            raise self.error(f"expected {what}, found {self.describe(self.cur)}")
        return self.advance()

    @staticmethod
    def describe(tok: Token) -> str:
        if tok.kind == "EOF":
            return "end of input"
        if tok.kind == "NEWLINE":
            return "end of line"
        return repr(tok.text)

    def error(self, msg: str, tok: Optional[Token] = None) -> SyntaxErrorRxnc:
        tok = tok or self.cur
        return SyntaxErrorRxnc(msg, tok.line, tok.col, self.src)

    def skip_newlines(self) -> None:
        while self.cur.kind == "NEWLINE":
            self.advance()

    def end_statement(self) -> None:
        if self.cur.kind not in ("NEWLINE", "EOF"):
            raise self.error(f"unexpected {self.describe(self.cur)} after statement")
        self.skip_newlines()

    # -- statements -------------------------------------------------------- #
    def parse_program(self) -> Program:
        prog = Program()
        self.skip_newlines()
        while self.cur.kind != "EOF":
            if self.cur.kind == "KEYWORD" and self.cur.text == "let":
                prog.statements.append(self.parse_param())
            elif self.cur.kind == "KEYWORD" and self.cur.text == "species":
                prog.statements.extend(self.parse_species())
            else:
                prog.statements.append(self.parse_reaction())
            self.skip_newlines()
        return prog

    def parse_param(self) -> ParamDef:
        kw = self.advance()
        name = self.expect("IDENT", "a parameter name")
        self.expect_op("=")
        expr = self.parse_expr()
        self.end_statement()
        return ParamDef(name.text, expr, (kw.line, kw.col))

    def parse_species(self) -> List[SpeciesDef]:
        kw = self.advance()
        out: List[SpeciesDef] = []
        while True:
            name = self.expect("IDENT", "a species name")
            formula = None
            if self.accept_op("="):
                if self.cur.kind in ("IDENT", "STRING"):
                    formula = self.advance().text
                else:
                    raise self.error("expected a chemical formula")
            out.append(SpeciesDef(name.text, formula, (name.line, name.col)))
            if not self.accept_op(","):
                break
        self.end_statement()
        if not out:  # pragma: no cover - loop always appends at least once
            raise self.error("empty species declaration", kw)
        return out

    def parse_reaction(self) -> ReactionNode:
        start = self.cur
        reactants = self.parse_side()
        if self.cur.kind == "ARROW":
            reversible = False
        elif self.cur.kind == "REV":
            reversible = True
        else:
            raise self.error(f"expected '->' or '<=>', found {self.describe(self.cur)}")
        self.advance()
        products = self.parse_side()
        node = ReactionNode(reactants, products, reversible, {}, {}, (start.line, start.col))
        while self.accept_op(","):
            key_tok = self.expect("IDENT", "a clause name such as k, rate or dH")
            key = key_tok.text.lower()
            if key not in CLAUSE_KEYS:
                raise self.error(
                    f"unknown clause {key_tok.text!r}; expected one of "
                    + ", ".join(sorted(CLAUSE_KEYS)),
                    key_tok,
                )
            if key in node.clauses:
                raise self.error(f"duplicate clause {key_tok.text!r}", key_tok)
            self.expect_op(":")
            if key == "name":
                node.clauses[key] = self.expect("STRING", "a quoted name").text
            else:
                node.clauses[key] = self.parse_expr()
            node.clause_pos[key] = (key_tok.line, key_tok.col)
        self.end_statement()
        return node

    def parse_side(self) -> List[Tuple[float, str]]:
        """Parse ``2 A + B``; a bare ``0`` means no species on that side."""
        if self.cur.kind == "NUMBER" and self.cur.value == 0.0:
            nxt = self.toks[self.i + 1]
            if nxt.kind != "IDENT":
                self.advance()
                return []
        terms: List[Tuple[float, str]] = []
        while True:
            coeff = 1.0
            if self.cur.kind == "NUMBER":
                coeff = float(self.advance().value or 0.0)
                self.accept_op("*")
            name = self.expect("IDENT", "a species name")
            terms.append((coeff, name.text))
            if not self.at_op("+"):
                break
            self.advance()
        return terms

    # -- expressions (precedence climbing) --------------------------------- #
    def parse_expr(self, min_prec: int = 1) -> Expr:
        left = self.parse_unary()
        while self.cur.kind == "OP" and self.cur.text in BINARY_PRECEDENCE:
            op = self.cur.text
            prec = BINARY_PRECEDENCE[op]
            if prec < min_prec:
                break
            tok = self.advance()
            next_min = prec if op in RIGHT_ASSOC else prec + 1
            right = self.parse_expr(next_min)
            left = Binary(op, left, right, (tok.line, tok.col))
        return left

    def parse_unary(self) -> Expr:
        if self.at_op("-"):
            tok = self.advance()
            return Unary("-", self.parse_unary_or_power(), (tok.line, tok.col))
        if self.at_op("+"):
            self.advance()
            return self.parse_unary()
        return self.parse_power()

    def parse_unary_or_power(self) -> Expr:
        """``-a^b`` parses as ``-(a^b)``, matching usual mathematical reading."""
        return self.parse_expr(BINARY_PRECEDENCE["^"])

    def parse_power(self) -> Expr:
        base = self.parse_atom()
        if self.at_op("^"):
            tok = self.advance()
            exponent = self.parse_unary()  # right associative, binds unary minus
            return Binary("^", base, exponent, (tok.line, tok.col))
        return base

    def parse_atom(self) -> Expr:
        tok = self.cur
        if tok.kind == "NUMBER":
            self.advance()
            return Num(float(tok.value or 0.0), (tok.line, tok.col))
        if tok.kind == "IDENT":
            self.advance()
            if self.at_op("("):
                self.advance()
                args: List[Expr] = []
                if not self.at_op(")"):
                    while True:
                        args.append(self.parse_expr())
                        if not self.accept_op(","):
                            break
                self.expect_op(")")
                return Call(tok.text, tuple(args), (tok.line, tok.col))
            return Var(tok.text, (tok.line, tok.col))
        if tok.kind == "OP" and tok.text == "[":
            self.advance()
            name = self.expect("IDENT", "a species name inside [...]")
            self.expect_op("]")
            return Conc(name.text, (tok.line, tok.col))
        if tok.kind == "OP" and tok.text == "(":
            self.advance()
            inner = self.parse_expr()
            self.expect_op(")")
            return inner
        raise self.error(f"expected a value, found {self.describe(tok)}")


def parse(src: str) -> Program:
    """Parse DSL source into a :class:`Program`."""
    return Parser(src).parse_program()
