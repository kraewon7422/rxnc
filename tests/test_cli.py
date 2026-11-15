"""Command line interface."""

import pytest
from mechanisms import GAS_PHASE, NONLINEAR

from rxnc.cli import main


@pytest.fixture()
def gas_file(tmp_path):
    path = tmp_path / "gas.rxn"
    path.write_text(GAS_PHASE, encoding="utf-8")
    return str(path)


def test_check_reports_balance(gas_file, capsys):
    assert main(["check", gas_file]) == 0
    out = capsys.readouterr().out
    assert "2/2 reactions element-balanced" in out
    assert "rate =" in out


def test_emit_writes_a_module(gas_file, tmp_path, capsys):
    out_path = tmp_path / "kernel.py"
    assert main(["emit", gas_file, "-o", str(out_path)]) == 0
    text = out_path.read_text(encoding="utf-8")
    assert "def kernel(C, T)" in text and "DC_PATTERN" in text
    assert "wrote" in capsys.readouterr().out


def test_emit_to_stdout(gas_file, capsys):
    assert main(["emit", gas_file]) == 0
    assert "def kernel(C, T)" in capsys.readouterr().out


def test_jacobian_lists_derivatives(gas_file, capsys):
    assert main(["jacobian", gas_file]) == 0
    out = capsys.readouterr().out
    assert "d(rate[R1])/d[NO]" in out and "/dT" in out


def test_eval_prints_rates_and_heat(gas_file, capsys):
    code = main(["eval", gas_file, "--at", "NO=100,O2=50", "-T", "600"])
    assert code == 0
    out = capsys.readouterr().out
    assert "heat release" in out and "d[NO2]/dt" in out


def test_syntax_error_exits_nonzero_with_a_caret(tmp_path, capsys):
    path = tmp_path / "bad.rxn"
    path.write_text("A -> B, k: *2\n", encoding="utf-8")
    assert main(["check", str(path)]) == 1
    err = capsys.readouterr().err
    assert "syntax error" in err and "^" in err


def test_strict_balance_flag_is_honoured(tmp_path, capsys):
    path = tmp_path / "placeholder.rxn"
    path.write_text("A -> B, k: 1\n", encoding="utf-8")
    assert main(["check", str(path)]) == 0
    assert main(["check", str(path), "--strict-balance"]) == 1
    assert "balance error" in capsys.readouterr().err


def test_missing_file_exits_two(tmp_path, capsys):
    assert main(["check", str(tmp_path / "nope.rxn")]) == 2


def test_eval_rejects_a_malformed_state(tmp_path):
    path = tmp_path / "m.rxn"
    path.write_text(NONLINEAR, encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["eval", str(path), "--at", "A", "-T", "400"])
