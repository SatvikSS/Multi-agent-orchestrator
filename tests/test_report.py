"""Tests for parsing pytest console output into a TestResult."""

from __future__ import annotations

from repo_surgeon.sandbox import parse_pytest_output


def test_all_passed() -> None:
    result = parse_pytest_output("2 passed in 0.03s", 0)
    assert result.passed is True
    assert result.total == 2
    assert result.failed == 0


def test_some_failed() -> None:
    out = "FAILED test_calc.py::test_add - assert 1 == 3\n1 failed, 1 passed in 0.05s"
    result = parse_pytest_output(out, 1)
    assert result.passed is False
    assert result.failed == 1
    assert result.total == 2
    assert any("test_add" in f for f in result.failures)


def test_errors_counted() -> None:
    out = "ERROR test_x.py - ImportError\n1 error in 0.01s"
    result = parse_pytest_output(out, 2)
    assert result.passed is False
    assert result.failed == 1
    assert any("test_x.py" in f for f in result.failures)


def test_no_tests_collected() -> None:
    result = parse_pytest_output("no tests ran in 0.00s", 5)
    assert result.passed is False
    assert result.total == 0
    assert "no tests collected" in result.failures[0]


def test_nonzero_exit_without_summary_still_fails() -> None:
    result = parse_pytest_output("some crash output", 3)
    assert result.passed is False
    assert result.failures  # a synthetic failure message is added
