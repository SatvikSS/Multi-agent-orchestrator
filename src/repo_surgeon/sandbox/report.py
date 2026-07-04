"""Parse pytest console output into a structured TestResult (pure, no Docker)."""

from __future__ import annotations

import re

from repo_surgeon.models import TestResult

_COUNT = {
    "passed": re.compile(r"(\d+) passed"),
    "failed": re.compile(r"(\d+) failed"),
    "error": re.compile(r"(\d+) error"),
}
# Short-summary lines emitted by `pytest -rfE`, e.g. "FAILED tests/test_x.py::test_y - ...".
_SUMMARY_LINE = re.compile(r"^(FAILED|ERROR)\s+(.+)$", re.MULTILINE)


def parse_pytest_output(stdout: str, return_code: int) -> TestResult:
    """Turn pytest output + exit code into a TestResult.

    pytest exit codes: 0 = all passed, 1 = tests failed, 5 = no tests collected,
    2/3/4 = internal/usage/interrupted errors. Only exit 0 counts as passing.
    """
    if return_code == 5:
        return TestResult(
            passed=False, total=0, failed=0, failures=("no tests collected",), raw_output=stdout
        )

    passed_n = _count(stdout, "passed")
    failed_n = _count(stdout, "failed") + _count(stdout, "error")
    failures = tuple(f"{kind} {name}" for kind, name in _SUMMARY_LINE.findall(stdout))

    ok = return_code == 0 and failed_n == 0
    if not ok and not failures:
        failures = (f"pytest exited with code {return_code}",)

    return TestResult(
        passed=ok,
        total=passed_n + failed_n,
        failed=failed_n,
        failures=failures,
        raw_output=stdout,
    )


def _count(text: str, key: str) -> int:
    match = _COUNT[key].search(text)
    return int(match.group(1)) if match else 0
