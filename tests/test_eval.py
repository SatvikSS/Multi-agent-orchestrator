"""Tests for the evaluation harness (offline with fakes; one Docker-marked full run)."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import content_test_runner, make_fake_provider
from repo_surgeon.config import AppConfig
from repo_surgeon.evaluation import builtin_cases_dir, load_cases, run_eval

_A_MINUS_B = "def add(a, b):\n    return a - b\n"
_RIGHT = (
    "calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"
)
_WRONG = (
    "calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a * b\n>>>>>>> REPLACE"
)


def _write_case(cases_dir: Path, name: str) -> None:
    case = cases_dir / name
    case.mkdir(parents=True)
    (case / "meta.json").write_text(
        json.dumps({"issue": "add() in calc.py subtracts instead of adding."}), encoding="utf-8"
    )
    (case / "calc.py").write_text(_A_MINUS_B, encoding="utf-8")
    (case / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8"
    )


def _provider(writer: list[str]):
    return make_fake_provider(
        {
            "summarizer": ["calc module"],
            "localizer": ["calc.py"],
            "planner": ["- make add() add"],
            "writer": writer,
        }
    )


def test_load_builtin_cases() -> None:
    cases = load_cases(builtin_cases_dir())
    names = {c.name for c in cases}
    assert {"pagination", "operator_bug", "safe_get"} <= names
    assert all(c.issue for c in cases)


def test_run_eval_all_resolved(tmp_path: Path, app_config: AppConfig) -> None:
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "case_a")
    _write_case(cases_dir, "case_b")
    cases = load_cases(cases_dir)

    summary = run_eval(
        cases,
        config=app_config,
        llm_provider=_provider([_RIGHT]),
        test_runner=content_test_runner("a + b"),
    )
    assert summary.total == 2
    assert summary.resolved == 2
    assert summary.resolved_rate == 1.0
    assert summary.avg_attempts == 1.0
    assert all(r.resolved for r in summary.results)


def test_run_eval_unresolved_still_produces_patch(
    tmp_path: Path, app_config: AppConfig
) -> None:
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "case_a")
    tight = app_config.model_copy(
        update={"budget": app_config.budget.model_copy(update={"max_attempts": 2})}
    )
    summary = run_eval(
        load_cases(cases_dir),
        config=tight,
        llm_provider=_provider([_WRONG]),  # applies but never passes tests
        test_runner=content_test_runner("a + b"),
    )
    assert summary.resolved_rate == 0.0
    assert summary.produced_patch_rate == 1.0  # a patch was produced, just wrong


def test_run_eval_handles_crashing_case(tmp_path: Path, app_config: AppConfig) -> None:
    cases_dir = tmp_path / "cases"
    _write_case(cases_dir, "case_a")

    def boom(config: AppConfig, role: str):
        raise RuntimeError("provider down")

    summary = run_eval(
        load_cases(cases_dir),
        config=app_config,
        llm_provider=boom,
        test_runner=content_test_runner("a + b"),
    )
    assert summary.total == 1
    assert summary.resolved == 0
    assert summary.results[0].status == "error"
    assert summary.results[0].error is not None


def test_empty_summary() -> None:
    from repo_surgeon.evaluation.harness import _summarize

    summary = _summarize([])
    assert summary.total == 0
    assert summary.resolved_rate == 0.0
