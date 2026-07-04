"""Docker integration tests for the sandbox and the full loop with a REAL test runner.

Marked `docker` and skipped when no Docker daemon is reachable. The first run builds the
`repo-surgeon-sandbox` base image (pulls python:3.11-slim + installs pytest), so it is slow;
subsequent runs reuse the cached image.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_fake_provider
from repo_surgeon.config import AppConfig, Sandbox
from repo_surgeon.graph import build_graph
from repo_surgeon.issues import LocalIssueSource
from repo_surgeon.models import RunStatus
from repo_surgeon.sandbox import make_docker_runner, run_pytest
from repo_surgeon.state import RunState
from repo_surgeon.workspace import LocalWorkspace

docker = pytest.importorskip("docker")


def _docker_available() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not _docker_available(), reason="no Docker daemon available"),
]

_CFG = Sandbox(timeout_seconds=120)


def test_run_pytest_passes_on_good_repo(toy_repo: Path) -> None:
    result = run_pytest(toy_repo, _CFG)
    assert result.passed is True
    assert result.total >= 1


def test_run_pytest_fails_on_buggy_repo(buggy_repo: Path) -> None:
    result = run_pytest(buggy_repo, _CFG)
    assert result.passed is False
    assert result.failed >= 1
    assert any("test_add" in f for f in result.failures)


def test_full_loop_fixes_seeded_bug_verified_by_real_docker(
    buggy_repo: Path, app_config: AppConfig
) -> None:
    """The milestone: a seeded bug is fixed autonomously and verified by real pytest in Docker."""
    right = "calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"
    wrong = "calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a * b\n>>>>>>> REPLACE"
    provider = make_fake_provider(
        {
            "summarizer": ["calc module"],
            "localizer": ["calc.py"],
            "planner": ["- add() should add"],
            "writer": [wrong, right],  # first fails real tests, second passes
        }
    )
    ws = LocalWorkspace(buggy_repo)
    ws.start_work("Fix add() in calc.py")
    graph = build_graph(ws, app_config, llm_provider=provider, test_runner=make_docker_runner(_CFG))

    entry: RunState = {
        "issue": LocalIssueSource("Fix add() in calc.py").fetch(),
        "repo_ref": str(buggy_repo),
        "attempts": 0,
        "tokens_spent": 0,
        "cost_usd": 0.0,
        "status": RunStatus.IN_PROGRESS,
        "patches": [],
        "test_results": [],
        "apply_ok": False,
        "writer_feedback": "",
        "notes": [],
    }
    final = graph.invoke(entry, config={"recursion_limit": 50})

    assert final["status"] == RunStatus.RESOLVED
    assert final["attempts"] == 2
    # The fix is committed on the delivered work branch.
    import subprocess

    shown = subprocess.run(
        ["git", "show", f"{final['delivery_ref']}:calc.py"],
        cwd=buggy_repo,
        capture_output=True,
        text=True,
    ).stdout
    assert "a + b" in shown


def test_eval_case_end_to_end_real_docker(app_config: AppConfig) -> None:
    """Run one built-in eval case with real pytest in Docker (fake LLM returns the fix)."""
    from conftest import make_fake_provider
    from repo_surgeon.evaluation import load_cases, run_eval
    from repo_surgeon.evaluation.harness import builtin_cases_dir

    cases = [c for c in load_cases(builtin_cases_dir()) if c.name == "operator_bug"]
    right = "calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"
    provider = make_fake_provider(
        {
            "summarizer": ["calc"],
            "localizer": ["calc.py"],
            "planner": ["- add() should add"],
            "writer": [right],
        }
    )
    summary = run_eval(cases, config=app_config, llm_provider=provider)
    assert summary.total == 1
    assert summary.resolved == 1
    assert summary.resolved_rate == 1.0
