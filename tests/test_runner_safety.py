"""Tests for the runner's safety rails: dirty-tree preflight and crash cleanup."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conftest import content_test_runner, make_fake_provider
from repo_surgeon.config import AppConfig
from repo_surgeon.models import RunStatus
from repo_surgeon.models import TestResult as SandboxResult  # alias: avoid pytest collection
from repo_surgeon.runner import run_issue
from repo_surgeon.workspace import LocalWorkspace
from repo_surgeon.workspace.base import DirtyWorkspaceError, Workspace

_RIGHT = (
    "calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"
)


def _provider():
    return make_fake_provider(
        {
            "summarizer": ["calc module"],
            "localizer": ["calc.py"],
            "planner": ["- fix add"],
            "writer": [_RIGHT],
        }
    )


def _branches(repo: Path) -> str:
    return subprocess.run(
        ["git", "branch", "--format=%(refname:short)"], cwd=repo, capture_output=True, text=True
    ).stdout


def test_refuses_dirty_tree(buggy_repo: Path, app_config: AppConfig) -> None:
    (buggy_repo / "calc.py").write_text("def add(a, b):\n    return -1\n", encoding="utf-8")
    with pytest.raises(DirtyWorkspaceError, match="uncommitted"):
        run_issue(
            "Fix add() in calc.py",
            str(buggy_repo),
            config=app_config,
            llm_provider=_provider(),
            test_runner=content_test_runner("a + b"),
        )
    # The dirty edit is untouched and no work branch was created.
    assert "return -1" in (buggy_repo / "calc.py").read_text(encoding="utf-8")
    assert "repo-surgeon/" not in _branches(buggy_repo)


def test_allow_dirty_overrides_guard(buggy_repo: Path, app_config: AppConfig) -> None:
    (buggy_repo / "scratch.txt").write_text("untracked, fine", encoding="utf-8")
    final = run_issue(
        "Fix add() in calc.py",
        str(buggy_repo),
        config=app_config,
        allow_dirty=True,
        llm_provider=_provider(),
        test_runner=content_test_runner("a + b"),
    )
    assert final["status"] == RunStatus.RESOLVED


def test_untracked_files_do_not_block(buggy_repo: Path, app_config: AppConfig) -> None:
    (buggy_repo / "notes.md").write_text("my notes", encoding="utf-8")
    final = run_issue(
        "Fix add() in calc.py",
        str(buggy_repo),
        config=app_config,
        llm_provider=_provider(),
        test_runner=content_test_runner("a + b"),
    )
    assert final["status"] == RunStatus.RESOLVED
    # The user's untracked file survives AND is not swept into the fix commit.
    assert (buggy_repo / "notes.md").exists()
    shown = subprocess.run(
        ["git", "show", "--stat", final["delivery_ref"]],
        cwd=buggy_repo,
        capture_output=True,
        text=True,
    ).stdout
    assert "notes.md" not in shown


def test_success_returns_to_original_branch(buggy_repo: Path, app_config: AppConfig) -> None:
    ws = LocalWorkspace(buggy_repo)
    original = ws.current_branch()
    final = run_issue(
        "Fix add() in calc.py",
        str(buggy_repo),
        config=app_config,
        llm_provider=_provider(),
        test_runner=content_test_runner("a + b"),
    )
    assert final["status"] == RunStatus.RESOLVED
    assert LocalWorkspace(buggy_repo).current_branch() == original
    assert final["delivery_ref"] in _branches(buggy_repo)


def test_crash_mid_run_cleans_up(buggy_repo: Path, app_config: AppConfig) -> None:
    def exploding_runner(workspace: Workspace) -> SandboxResult:
        raise RuntimeError("sandbox exploded")

    ws = LocalWorkspace(buggy_repo)
    original = ws.current_branch()
    with pytest.raises(RuntimeError, match="sandbox exploded"):
        run_issue(
            "Fix add() in calc.py",
            str(buggy_repo),
            config=app_config,
            llm_provider=_provider(),
            test_runner=exploding_runner,
        )
    # Back on the original branch, edits discarded, no leftover work branch.
    fresh = LocalWorkspace(buggy_repo)
    assert fresh.current_branch() == original
    assert "a - b" in fresh.read_file("calc.py")
    assert "repo-surgeon/" not in _branches(buggy_repo)
