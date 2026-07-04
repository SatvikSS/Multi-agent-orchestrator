"""Tests for the local workspace and workspace factory."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repo_surgeon.models import Patch
from repo_surgeon.workspace import LocalWorkspace, open_workspace


def test_list_files_filters_by_suffix(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    py = ws.list_files(suffix=".py")
    assert "calc.py" in py
    assert all(f.endswith(".py") for f in py)


def test_read_file(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    assert "def add" in ws.read_file("calc.py")


def test_apply_patch_success(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    patch = Patch(file_path="calc.py", search="return a + b", replace="return a + b + 0")
    assert ws.apply_patch(patch) is True
    assert "a + b + 0" in ws.read_file("calc.py")


def test_apply_patch_no_match_returns_false(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    patch = Patch(file_path="calc.py", search="nonexistent text", replace="x")
    assert ws.apply_patch(patch) is False


def test_apply_patch_rejects_path_escape(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    patch = Patch(file_path="../escape.py", search="a", replace="b")
    with pytest.raises(ValueError, match="escapes"):
        ws.apply_patch(patch)


def test_deliver_commits_on_work_branch_and_returns_home(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    original = ws.current_branch()
    branch = ws.start_work("fix add rounding")
    assert branch == "repo-surgeon/fix-add-rounding"
    assert ws.current_branch() == branch
    ws.apply_patch(Patch(file_path="calc.py", search="a + b", replace="a + b  # touched"))
    ref = ws.deliver(title="fix: x", body="body")
    assert ref == branch
    # Back on the original branch; the edit lives only on the work branch.
    assert ws.current_branch() == original
    assert "# touched" not in ws.read_file("calc.py")


def test_start_work_twice_raises(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    ws.start_work("first")
    with pytest.raises(RuntimeError, match="already in progress"):
        ws.start_work("second")


def test_start_work_branch_collision_gets_suffix(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    first = ws.start_work("same title")
    ws.deliver(title="fix: a", body="b")
    second = ws.start_work("same title")
    assert first == "repo-surgeon/same-title"
    assert second == "repo-surgeon/same-title-2"


def test_abort_work_restores_everything(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    original = ws.current_branch()
    branch = ws.start_work("doomed fix")
    ws.write_file("calc.py", "def add(a, b):\n    return 999\n")
    ws.abort_work()
    assert ws.current_branch() == original
    assert "999" not in ws.read_file("calc.py")
    branches = subprocess.run(
        ["git", "branch", "--format=%(refname:short)"],
        cwd=toy_repo,
        capture_output=True,
        text=True,
    ).stdout
    assert branch not in branches


def test_is_dirty_tracked_only(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    assert ws.is_dirty() is False
    # Untracked files do not make the workspace dirty.
    (toy_repo / "scratch.txt").write_text("notes", encoding="utf-8")
    assert ws.is_dirty() is False
    # Modifying a tracked file does.
    ws.write_file("calc.py", "def add(a, b):\n    return 0\n")
    assert ws.is_dirty() is True


def test_kb_cache_never_committed(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    cache = toy_repo / ".repo_surgeon_cache"
    cache.mkdir()
    (cache / "kb.json").write_text("{}", encoding="utf-8")
    ws.start_work("cache exclusion")
    ws.apply_patch(Patch(file_path="calc.py", search="a + b", replace="a + b  # x"))
    branch = ws.deliver(title="fix: x", body="body")
    shown = subprocess.run(
        ["git", "show", "--stat", branch], cwd=toy_repo, capture_output=True, text=True
    ).stdout
    assert ".repo_surgeon_cache" not in shown


def test_reset_restores_tracked_files(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    ws.write_file("calc.py", "def add(a, b):\n    return 999\n")
    assert "999" in ws.read_file("calc.py")
    ws.reset()
    assert "999" not in ws.read_file("calc.py")
    assert "a + b" in ws.read_file("calc.py")


def test_workspace_factory_local(toy_repo: Path) -> None:
    assert isinstance(open_workspace(str(toy_repo)), LocalWorkspace)


def test_workspace_factory_github_shorthand(monkeypatch: pytest.MonkeyPatch) -> None:
    # GitHubWorkspace clones eagerly, so stub it: assert the factory routes to it.
    created: dict[str, str] = {}

    class _Stub:
        def __init__(self, repo: str, *, token: str | None = None) -> None:
            created["repo"] = repo

    import repo_surgeon.workspace.factory as factory_module

    monkeypatch.setattr(factory_module, "GitHubWorkspace", _Stub)
    open_workspace("octocat/hello")
    assert created["repo"] == "octocat/hello"


def test_non_git_dir_raises(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ValueError, match="not a git repository"):
        LocalWorkspace(plain)
