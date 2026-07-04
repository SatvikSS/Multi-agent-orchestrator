"""Tests for the workspace resolver: walk-up, scan-down, ambiguity, duplicates."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repo_surgeon.workspace import LocalWorkspace, open_workspace
from repo_surgeon.workspace.resolver import (
    AmbiguousWorkspaceError,
    NoRepoFoundError,
    resolve_repo,
)


def _make_repo(path: Path, *, origin: str | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@t.co")
    git("config", "user.name", "T")
    (path / "main.py").write_text("x = 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    if origin:
        git("remote", "add", "origin", origin)
    return path


def test_path_is_repo_root(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "proj")
    resolution = resolve_repo(repo)
    assert resolution.kind == "repo"
    assert resolution.root == str(repo)


def test_path_inside_repo_walks_up(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "proj")
    sub = repo / "src" / "pkg"
    sub.mkdir(parents=True)
    resolution = resolve_repo(sub)
    assert resolution.kind == "inside_repo"
    assert resolution.root == str(repo)
    assert "using the repo root" in resolution.note


def test_container_with_single_nested_repo(tmp_path: Path) -> None:
    container = tmp_path / "CPO_v3"
    container.mkdir()
    (container / "notes.txt").write_text("loose file", encoding="utf-8")
    nested = _make_repo(container / "chillerPerformanceCalculator")
    resolution = resolve_repo(container)
    assert resolution.kind == "nested_repo"
    assert resolution.root == str(nested)
    assert "chillerPerformanceCalculator" in resolution.note


def test_container_with_multiple_repos_is_ambiguous(tmp_path: Path) -> None:
    container = tmp_path / "CPC Models"
    _make_repo(container / "model-a", origin="https://github.com/org/model-a.git")
    _make_repo(container / "model-b", origin="https://github.com/org/model-b.git")
    with pytest.raises(AmbiguousWorkspaceError) as excinfo:
        resolve_repo(container)
    rels = [c.rel for c in excinfo.value.candidates]
    assert rels == ["model-a", "model-b"]
    assert "2 git repositories" in str(excinfo.value)


def test_duplicate_clones_flagged_by_origin(tmp_path: Path) -> None:
    container = tmp_path / "models"
    # Same repo, one with and one without the '.git' suffix — still duplicates.
    _make_repo(container / "copy1", origin="https://github.com/org/chillerPC.git")
    _make_repo(container / "copy2", origin="https://github.com/org/chillerPC")
    with pytest.raises(AmbiguousWorkspaceError) as excinfo:
        resolve_repo(container)
    assert "duplicate clones" in str(excinfo.value)


def test_repos_within_repos_not_double_counted(tmp_path: Path) -> None:
    container = tmp_path / "stack"
    outer = _make_repo(container / "outer")
    _make_repo(outer / "vendored-inner")  # like influx-client inside chillerPerformanceCalculator
    resolution = resolve_repo(container)
    # Only the outer repo is a candidate; its vendored inner repo is not surfaced.
    assert resolution.kind == "nested_repo"
    assert resolution.root == str(outer)


def test_scan_respects_depth_limit(tmp_path: Path) -> None:
    container = tmp_path / "deep"
    _make_repo(container / "a" / "b" / "c" / "repo")  # depth 4 — beyond the limit
    with pytest.raises(NoRepoFoundError):
        resolve_repo(container)


def test_scan_skips_junk_dirs(tmp_path: Path) -> None:
    container = tmp_path / "proj"
    _make_repo(container / ".venv" / "some-vendored-repo")
    _make_repo(container / "node_modules" / "dep")
    with pytest.raises(NoRepoFoundError):
        resolve_repo(container)


def test_plain_folder_raises_no_repo_found(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "script.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(NoRepoFoundError, match="shadow-git"):
        resolve_repo(plain)


def test_open_workspace_resolves_nested(tmp_path: Path) -> None:
    container = tmp_path / "wrap"
    nested = _make_repo(container / "actual-repo")
    ws = open_workspace(str(container))
    assert isinstance(ws, LocalWorkspace)
    assert ws.root_path == nested


def test_open_workspace_from_subdir(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "proj")
    sub = repo / "src"
    sub.mkdir()
    ws = open_workspace(str(sub))
    assert isinstance(ws, LocalWorkspace)
    assert ws.root_path == repo
