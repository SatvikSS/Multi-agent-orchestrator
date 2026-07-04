"""Tests for plain-folder support: shadow-git init and staging mode."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conftest import content_test_runner, make_fake_provider
from repo_surgeon.config import AppConfig
from repo_surgeon.models import RunStatus
from repo_surgeon.runner import run_issue
from repo_surgeon.workspace import LocalWorkspace
from repo_surgeon.workspace.ignore import GITIGNORE_MARKER, iter_project_files
from repo_surgeon.workspace.shadow import init_shadow_git
from repo_surgeon.workspace.staging import StagingWorkspace, apply_patch_file

_RIGHT = (
    "calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ~/.repo-surgeon at a temp dir so tests never touch the real home."""
    monkeypatch.setenv("REPO_SURGEON_HOME", str(tmp_path / "surgeon-home"))


@pytest.fixture
def plain_project(tmp_path: Path) -> Path:
    """A non-git data-science-looking folder: code + data + venv junk."""
    proj = tmp_path / "ds-project"
    proj.mkdir()
    (proj / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (proj / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (proj / "data.csv").write_text("a,b\n1,2\n" * 1000, encoding="utf-8")
    (proj / "model.pkl").write_bytes(b"\x80\x04fake-pickle")
    venv = proj / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "big_dep.py").write_text("x = 1\n", encoding="utf-8")
    return proj


def _provider():
    return make_fake_provider(
        {
            "summarizer": ["calc module"],
            "localizer": ["calc.py"],
            "planner": ["- fix add"],
            "writer": [_RIGHT],
        }
    )


# --- ignore rules -----------------------------------------------------------


def test_iter_project_files_excludes_data_and_junk(plain_project: Path) -> None:
    files = set(iter_project_files(plain_project))
    assert "calc.py" in files
    assert "test_calc.py" in files
    assert "data.csv" not in files
    assert "model.pkl" not in files
    assert not any(".venv" in f for f in files)


def test_iter_project_files_size_cap(plain_project: Path) -> None:
    (plain_project / "huge.py").write_text("# " + "x" * 200_000, encoding="utf-8")
    files = set(iter_project_files(plain_project, max_file_mb=0.1))
    assert "huge.py" not in files
    assert "calc.py" in files


# --- shadow-git -------------------------------------------------------------


def test_init_shadow_git_baseline_excludes_data(plain_project: Path) -> None:
    init_shadow_git(plain_project)
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=plain_project, capture_output=True, text=True
    ).stdout.splitlines()
    assert "calc.py" in tracked
    assert "data.csv" not in tracked
    assert "model.pkl" not in tracked
    assert not any(".venv" in f for f in tracked)
    # It's now a working LocalWorkspace.
    ws = LocalWorkspace(plain_project)
    assert ws.is_dirty() is False


def test_init_shadow_git_appends_to_existing_gitignore(plain_project: Path) -> None:
    (plain_project / ".gitignore").write_text("my-own-pattern/\n", encoding="utf-8")
    init_shadow_git(plain_project)
    content = (plain_project / ".gitignore").read_text(encoding="utf-8")
    assert content.startswith("my-own-pattern/\n")
    assert GITIGNORE_MARKER in content
    assert "*.csv" in content


def test_init_shadow_git_lists_oversized_files(plain_project: Path) -> None:
    (plain_project / "big_script.py").write_text("# " + "x" * 200_000, encoding="utf-8")
    init_shadow_git(plain_project, max_file_mb=0.1)
    content = (plain_project / ".gitignore").read_text(encoding="utf-8")
    assert "/big_script.py" in content


def test_init_shadow_git_refuses_existing_repo(toy_repo: Path) -> None:
    with pytest.raises(ValueError, match="already a git repository"):
        init_shadow_git(toy_repo)


def test_full_run_on_shadow_git_folder(plain_project: Path, app_config: AppConfig) -> None:
    init_shadow_git(plain_project)
    final = run_issue(
        "Fix add() in calc.py",
        str(plain_project),
        config=app_config,
        llm_provider=_provider(),
        test_runner=content_test_runner("a + b"),
    )
    assert final["status"] == RunStatus.RESOLVED
    assert final["delivery_ref"].startswith("repo-surgeon/")


# --- staging mode ------------------------------------------------------------


def test_staging_copies_code_only(plain_project: Path) -> None:
    ws = StagingWorkspace(plain_project)
    staged = set(ws.list_files())
    assert "calc.py" in staged
    assert "data.csv" not in staged
    assert (ws.root_path / "calc.py").exists()
    # Original folder untouched: no .git created there.
    assert not (plain_project / ".git").exists()


def test_staging_run_produces_patch_and_applies_back(
    plain_project: Path, app_config: AppConfig
) -> None:
    ws = StagingWorkspace(plain_project)
    final = run_issue(
        "Fix add() in calc.py",
        str(plain_project),
        config=app_config,
        llm_provider=_provider(),
        test_runner=content_test_runner("a + b"),
        workspace=ws,
    )
    assert final["status"] == RunStatus.RESOLVED
    patch_path = final["delivery_ref"]
    assert patch_path.endswith(".patch")
    assert Path(patch_path).is_file()
    # Original still has the bug until the patch is applied.
    assert "a - b" in (plain_project / "calc.py").read_text(encoding="utf-8")

    touched = apply_patch_file(patch_path, plain_project)
    assert touched == ["calc.py"]
    assert "a + b" in (plain_project / "calc.py").read_text(encoding="utf-8")


def test_apply_creates_backups(plain_project: Path, app_config: AppConfig) -> None:
    ws = StagingWorkspace(plain_project)
    final = run_issue(
        "Fix add() in calc.py",
        str(plain_project),
        config=app_config,
        llm_provider=_provider(),
        test_runner=content_test_runner("a + b"),
        workspace=ws,
    )
    apply_patch_file(final["delivery_ref"], plain_project)
    from repo_surgeon.home import home_dir

    backups = list((home_dir() / "backups").rglob("calc.py"))
    assert backups, "expected a backup of calc.py before patching"
    assert "a - b" in backups[0].read_text(encoding="utf-8")  # backup holds the original


def test_apply_bad_patch_reports_error(plain_project: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.patch"
    bad.write_text(
        "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n-not in file\n+replacement\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not apply cleanly"):
        apply_patch_file(bad, plain_project)
