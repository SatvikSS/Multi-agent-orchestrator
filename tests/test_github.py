"""Tests for GitHub integration — ref parsing, issue fetch, and the clone→push→PR flow.

No network: the workspace clones from a local bare repository and PRs are created via a
fake PyGithub client. The bare repo doubles as 'origin' so pushes are verifiable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import content_test_runner, make_fake_provider
from repo_surgeon.config import AppConfig
from repo_surgeon.issues.github import GitHubIssueSource, parse_issue_ref
from repo_surgeon.models import RunStatus
from repo_surgeon.runner import run_issue
from repo_surgeon.workspace.github import GitHubWorkspace, parse_repo_ref

_RIGHT = (
    "calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"
)


# --- ref parsing --------------------------------------------------------------


def test_parse_issue_ref_url() -> None:
    assert parse_issue_ref("https://github.com/octo/hello/issues/42") == ("octo/hello", 42)


def test_parse_issue_ref_shorthand() -> None:
    assert parse_issue_ref("octo/hello#7") == ("octo/hello", 7)


def test_parse_issue_ref_rejects_non_issue() -> None:
    assert parse_issue_ref("just a description of a bug") is None
    assert parse_issue_ref("https://github.com/octo/hello") is None


def test_parse_repo_ref_forms() -> None:
    assert parse_repo_ref("octo/hello") == "octo/hello"
    assert parse_repo_ref("https://github.com/octo/hello") == "octo/hello"
    assert parse_repo_ref("https://github.com/octo/hello.git") == "octo/hello"
    assert parse_repo_ref("git@github.com:octo/hello.git") == "octo/hello"
    assert parse_repo_ref("/some/local/path") is None


# --- issue source ---------------------------------------------------------------


class _FakeGhIssue:
    title = "Fix off-by-one in paginate()"
    body = "The bug is in pager.py — paginate() returns one extra row."

    def get_comments(self):
        return [SimpleNamespace(body="Stack trace points at line 2 of pager.py")]


class _FakeGhForIssues:
    def get_repo(self, name: str):
        assert name == "octo/hello"
        return SimpleNamespace(get_issue=lambda n: _FakeGhIssue())


def test_github_issue_source_fetch_normalizes() -> None:
    source = GitHubIssueSource("octo/hello#42", client=_FakeGhForIssues())
    issue = source.fetch()
    assert issue.title == "Fix off-by-one in paginate()"
    assert issue.source == "github:octo/hello#42"
    assert "Stack trace" in issue.body  # comments folded in
    assert "pager.py" in issue.hint_paths
    assert "paginate" in issue.hint_symbols


def test_github_issue_source_bad_ref_raises() -> None:
    with pytest.raises(ValueError, match="Not a GitHub issue reference"):
        GitHubIssueSource("not-an-issue", client=_FakeGhForIssues()).fetch()


# --- workspace: clone → push → PR against a local bare origin ---------------------


class _FakePrClient:
    def __init__(self) -> None:
        self.created: dict | None = None

    def get_repo(self, name: str):
        def create_pull(**kwargs):
            self.created = kwargs
            return SimpleNamespace(html_url="https://github.com/octo/hello/pull/1")

        return SimpleNamespace(create_pull=create_pull)


@pytest.fixture
def bare_origin(tmp_path: Path) -> tuple[Path, str]:
    """A local bare repo seeded with a buggy calc.py, acting as GitHub 'origin'."""
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)

    seed = tmp_path / "seed"
    seed.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=seed, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@t.co")
    git("config", "user.name", "T")
    (seed / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (seed / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    default_branch = git("rev-parse", "--abbrev-ref", "HEAD")
    git("remote", "add", "origin", str(bare))
    git("push", "-q", "origin", default_branch)
    subprocess.run(
        ["git", "--git-dir", str(bare), "symbolic-ref", "HEAD", f"refs/heads/{default_branch}"],
        check=True,
    )
    return bare, default_branch


def _provider():
    return make_fake_provider(
        {
            "summarizer": ["calc module"],
            "localizer": ["calc.py"],
            "planner": ["- fix add"],
            "writer": [_RIGHT],
        }
    )


def test_full_github_flow_clone_fix_push_pr(
    bare_origin: tuple[Path, str], app_config: AppConfig
) -> None:
    bare, default_branch = bare_origin
    pr_client = _FakePrClient()
    ws = GitHubWorkspace("octo/hello", clone_url=str(bare), github_client=pr_client)

    final = run_issue(
        "Fix add() in calc.py",
        "octo/hello",
        config=app_config,
        llm_provider=_provider(),
        test_runner=content_test_runner("a + b"),
        workspace=ws,
    )

    assert final["status"] == RunStatus.RESOLVED
    assert final["delivery_ref"] == "https://github.com/octo/hello/pull/1"
    # The PR targets the default branch from the work branch.
    assert pr_client.created is not None
    assert pr_client.created["base"] == default_branch
    assert pr_client.created["head"].startswith("repo-surgeon/")
    # The work branch was actually pushed to origin with the fix on it.
    shown = subprocess.run(
        ["git", "--git-dir", str(bare), "show", f"{pr_client.created['head']}:calc.py"],
        capture_output=True,
        text=True,
    ).stdout
    assert "a + b" in shown


def test_deliver_without_client_pushes_and_returns_branch(
    bare_origin: tuple[Path, str],
) -> None:
    bare, _ = bare_origin
    ws = GitHubWorkspace("octo/hello", clone_url=str(bare), github_client=None)
    try:
        ws.start_work("manual pr fix")
        ref = ws.deliver(title="fix: manual", body="no token available")
        assert ref.startswith("repo-surgeon/")  # branch name, not a PR URL
        branches = subprocess.run(
            ["git", "--git-dir", str(bare), "branch", "--format=%(refname:short)"],
            capture_output=True,
            text=True,
        ).stdout
        assert ref in branches
    finally:
        ws.cleanup()


def test_cleanup_removes_clone(bare_origin: tuple[Path, str]) -> None:
    bare, _ = bare_origin
    ws = GitHubWorkspace("octo/hello", clone_url=str(bare))
    clone_root = ws.root_path
    assert clone_root.exists()
    ws.cleanup()
    assert not clone_root.exists()


def test_bad_clone_raises_clean_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Could not clone"):
        GitHubWorkspace("octo/hello", clone_url=str(tmp_path / "nope.git"))