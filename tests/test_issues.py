"""Tests for issue-source resolution and hint extraction."""

from __future__ import annotations

from pathlib import Path

from repo_surgeon.issues import LocalIssueSource, open_issue_source
from repo_surgeon.issues.base import extract_hints
from repo_surgeon.issues.github import GitHubIssueSource


def test_extract_hints_finds_paths_and_symbols() -> None:
    text = "The bug is in pkg/utils.py: paginate() returns one extra row. See def paginate."
    paths, symbols = extract_hints(text)
    assert "pkg/utils.py" in paths
    assert "paginate" in symbols


def test_local_issue_from_string() -> None:
    issue = LocalIssueSource("Fix off-by-one in paginate()").fetch()
    assert issue.title == "Fix off-by-one in paginate()"
    assert issue.source == "local"
    assert "paginate" in issue.hint_symbols


def test_local_issue_from_file(tmp_path: Path) -> None:
    f = tmp_path / "issue.md"
    f.write_text("Crash on empty config\n\nRaised in config.py loader.", encoding="utf-8")
    issue = LocalIssueSource(str(f)).fetch()
    assert issue.title == "Crash on empty config"
    assert "config.py" in issue.hint_paths


def test_factory_routes_github_url_to_github_source() -> None:
    src = open_issue_source("https://github.com/octocat/hello/issues/42")
    assert isinstance(src, GitHubIssueSource)


def test_factory_routes_plain_text_to_local_source() -> None:
    src = open_issue_source("just a description")
    assert isinstance(src, LocalIssueSource)
