"""Tests for grep, reader, and knowledge-base routing (no LLM)."""

from __future__ import annotations

from pathlib import Path

from repo_surgeon.issues import LocalIssueSource
from repo_surgeon.retrieval import KnowledgeBase, grep, read_numbered, route_candidates
from repo_surgeon.workspace import LocalWorkspace


def test_grep_finds_symbol(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    hits = grep(ws, "def add")
    assert any(h.file_path == "calc.py" for h in hits)
    assert all(h.line > 0 for h in hits)


def test_grep_no_match_returns_empty(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    assert grep(ws, "zzz_no_such_symbol") == []


def test_read_numbered_range(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    out = read_numbered(ws, "calc.py", start=1, end=1)
    assert out.startswith("1 | ")
    assert "def add" in out


def test_route_candidates_uses_hint_paths(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    kb = KnowledgeBase(summaries={"calc.py": "adds two numbers via add()"}, total_files=2)
    issue = LocalIssueSource("Bug in calc.py add()").fetch()
    candidates = route_candidates(kb, issue, ws)
    assert "calc.py" in candidates


def test_route_candidates_fallback_to_all_files(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    kb = KnowledgeBase(summaries={}, total_files=2)
    # An issue with no path/symbol hints and empty KB falls back to listing repo files.
    issue = LocalIssueSource("something is wrong somewhere").fetch()
    candidates = route_candidates(kb, issue, ws)
    assert "calc.py" in candidates
