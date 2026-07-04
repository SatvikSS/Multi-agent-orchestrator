"""Tests for doc/non-test task support: non-.py localization, file creation, no-verify."""

from __future__ import annotations

import subprocess
from pathlib import Path

from conftest import make_fake_provider
from repo_surgeon.editing import apply_patches, parse_search_replace
from repo_surgeon.graph import _route_after_write, build_graph
from repo_surgeon.issues import LocalIssueSource
from repo_surgeon.issues.base import extract_hints
from repo_surgeon.models import Patch, RunStatus
from repo_surgeon.retrieval import KnowledgeBase, route_candidates
from repo_surgeon.state import RunState
from repo_surgeon.workspace import LocalWorkspace

# --- broader hint extraction & routing --------------------------------------


def test_extract_hints_catches_doc_files() -> None:
    paths, _ = extract_hints("Please update README.md and config.yaml with the new info.")
    assert "README.md" in paths
    assert "config.yaml" in paths


def test_route_surfaces_readme_on_keyword(toy_repo: Path) -> None:
    (toy_repo / "README.md").write_text("# old\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=toy_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "add readme"], cwd=toy_repo, check=True, capture_output=True
    )
    ws = LocalWorkspace(toy_repo)
    kb = KnowledgeBase(summaries={}, total_files=3)
    issue = LocalIssueSource("the readme is out of date, update it").fetch()
    assert "README.md" in route_candidates(kb, issue, ws)


# --- whole-file create / rewrite --------------------------------------------


def test_empty_search_creates_new_file(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    patch = Patch(file_path="docs/NOTES.md", search="", replace="# Notes\n\nHello.")
    result = apply_patches(ws, [patch])
    assert result.ok is True
    assert (toy_repo / "docs" / "NOTES.md").read_text(encoding="utf-8").startswith("# Notes")


def test_empty_search_rewrites_whole_file(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    (toy_repo / "README.md").write_text("# old content\n", encoding="utf-8")
    block = "README.md\n<<<<<<< SEARCH\n=======\n# New Title\n\nFresh docs.\n>>>>>>> REPLACE"
    result = apply_patches(ws, parse_search_replace(block))
    assert result.ok is True
    assert ws.read_file("README.md") == "# New Title\n\nFresh docs.\n"


# --- no-verify routing + delivery -------------------------------------------


def test_route_after_write_no_verify_delivers() -> None:
    assert _route_after_write({"apply_ok": True, "require_tests": False}) == "deliver"
    assert _route_after_write({"apply_ok": True, "require_tests": True}) == "test"
    assert _route_after_write({"apply_ok": False, "require_tests": False}) == "critic"


def test_graph_no_verify_delivers_unverified(toy_repo: Path, app_config) -> None:
    (toy_repo / "README.md").write_text("# old\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=toy_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "readme"], cwd=toy_repo, check=True, capture_output=True
    )
    ws = LocalWorkspace(toy_repo)
    ws.start_work("update the README")

    block = (
        "README.md\n<<<<<<< SEARCH\n=======\n# Calc\n\nA tiny calculator library.\n>>>>>>> REPLACE"
    )
    provider = make_fake_provider(
        {
            "summarizer": ["calc"],
            "localizer": ["README.md"],
            "planner": ["- rewrite README with project info"],
            "writer": [block],
        }
    )

    def _must_not_run(_ws):
        raise AssertionError("test runner must be skipped in no-verify mode")

    graph = build_graph(ws, app_config, llm_provider=provider, test_runner=_must_not_run)
    entry: RunState = {
        "issue": LocalIssueSource("the README is not updated, update it with project info").fetch(),
        "repo_ref": str(toy_repo),
        "attempts": 0,
        "tokens_spent": 0,
        "cost_usd": 0.0,
        "status": RunStatus.IN_PROGRESS,
        "patches": [],
        "test_results": [],
        "apply_ok": False,
        "writer_feedback": "",
        "require_tests": False,
        "notes": [],
    }
    final = graph.invoke(entry, config={"recursion_limit": 50})

    assert final["status"] == RunStatus.RESOLVED
    assert any("UNVERIFIED" in n for n in final["notes"])
    shown = subprocess.run(
        ["git", "show", f"{final['delivery_ref']}:README.md"],
        cwd=toy_repo,
        capture_output=True,
        text=True,
    ).stdout
    assert "A tiny calculator library." in shown
