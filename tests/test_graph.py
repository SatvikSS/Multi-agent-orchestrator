"""End-to-end tests of the graph on a buggy repo, using fake LLMs + a fake runner.

The fake test runner "passes" only when calc.py has been corrected to `a + b`, mirroring
how a real suite would react — so these assert the write → test → critic loop actually
converges on a fix, resetting the tree between attempts. Runs follow the production
lifecycle: start_work() first, fix delivered on the work branch, original branch restored.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from conftest import content_test_runner, make_fake_provider
from repo_surgeon.config import AppConfig
from repo_surgeon.graph import _route_after_critic, _route_after_write, build_graph
from repo_surgeon.issues import LocalIssueSource
from repo_surgeon.models import RunStatus
from repo_surgeon.state import RunState
from repo_surgeon.workspace import LocalWorkspace

_RIGHT = (
    "calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"
)
_WRONG = (
    "calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a * b\n>>>>>>> REPLACE"
)


def _entry(repo: Path) -> RunState:
    issue = LocalIssueSource("Fix add() in calc.py — it should add").fetch()
    return {
        "issue": issue,
        "repo_ref": str(repo),
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


def _provider(writer_responses: list[str]):
    return make_fake_provider(
        {
            "summarizer": ["calc module: add(a, b)"],
            "localizer": ["calc.py"],
            "planner": ["- Make add() return a + b"],
            "writer": writer_responses,
        }
    )


def _run(
    repo: Path, cfg: AppConfig, writer_responses: list[str]
) -> tuple[RunState, LocalWorkspace]:
    ws = LocalWorkspace(repo)
    ws.start_work("Fix add() in calc.py — it should add")
    graph = build_graph(
        ws,
        cfg,
        llm_provider=_provider(writer_responses),
        test_runner=content_test_runner("a + b"),
    )
    return graph.invoke(_entry(repo), config={"recursion_limit": 50}), ws


def _file_at(repo: Path, ref: str, rel_path: str) -> str:
    out = subprocess.run(
        ["git", "show", f"{ref}:{rel_path}"], cwd=repo, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_fix_passes_tests_first_try(buggy_repo: Path, app_config: AppConfig) -> None:
    final, ws = _run(buggy_repo, app_config, [_RIGHT])
    assert final["status"] == RunStatus.RESOLVED
    assert final["attempts"] == 1
    # The fix is committed on the work branch; the original branch is checked out again.
    assert "a + b" in _file_at(buggy_repo, final["delivery_ref"], "calc.py")
    assert ws.current_branch() != final["delivery_ref"]
    assert "a - b" in ws.read_file("calc.py")  # original branch untouched


def test_self_corrects_after_failing_tests(buggy_repo: Path, app_config: AppConfig) -> None:
    # First edit applies but tests still fail (a * b); second edit makes them pass (a + b).
    final, _ = _run(buggy_repo, app_config, [_WRONG, _RIGHT])
    assert final["status"] == RunStatus.RESOLVED
    assert final["attempts"] == 2
    assert "a + b" in _file_at(buggy_repo, final["delivery_ref"], "calc.py")


def test_budget_exhausted_when_tests_never_pass(buggy_repo: Path, app_config: AppConfig) -> None:
    tight_budget = app_config.budget.model_copy(update={"max_attempts": 2})
    tight = app_config.model_copy(update={"budget": tight_budget})
    final, ws = _run(buggy_repo, tight, [_WRONG])
    assert final["status"] == RunStatus.BUDGET_EXHAUSTED
    assert not final.get("delivery_ref")
    # abort_work: bug intact, back on the original branch, work branch deleted.
    assert "a - b" in ws.read_file("calc.py")
    branches = subprocess.run(
        ["git", "branch", "--format=%(refname:short)"],
        cwd=buggy_repo,
        capture_output=True,
        text=True,
    ).stdout
    assert "repo-surgeon/" not in branches


def test_route_after_write() -> None:
    assert _route_after_write({"apply_ok": True}) == "test"
    assert _route_after_write({"apply_ok": False}) == "critic"
    assert _route_after_write({}) == "critic"


def test_route_after_critic_maps_decisions() -> None:
    assert _route_after_critic({"decision": "resolved"}) == "deliver"
    assert _route_after_critic({"decision": "retry"}) == "write"
    assert _route_after_critic({"decision": "replan"}) == "plan"
    assert _route_after_critic({"decision": "give_up"}) == "report_failure"
    assert _route_after_critic({}) == "report_failure"
