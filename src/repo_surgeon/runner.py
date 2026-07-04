"""Orchestration entry point: resolve inputs, run the graph, return the final state.

This is the seam between side-effecting resolution (issue fetch, workspace checkout) and
the pure-ish graph. The CLI and the Streamlit UI both call `run_issue`.

Safety sequence: preflight (refuse a dirty tree) → create the work branch BEFORE any
edit → run the graph → on success the fix lives on that branch and the user's original
branch is checked out again; on failure or crash, edits are discarded and the work
branch is deleted, leaving the checkout exactly as it started.
"""

from __future__ import annotations

from typing import cast

from langchain_core.runnables import RunnableConfig

from repo_surgeon.config import AppConfig, load_config
from repo_surgeon.graph import build_graph
from repo_surgeon.issues import open_issue_source
from repo_surgeon.llm.embeddings import build_embeddings
from repo_surgeon.llm.factory import LLMProvider, build_llm
from repo_surgeon.models import RunStatus
from repo_surgeon.retrieval import build_semantic_index
from repo_surgeon.sandbox import TestRunner
from repo_surgeon.state import RunState
from repo_surgeon.workspace import open_workspace
from repo_surgeon.workspace.base import DirtyWorkspaceError, Workspace

# Backstop on total graph steps, above the attempt budget, in case a cycle misbehaves.
_RECURSION_LIMIT = 50


def run_issue(
    issue_ref: str,
    repo_ref: str,
    *,
    config: AppConfig | None = None,
    allow_dirty: bool = False,
    require_tests: bool | None = None,
    llm_provider: LLMProvider = build_llm,
    test_runner: TestRunner | None = None,
    workspace: Workspace | None = None,
) -> RunState:
    """Resolve `issue_ref` against `repo_ref` and run the full orchestration graph.

    `workspace` overrides the default resolution — used for staging mode, where the
    caller has already built an isolated copy to work in.
    """
    cfg = config or load_config()
    token = cfg.secrets.github_token
    verify = cfg.require_tests if require_tests is None else require_tests

    workspace = workspace or open_workspace(repo_ref, github_token=token)
    try:
        issue = open_issue_source(issue_ref, github_token=token).fetch()

        if not allow_dirty and workspace.is_dirty():
            raise DirtyWorkspaceError(
                f"'{repo_ref}' has uncommitted changes to tracked files. The retry loop "
                f"resets tracked files between attempts, which would destroy them. "
                f"Commit or stash your changes first, or pass --allow-dirty to override."
            )

        work_branch = workspace.start_work(issue.title)

        semantic_index = None
        if cfg.semantic.enabled:
            try:
                embeddings = build_embeddings(cfg)
                semantic_index = build_semantic_index(workspace, embeddings, cfg.semantic)
            except Exception:  # noqa: BLE001 - semantic search is optional; never block a run
                semantic_index = None

        entry: RunState = {
            "issue": issue,
            "repo_ref": repo_ref,
            "attempts": 0,
            "tokens_spent": 0,
            "cost_usd": 0.0,
            "status": RunStatus.IN_PROGRESS,
            "patches": [],
            "test_results": [],
            "apply_ok": False,
            "writer_feedback": "",
            "require_tests": verify,
            "notes": [f"[runner] working on branch '{work_branch}'"],
        }

        graph = build_graph(
            workspace,
            cfg,
            llm_provider=llm_provider,
            test_runner=test_runner,
            semantic_index=semantic_index,
        )
        # run_name/tags/metadata flow into LangSmith when tracing is enabled
        # (LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY in the environment).
        invoke_config: RunnableConfig = {
            "recursion_limit": _RECURSION_LIMIT,
            "run_name": f"repo-surgeon: {issue.title[:60]}",
            "tags": ["repo-surgeon"],
            "metadata": {"repo": repo_ref, "issue_source": issue.source},
        }
        try:
            result = graph.invoke(entry, config=invoke_config)
        except BaseException:
            # Crash mid-run: discard edits, return to the original branch, drop the work branch.
            workspace.abort_work()
            raise
        return cast(RunState, result)
    finally:
        workspace.cleanup()
