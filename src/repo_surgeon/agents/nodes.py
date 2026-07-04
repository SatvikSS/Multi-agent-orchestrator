"""Graph nodes wiring the real agents, retrieval, and Docker test sandbox together.

Each node takes the run state plus bound dependencies (workspace, config, llm_provider,
test_runner) and returns a partial state update (never mutating the incoming state).

The self-correction loop is now real: `write` resets the tree and applies fresh edits,
`test` runs the repo's suite in Docker, and `critic` feeds either an apply failure or the
actual test failures back to the writer until the suite is green or the budget runs out.
"""

from __future__ import annotations

from repo_surgeon.agents.context import file_context
from repo_surgeon.agents.localizer import localize as run_localizer
from repo_surgeon.agents.planner import make_plan
from repo_surgeon.agents.writer import write_patches
from repo_surgeon.config import AppConfig
from repo_surgeon.editing import apply_patches
from repo_surgeon.llm.factory import LLMProvider
from repo_surgeon.models import Location, RunStatus, TestResult
from repo_surgeon.retrieval import (
    SemanticIndex,
    build_knowledge_base,
    related_code_context,
    route_candidates,
)
from repo_surgeon.sandbox import TestRunner
from repo_surgeon.state import RunState
from repo_surgeon.workspace.base import Workspace

# Decision labels the critic emits; the graph's conditional edge routes on these.
DECISION_RESOLVED = "resolved"
DECISION_RETRY = "retry"
DECISION_REPLAN = "replan"
DECISION_GIVE_UP = "give_up"

_MAX_FEEDBACK_FAILURES = 10


def build_kb(
    state: RunState,
    *,
    workspace: Workspace,
    app_config: AppConfig,
    llm_provider: LLMProvider,
    test_runner: TestRunner,
    semantic_index: SemanticIndex | None,
) -> dict:
    """Build/load the knowledge base and route the issue to candidate files."""
    kb = build_knowledge_base(workspace, llm_provider, app_config)
    route = route_candidates(kb, state["issue"], workspace)
    return {
        "kb_route": route,
        "notes": [
            f"[build_kb] {len(kb.summaries)}/{kb.total_files} files summarized; "
            f"routed to {route or '∅'}"
        ],
    }


def localize(
    state: RunState,
    *,
    workspace: Workspace,
    app_config: AppConfig,
    llm_provider: LLMProvider,
    test_runner: TestRunner,
    semantic_index: SemanticIndex | None,
) -> dict:
    """Pick the file(s) to change from the routed candidates."""
    candidates = state.get("kb_route", [])
    context = file_context(workspace, candidates, numbered=True)
    llm = llm_provider(app_config, "localizer")
    chosen = run_localizer(state["issue"], candidates, context, llm)
    locations = [Location(file_path=p, reason="localizer selection") for p in chosen]
    return {"locations": locations, "notes": [f"[localize] selected {chosen or '∅'}"]}


def plan(
    state: RunState,
    *,
    workspace: Workspace,
    app_config: AppConfig,
    llm_provider: LLMProvider,
    test_runner: TestRunner,
    semantic_index: SemanticIndex | None,
) -> dict:
    """Draft a concrete fix plan from the located code."""
    paths = [loc.file_path for loc in state.get("locations", [])]
    context = file_context(workspace, paths, numbered=True)
    llm = llm_provider(app_config, "planner")
    fix_plan = make_plan(state["issue"], context, llm)
    return {"plan": fix_plan, "notes": ["[plan] drafted fix plan"]}


def write(
    state: RunState,
    *,
    workspace: Workspace,
    app_config: AppConfig,
    llm_provider: LLMProvider,
    test_runner: TestRunner,
    semantic_index: SemanticIndex | None,
) -> dict:
    """Reset the tree, generate search-replace edits, and apply them transactionally.

    Resetting first makes each attempt independent: the writer always edits the original
    code (plus feedback from the previous attempt), never a half-patched tree.
    """
    workspace.reset()
    attempts = state.get("attempts", 0) + 1
    paths = [loc.file_path for loc in state.get("locations", [])]
    context = file_context(workspace, paths, numbered=False)  # raw text → exact SEARCH match

    # Optional semantic enrichment: similar code elsewhere (call sites, patterns to match).
    query = f"{state['issue'].title}\n{state.get('plan', '')}"
    related = related_code_context(semantic_index, query, app_config.semantic.top_k)
    if related:
        context = f"{context}\n\n{related}"

    llm = llm_provider(app_config, "writer")
    patches = write_patches(
        state["issue"],
        state.get("plan", ""),
        context,
        llm,
        feedback=state.get("writer_feedback", ""),
    )
    result = apply_patches(workspace, patches)
    verb = "applied" if result.ok else "failed to apply"
    enriched = " (+semantic context)" if related else ""
    return {
        "patches": patches,
        "apply_ok": result.ok,
        "writer_feedback": result.feedback(),  # empty when ok; apply guidance when not
        "attempts": attempts,
        "notes": [f"[write] attempt {attempts}: {len(patches)} edit(s) {verb}{enriched}"],
    }


def test(
    state: RunState,
    *,
    workspace: Workspace,
    app_config: AppConfig,
    llm_provider: LLMProvider,
    test_runner: TestRunner,
    semantic_index: SemanticIndex | None,
) -> dict:
    """Run the target repo's test suite (Docker sandbox) against the patched tree."""
    result = test_runner(workspace)
    passed_n = result.total - result.failed
    summary = f"{passed_n}/{result.total} passed" if result.total else "no tests"
    return {
        "test_results": [result],
        "notes": [f"[test] {'passed' if result.passed else 'failed'} ({summary})"],
    }


def critic(
    state: RunState,
    *,
    workspace: Workspace,
    app_config: AppConfig,
    llm_provider: LLMProvider,
    test_runner: TestRunner,
    semantic_index: SemanticIndex | None,
) -> dict:
    """Decide next step and, on retry, feed the failure back to the writer."""
    attempts = state.get("attempts", 0)
    budget = app_config.budget.max_attempts

    # Patch never applied → the write node already set apply guidance in writer_feedback.
    if not state.get("apply_ok", False):
        return _decision(DECISION_RETRY if attempts < budget else DECISION_GIVE_UP, attempts)

    results = state.get("test_results", [])
    if results and results[-1].passed:
        return _decision(DECISION_RESOLVED, attempts)
    if attempts >= budget:
        return _decision(DECISION_GIVE_UP, attempts)

    return _decision(DECISION_RETRY, attempts, feedback=_failure_feedback(results))


def deliver(
    state: RunState,
    *,
    workspace: Workspace,
    app_config: AppConfig,
    llm_provider: LLMProvider,
    test_runner: TestRunner,
    semantic_index: SemanticIndex | None,
) -> dict:
    """Finalize a resolved run: commit on the work branch and return to the original branch."""
    issue = state["issue"]
    verified = state.get("require_tests", True)
    tag = "" if verified else " (UNVERIFIED — no tests run; review the diff)"
    ref = workspace.deliver(
        title=f"fix: {issue.title}",
        body=(
            f"Automated fix by repo-surgeon{tag}.\n\nPlan:\n{state.get('plan', '')}"
        ),
    )
    return {
        "delivery_ref": ref,
        "status": RunStatus.RESOLVED,
        "notes": [f"[deliver] delivered on '{ref}'{tag}"],
    }


def report_failure(
    state: RunState,
    *,
    workspace: Workspace,
    app_config: AppConfig,
    llm_provider: LLMProvider,
    test_runner: TestRunner,
    semantic_index: SemanticIndex | None,
) -> dict:
    """Terminal node when the budget is exhausted without a green test run."""
    # Discard edits, return to the original branch, delete the work branch.
    workspace.abort_work()
    return {
        "status": RunStatus.BUDGET_EXHAUSTED,
        "notes": ["[report_failure] budget exhausted; edits discarded, work branch removed"],
    }


def _decision(decision: str, attempts: int, *, feedback: str | None = None) -> dict:
    status = {
        DECISION_RESOLVED: RunStatus.RESOLVED,
        DECISION_GIVE_UP: RunStatus.BUDGET_EXHAUSTED,
    }.get(decision, RunStatus.IN_PROGRESS)
    update = {
        "status": status,
        "decision": decision,
        "notes": [f"[critic] decision={decision} (attempt {attempts})"],
    }
    if feedback is not None:
        update["writer_feedback"] = feedback
    return update


def _failure_feedback(results: list[TestResult]) -> str:
    if not results:
        return "Tests did not pass. Revisit the fix."
    failures = results[-1].failures[:_MAX_FEEDBACK_FAILURES]
    listed = "\n".join(f"  - {f}" for f in failures)
    return (
        f"The test suite still fails. Failing tests:\n{listed}\n"
        "Revise the edit to make them pass."
    )
