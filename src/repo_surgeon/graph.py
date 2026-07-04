"""Assemble the LangGraph orchestration graph.

The workspace, config, LLM provider, and test runner are bound into each node via
`functools.partial`, so the graph state stays serializable (issue, locations, patches, …)
while side-effecting resources live in the closure. The write → test → critic cycle is the
self-correction loop; after `write`, an apply failure skips straight to the critic (no point
testing an unapplied patch), and the critic's decision drives the retry/deliver routing.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from repo_surgeon.agents import nodes
from repo_surgeon.config import AppConfig
from repo_surgeon.llm.factory import LLMProvider, build_llm
from repo_surgeon.retrieval import SemanticIndex
from repo_surgeon.sandbox import TestRunner, make_docker_runner
from repo_surgeon.state import RunState
from repo_surgeon.workspace.base import Workspace


def _route_after_write(state: RunState) -> str:
    """Test the patch if it applied cleanly; otherwise go straight to the critic."""
    return "test" if state.get("apply_ok", False) else "critic"


def _route_after_critic(state: RunState) -> str:
    """Map the critic's decision to the next node."""
    decision = state.get("decision", nodes.DECISION_GIVE_UP)
    return {
        nodes.DECISION_RESOLVED: "deliver",
        nodes.DECISION_RETRY: "write",
        nodes.DECISION_REPLAN: "plan",
        nodes.DECISION_GIVE_UP: "report_failure",
    }.get(decision, "report_failure")


def build_graph(
    workspace: Workspace,
    config: AppConfig,
    *,
    llm_provider: LLMProvider = build_llm,
    test_runner: TestRunner | None = None,
    semantic_index: SemanticIndex | None = None,
) -> CompiledStateGraph:
    """Build and compile the orchestration graph for one workspace + config.

    `llm_provider` and `test_runner` are injectable so tests can supply fakes without the
    network or Docker; they default to the real provider factory and Docker sandbox.
    `semantic_index` (optional) enriches the writer's context with similar code.
    """
    runner = test_runner or make_docker_runner(config.sandbox)
    bind = lambda fn: partial(  # noqa: E731
        fn,
        workspace=workspace,
        app_config=config,
        llm_provider=llm_provider,
        test_runner=runner,
        semantic_index=semantic_index,
    )

    g: StateGraph = StateGraph(RunState)
    g.add_node("build_kb", bind(nodes.build_kb))
    g.add_node("localize", bind(nodes.localize))
    g.add_node("plan", bind(nodes.plan))
    g.add_node("write", bind(nodes.write))
    g.add_node("test", bind(nodes.test))
    g.add_node("critic", bind(nodes.critic))
    g.add_node("deliver", bind(nodes.deliver))
    g.add_node("report_failure", bind(nodes.report_failure))

    g.set_entry_point("build_kb")
    g.add_edge("build_kb", "localize")
    g.add_edge("localize", "plan")
    g.add_edge("plan", "write")
    g.add_conditional_edges("write", _route_after_write, {"test": "test", "critic": "critic"})
    g.add_edge("test", "critic")
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "deliver": "deliver",
            "write": "write",
            "plan": "plan",
            "report_failure": "report_failure",
        },
    )
    g.add_edge("deliver", END)
    g.add_edge("report_failure", END)

    return g.compile()
