"""LangGraph state schema for an orchestration run.

State flows through every node. Nodes return partial dicts that LangGraph merges;
list-valued fields use an additive reducer so successive attempts accumulate history.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from repo_surgeon.models import Issue, Location, Patch, RunStatus, TestResult


class RunState(TypedDict, total=False):
    """Shared state for one issue-resolution run."""

    # Inputs (set once at graph entry).
    issue: Issue
    repo_ref: str  # local path or GitHub "owner/repo"

    # Retrieval / planning outputs.
    kb_route: list[str]  # candidate folders/files chosen by the knowledge-base router
    locations: list[Location]
    plan: str

    # Edit / test cycle. `patches` and `test_results` accumulate across attempts.
    patches: Annotated[list[Patch], operator.add]
    test_results: Annotated[list[TestResult], operator.add]
    attempts: int
    apply_ok: bool  # did the latest attempt's patches apply cleanly?
    writer_feedback: str  # feedback for the next write attempt (apply failure or test failures)

    # Budget accounting.
    tokens_spent: int
    cost_usd: float

    # Control + delivery.
    status: RunStatus
    decision: str  # critic's routing decision; the conditional edge dispatches on this
    delivery_ref: str  # branch name or PR URL
    notes: Annotated[list[str], operator.add]  # human-readable trace of what each node did
