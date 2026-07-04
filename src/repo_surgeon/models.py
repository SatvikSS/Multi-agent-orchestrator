"""Core domain models shared across agents and the graph.

All models are immutable (`frozen=True`) so state updates create new objects rather
than mutating in place — this keeps the LangGraph state easy to reason about.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    """Terminal and in-flight status of an orchestration run."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


class Issue(BaseModel):
    """A normalized problem statement, regardless of source (GitHub / local)."""

    model_config = {"frozen": True}

    title: str
    body: str
    source: str = Field(description="Origin, e.g. 'github:owner/repo#12' or 'local'.")
    hint_paths: tuple[str, ...] = Field(
        default=(),
        description="File paths mentioned in the issue or extracted from stack traces.",
    )
    hint_symbols: tuple[str, ...] = Field(
        default=(),
        description="Symbols (functions/classes) mentioned in the issue text.",
    )


class Location(BaseModel):
    """A suspect code location produced by the localizer."""

    model_config = {"frozen": True}

    file_path: str
    symbol: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    reason: str = ""


class Patch(BaseModel):
    """A single search-replace edit against one file."""

    model_config = {"frozen": True}

    file_path: str
    search: str = Field(description="Exact text to find in the file.")
    replace: str = Field(description="Text to replace it with.")


class TestResult(BaseModel):
    """Outcome of running the target repo's test suite in the sandbox."""

    model_config = {"frozen": True}

    passed: bool
    total: int = 0
    failed: int = 0
    failures: tuple[str, ...] = Field(default=(), description="Per-test failure summaries.")
    raw_output: str = ""
