"""File reader that returns line-numbered context for agents and prompts."""

from __future__ import annotations

from repo_surgeon.workspace.base import Workspace


def read_numbered(
    workspace: Workspace,
    rel_path: str,
    *,
    start: int | None = None,
    end: int | None = None,
) -> str:
    """Return the file (or a line range) with 1-based line-number gutters.

    Line numbers help the agent reason about locations and help you eyeball the context;
    the writer strips them and edits against the real text via SEARCH/REPLACE.
    """
    lines = workspace.read_file(rel_path).splitlines()
    lo = max(1, start or 1)
    hi = min(len(lines), end or len(lines))
    width = len(str(hi))
    return "\n".join(f"{n:>{width}} | {lines[n - 1]}" for n in range(lo, hi + 1))
