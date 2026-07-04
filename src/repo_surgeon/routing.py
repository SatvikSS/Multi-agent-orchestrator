"""Route an issue to the right project using the registry — no --repo needed.

Two stages, cheap first:
  1. Deterministic scoring over registry entries: project-name tokens matching the
     issue text weigh most, then summary keyword overlap, plus a bonus when a file
     path mentioned in the issue actually exists inside the project.
  2. Optional LLM pick among the top candidates (used when available; scoring order
     is the fallback). The CLI always confirms the choice with the user before running.

Containers are skipped (their nested repos are routed instead) and stale duplicate
clones (is_canonical=False) are never routed to.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from repo_surgeon.llm.response import content_text
from repo_surgeon.models import Issue
from repo_surgeon.registry import ProjectEntry, Registry

logger = logging.getLogger(__name__)

_NAME_TOKEN_WEIGHT = 3
_SUMMARY_TOKEN_WEIGHT = 1
_HINT_PATH_BONUS = 5

_PICK_SYSTEM = (
    "You route a software issue to the project it belongs to. Given the issue and a "
    "numbered list of candidate projects, answer with ONLY the number of the best match."
)


class ScoredProject(BaseModel):
    """A routing candidate with its deterministic score."""

    model_config = {"frozen": True}

    entry: ProjectEntry
    score: int


def route_issue(issue: Issue, registry: Registry, *, limit: int = 5) -> list[ScoredProject]:
    """Rank routable projects for this issue, best first. Zero-score entries are dropped
    unless nothing scores, in which case the (alphabetical) head of the list is returned
    so the caller can still present choices."""
    routable = [
        p
        for p in registry.projects
        if p.kind != "container" and p.is_canonical
    ]
    issue_tokens = _tokens(f"{issue.title} {issue.body}")

    scored = [
        ScoredProject(entry=entry, score=_score(entry, issue_tokens, issue))
        for entry in routable
    ]
    positive = sorted(
        (s for s in scored if s.score > 0), key=lambda s: (-s.score, s.entry.name)
    )
    if positive:
        return positive[:limit]
    return sorted(scored, key=lambda s: s.entry.name)[:limit]


def pick_with_llm(
    issue: Issue, candidates: list[ScoredProject], llm: BaseChatModel
) -> ScoredProject:
    """Ask the LLM to pick among candidates; fall back to the top-scored on any failure."""
    if len(candidates) == 1:
        return candidates[0]
    listing = "\n".join(
        f"{i + 1}. {c.entry.name} ({c.entry.kind}, {c.entry.py_files} py files)"
        f" — {c.entry.summary or 'no summary'}"
        for i, c in enumerate(candidates)
    )
    prompt = f"Issue: {issue.title}\n\n{issue.body}\n\nCandidate projects:\n{listing}"
    try:
        response = llm.invoke(
            [SystemMessage(content=_PICK_SYSTEM), HumanMessage(content=prompt)]
        )
        match = re.search(r"\d+", content_text(response))
        if match:
            index = int(match.group()) - 1
            if 0 <= index < len(candidates):
                return candidates[index]
    except Exception as exc:  # noqa: BLE001 - routing must degrade, not die
        logger.warning("LLM routing failed, using top-scored candidate: %s", exc)
    return candidates[0]


def _score(entry: ProjectEntry, issue_tokens: set[str], issue: Issue) -> int:
    score = 0
    name_tokens = _tokens(entry.name)
    score += _NAME_TOKEN_WEIGHT * len(name_tokens & issue_tokens)
    if entry.summary:
        score += _SUMMARY_TOKEN_WEIGHT * len(_tokens(entry.summary) & issue_tokens)
    root = Path(entry.path)
    for hint in issue.hint_paths:
        if _hint_exists(root, hint):
            score += _HINT_PATH_BONUS
    return score


def _hint_exists(root: Path, hint: str) -> bool:
    """Cheap, bounded check: exact path, or the bare filename up to two levels deep.
    (No recursive glob — some project folders are gigabytes.)"""
    name = Path(hint).name
    try:
        return (
            (root / hint).exists()
            or next(root.glob(name), None) is not None
            or next(root.glob(f"*/{name}"), None) is not None
            or next(root.glob(f"*/*/{name}"), None) is not None
        )
    except OSError:
        return False


# Split camelCase boundaries so 'chillerPerformanceCalculator' matches "chiller performance".
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokens(text: str) -> set[str]:
    spaced = _CAMEL_BOUNDARY.sub(" ", text)
    normalized = "".join(c if c.isalnum() else " " for c in spaced.lower())
    # Keep words >2 chars, plus short version-ish tokens like 'v2' (len 2 with a digit).
    return {
        w
        for w in normalized.split()
        if len(w) > 2 or (len(w) == 2 and any(c.isdigit() for c in w))
    }
