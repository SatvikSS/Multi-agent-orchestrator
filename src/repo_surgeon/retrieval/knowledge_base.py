"""Tier-1 retrieval: a cached knowledge base of per-file summaries + candidate routing.

For 10-15 small/medium repos this replaces a heavyweight PageRank repo map: summarize
each source file once (cached), then route an issue to a handful of candidate files using
its path/symbol hints, keyword overlap with the summaries, and ripgrep.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from repo_surgeon.config import AppConfig
from repo_surgeon.llm.factory import LLMProvider
from repo_surgeon.llm.response import content_text
from repo_surgeon.models import Issue
from repo_surgeon.retrieval.grep import grep
from repo_surgeon.workspace.base import Workspace

logger = logging.getLogger(__name__)

_CACHE_DIR = ".repo_surgeon_cache"
_CACHE_FILE = "kb.json"
_MAX_SUMMARY_INPUT = 3000  # chars of file content fed to the summarizer
_SUMMARY_SYSTEM = (
    "You summarize a source file in 1-2 sentences: its purpose and key functions/classes. "
    "Be terse and concrete. No preamble."
)


class KnowledgeBase(BaseModel):
    """Per-file summaries plus a count of how many files exist (for transparency)."""

    model_config = {"frozen": True}

    summaries: dict[str, str]
    total_files: int = 0


def build_knowledge_base(
    workspace: Workspace,
    llm_provider: LLMProvider,
    app_config: AppConfig,
    *,
    max_files: int = 40,
    force: bool = False,
) -> KnowledgeBase:
    """Build (or load from cache) a summary knowledge base for the repo."""
    cache = Path(workspace.root_path) / _CACHE_DIR / _CACHE_FILE
    if cache.exists() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        return KnowledgeBase(**data)

    py_files = workspace.list_files(suffix=".py")
    selected = py_files[:max_files]
    if len(py_files) > max_files:
        logger.warning(
            "Knowledge base capped: summarizing %d of %d files (max_files=%d).",
            len(selected),
            len(py_files),
            max_files,
        )

    llm = llm_provider(app_config, "summarizer")
    summaries: dict[str, str] = {}
    for path in selected:
        content = workspace.read_file(path)[:_MAX_SUMMARY_INPUT]
        response = llm.invoke(
            [
                SystemMessage(content=_SUMMARY_SYSTEM),
                HumanMessage(content=f"File {path}:\n\n{content}"),
            ]
        )
        summaries[path] = content_text(response).strip()

    kb = KnowledgeBase(summaries=summaries, total_files=len(py_files))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(kb.model_dump_json(indent=2), encoding="utf-8")
    return kb


def route_candidates(
    kb: KnowledgeBase,
    issue: Issue,
    workspace: Workspace,
    *,
    limit: int = 5,
) -> list[str]:
    """Choose candidate files for an issue, most-relevant first, capped at `limit`.

    Considers code (.py) plus doc/config files, so non-code tasks (e.g. "update the
    README") can be localized too.
    """
    all_files = set(workspace.list_files())
    py_files = {f for f in all_files if f.endswith(".py")}
    text = f"{issue.title} {issue.body}".lower()
    ordered: list[str] = []

    def add(path: str) -> None:
        if path in all_files and path not in ordered:
            ordered.append(path)

    # 1) Explicit path hints from the issue (any extension).
    for path in issue.hint_paths:
        add(path)

    # 2) Doc-keyword targets (README / changelog / license / docs).
    for path in _doc_targets(all_files, text):
        add(path)

    # 3) Files containing any symbol named in the issue.
    for symbol in issue.hint_symbols:
        for hit in grep(workspace, symbol, max_results=10):
            add(hit.file_path)

    # 4) Keyword overlap between the issue text and each file summary.
    for path in _rank_by_summary_overlap(kb, issue):
        add(path)

    # 5) Fallback for tiny repos: the known code files.
    if not ordered:
        for path in sorted(py_files):
            add(path)

    return ordered[:limit]


_DOC_KEYWORDS = (
    "readme",
    "changelog",
    "license",
    "licence",
    "documentation",
    "docs",
    "contributing",
)


def _doc_targets(all_files: set[str], issue_text: str) -> list[str]:
    """Doc/config files worth surfacing when the issue references them by keyword."""
    from pathlib import PurePosixPath

    targets: list[str] = []
    for keyword in _DOC_KEYWORDS:
        if keyword not in issue_text:
            continue
        for f in sorted(all_files):
            if keyword.rstrip("s") in PurePosixPath(f).name.lower():
                targets.append(f)
    return targets


def _rank_by_summary_overlap(kb: KnowledgeBase, issue: Issue) -> list[str]:
    """Rank summarized files by word overlap with the issue text (descending)."""
    issue_words = _words(f"{issue.title} {issue.body}")
    scored = [
        (path, len(issue_words & _words(summary)))
        for path, summary in kb.summaries.items()
    ]
    scored = [item for item in scored if item[1] > 0]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [path for path, _ in scored]


def _words(text: str) -> set[str]:
    normalized = "".join(c if c.isalnum() else " " for c in text.lower())
    return {w for w in normalized.split() if len(w) > 2}
