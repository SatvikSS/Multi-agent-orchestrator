"""Localizer agent: pick the file(s) that need changing from the candidate set."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from repo_surgeon.agents.prompts import LOCALIZER_SYSTEM, localizer_prompt
from repo_surgeon.llm.response import content_text
from repo_surgeon.models import Issue


def localize(
    issue: Issue,
    candidate_paths: list[str],
    context: str,
    llm: BaseChatModel,
) -> list[str]:
    """Return the chosen file paths (validated against candidates), most-likely first."""
    if not candidate_paths:
        return []
    response = llm.invoke(
        [
            SystemMessage(content=LOCALIZER_SYSTEM),
            HumanMessage(content=localizer_prompt(issue, candidate_paths, context)),
        ]
    )
    allowed = set(candidate_paths)
    lines = content_text(response).splitlines()
    chosen = [ln.strip() for ln in lines if ln.strip() in allowed]
    # De-dupe preserving order; fall back to the top candidate if nothing usable was returned.
    ordered = list(dict.fromkeys(chosen))
    return ordered or candidate_paths[:1]
