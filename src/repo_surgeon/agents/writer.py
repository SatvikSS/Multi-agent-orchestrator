"""Writer agent: emit search-replace edits for the planned fix."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from repo_surgeon.agents.prompts import WRITER_SYSTEM, writer_prompt
from repo_surgeon.editing import parse_search_replace
from repo_surgeon.llm.response import content_text
from repo_surgeon.models import Issue, Patch


def write_patches(
    issue: Issue,
    plan: str,
    context: str,
    llm: BaseChatModel,
    *,
    feedback: str = "",
) -> list[Patch]:
    """Ask the model for SEARCH/REPLACE blocks and parse them into patches.

    `feedback` carries the previous attempt's failure so the model can correct its SEARCH text.
    """
    response = llm.invoke(
        [
            SystemMessage(content=WRITER_SYSTEM),
            HumanMessage(content=writer_prompt(issue, plan, context, feedback)),
        ]
    )
    return parse_search_replace(content_text(response))
