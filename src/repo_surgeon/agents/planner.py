"""Planner agent: turn the issue + located code into a concrete fix plan."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from repo_surgeon.agents.prompts import PLANNER_SYSTEM, planner_prompt
from repo_surgeon.llm.response import content_text
from repo_surgeon.models import Issue


def make_plan(issue: Issue, context: str, llm: BaseChatModel) -> str:
    """Return a short natural-language fix plan."""
    response = llm.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=planner_prompt(issue, context)),
        ]
    )
    return content_text(response).strip()
