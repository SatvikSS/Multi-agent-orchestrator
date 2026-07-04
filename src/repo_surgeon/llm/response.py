"""Helpers for normalizing LLM responses.

Chat models return message content that may be a plain string or a list of content
blocks (depending on provider); this collapses either shape to plain text.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage


def content_text(message: BaseMessage) -> str:
    """Return the message content as a single plain-text string."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and "text" in block:
            parts.append(str(block["text"]))
    return "".join(parts)
