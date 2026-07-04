"""Abstract IssueSource and the shared hint-extraction helper."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from repo_surgeon.models import Issue

# File paths like `pkg/module.py` and bare Python files.
_PATH_RE = re.compile(r"\b[\w./-]+\.py\b")
# Symbols named in prose, e.g. `paginate()` or `class Foo`.
_CALL_RE = re.compile(r"\b([a-zA-Z_]\w+)\s*\(")
_DEF_RE = re.compile(r"\b(?:def|class)\s+([a-zA-Z_]\w+)")


def extract_hints(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Pull candidate file paths and symbol names out of free-text issue content."""
    paths = tuple(dict.fromkeys(_PATH_RE.findall(text)))
    symbols = tuple(
        dict.fromkeys(_CALL_RE.findall(text) + _DEF_RE.findall(text))
    )
    return paths, symbols


class IssueSource(ABC):
    """Something that yields a normalized Issue (from GitHub, a file, or a raw string)."""

    @abstractmethod
    def fetch(self) -> Issue:
        """Return the normalized issue."""
