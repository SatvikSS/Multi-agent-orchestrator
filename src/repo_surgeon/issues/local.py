"""Local issue source: a raw description string or a path to a text/markdown file."""

from __future__ import annotations

from pathlib import Path

from repo_surgeon.issues.base import IssueSource, extract_hints
from repo_surgeon.models import Issue


class LocalIssueSource(IssueSource):
    """Build an Issue from a CLI string or a local description file."""

    def __init__(self, text_or_path: str) -> None:
        self._raw = text_or_path

    def fetch(self) -> Issue:
        candidate = Path(self._raw).expanduser()
        if candidate.is_file():
            body = candidate.read_text(encoding="utf-8")
            title = _first_line(body)
        else:
            body = self._raw
            title = _first_line(self._raw)

        paths, symbols = extract_hints(body)
        return Issue(
            title=title,
            body=body,
            source="local",
            hint_paths=paths,
            hint_symbols=symbols,
        )


def _first_line(text: str, *, limit: int = 80) -> str:
    line = text.strip().splitlines()[0] if text.strip() else "Untitled issue"
    return line[:limit]
