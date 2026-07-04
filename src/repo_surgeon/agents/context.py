"""Build code-context bundles to feed agents."""

from __future__ import annotations

from repo_surgeon.retrieval.reader import read_numbered
from repo_surgeon.workspace.base import Workspace

_MAX_CHARS_PER_FILE = 6000


def file_context(
    workspace: Workspace,
    paths: list[str],
    *,
    numbered: bool = False,
    max_chars: int = _MAX_CHARS_PER_FILE,
) -> str:
    """Concatenate the given files into a labeled bundle for a prompt.

    `numbered=True` adds line-number gutters (useful for reasoning/localization);
    the writer uses raw text so its SEARCH blocks match the file byte-for-byte.
    """
    sections: list[str] = []
    for path in paths:
        try:
            body = read_numbered(workspace, path) if numbered else workspace.read_file(path)
        except (FileNotFoundError, OSError):
            continue
        if len(body) > max_chars:
            body = body[:max_chars] + "\n... (truncated)"
        sections.append(f"===== {path} =====\n{body}")
    return "\n\n".join(sections)
