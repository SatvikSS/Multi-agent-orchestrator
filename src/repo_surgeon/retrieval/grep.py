"""Keyword/symbol search — the exact-match half of localization.

Uses ripgrep when available (fast on large repos) and falls back to a pure-Python scan
otherwise, so the tool has no hard external dependency and behaves the same in any env.
"""

from __future__ import annotations

import shutil
import subprocess

from pydantic import BaseModel

from repo_surgeon.workspace.base import Workspace


class GrepHit(BaseModel):
    """A single search match."""

    model_config = {"frozen": True}

    file_path: str
    line: int
    text: str


def grep(workspace: Workspace, pattern: str, *, max_results: int = 50) -> list[GrepHit]:
    """Search the repo for the fixed string `pattern` and return matches."""
    rg = shutil.which("rg")
    if rg is not None:
        return _grep_ripgrep(rg, workspace, pattern, max_results=max_results)
    return _grep_python(workspace, pattern, max_results=max_results)


def _grep_ripgrep(
    rg: str, workspace: Workspace, pattern: str, *, max_results: int
) -> list[GrepHit]:
    result = subprocess.run(
        [rg, "--no-heading", "--line-number", "--with-filename", "--fixed-strings", pattern, "."],
        cwd=workspace.root_path,
        capture_output=True,
        text=True,
    )
    # rg exits 1 when there are simply no matches; 2 signals a real error.
    if result.returncode == 1:
        return []
    if result.returncode >= 2:
        raise RuntimeError(f"ripgrep failed: {result.stderr.strip()}")

    hits: list[GrepHit] = []
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)  # ./path:42:text
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        hits.append(
            GrepHit(
                file_path=parts[0].removeprefix("./"),
                line=int(parts[1]),
                text=parts[2].strip(),
            )
        )
        if len(hits) >= max_results:
            break
    return hits


def _grep_python(workspace: Workspace, pattern: str, *, max_results: int) -> list[GrepHit]:
    hits: list[GrepHit] = []
    for path in workspace.list_files():
        try:
            content = workspace.read_file(path)
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            continue
        for lineno, text in enumerate(content.splitlines(), start=1):
            if pattern in text:
                hits.append(GrepHit(file_path=path, line=lineno, text=text.strip()))
                if len(hits) >= max_results:
                    return hits
    return hits
