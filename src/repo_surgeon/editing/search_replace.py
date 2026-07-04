"""Parse and apply Aider-style SEARCH/REPLACE edit blocks.

The writer agent emits edits as text blocks (provider-agnostic, unlike tool-calling
structured output which local models handle poorly):

    path/to/file.py
    <<<<<<< SEARCH
    old code
    =======
    new code
    >>>>>>> REPLACE

Application is transactional across all patches: every SEARCH must locate its target
(exact match, then an indentation-tolerant fallback) or nothing is written and the
failures are reported back to the writer for another attempt.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from repo_surgeon.models import Patch
from repo_surgeon.workspace.base import Workspace

# Matches one SEARCH/REPLACE block; the path is recovered from the preceding line.
_BLOCK = re.compile(
    r"<<<<<<< SEARCH\n(?P<search>.*?)\n?=======\n(?P<replace>.*?)\n?>>>>>>> REPLACE",
    re.DOTALL,
)


class ApplyResult(BaseModel):
    """Outcome of applying a batch of patches to a workspace."""

    model_config = {"frozen": True}

    ok: bool
    applied: int = 0
    failures: tuple[str, ...] = ()

    def feedback(self) -> str:
        """A message the writer can use to correct a failed attempt."""
        if self.ok:
            return ""
        detail = "\n".join(f"  - {f}" for f in self.failures)
        return f"Some edits did not apply. Fix the SEARCH text to match the file exactly:\n{detail}"


def parse_search_replace(text: str) -> list[Patch]:
    """Extract Patch objects from LLM output containing SEARCH/REPLACE blocks."""
    patches: list[Patch] = []
    for match in _BLOCK.finditer(text):
        path = _path_before(text, match.start())
        if path is None:
            continue
        patches.append(
            Patch(file_path=path, search=match.group("search"), replace=match.group("replace"))
        )
    return patches


def apply_patches(workspace: Workspace, patches: list[Patch]) -> ApplyResult:
    """Apply all patches transactionally: write only if every SEARCH matches."""
    if not patches:
        return ApplyResult(ok=False, failures=("No edits were produced.",))

    # Compute new content per file in memory first; write nothing until all resolve.
    pending: dict[str, str] = {}
    failures: list[str] = []
    for patch in patches:
        current = pending.get(patch.file_path)
        if current is None:
            try:
                current = workspace.read_file(patch.file_path)
            except (FileNotFoundError, OSError):
                failures.append(f"{patch.file_path}: file not found")
                continue
        updated = _replace_once(current, patch.search, patch.replace)
        if updated is None:
            failures.append(f"{patch.file_path}: SEARCH text not found")
            continue
        pending[patch.file_path] = updated

    if failures:
        return ApplyResult(ok=False, failures=tuple(failures))

    for rel_path, content in pending.items():
        workspace.write_file(rel_path, content)
    return ApplyResult(ok=True, applied=len(patches))


def _path_before(text: str, block_start: int) -> str | None:
    """The file path is the last non-empty, non-fence line before the SEARCH marker."""
    for line in reversed(text[:block_start].splitlines()):
        stripped = line.strip().strip("`").strip()
        if stripped:
            return stripped
    return None


def _replace_once(content: str, search: str, replace: str) -> str | None:
    """Replace the first occurrence of `search`. Exact match first, then indentation-tolerant."""
    if search in content:
        return content.replace(search, replace, 1)
    return _replace_flexible(content, search, replace)


def _replace_flexible(content: str, search: str, replace: str) -> str | None:
    """Match a block of lines ignoring per-line leading/trailing whitespace differences."""
    content_lines = content.splitlines(keepends=True)
    search_lines = [ln.strip() for ln in search.strip().splitlines()]
    if not search_lines:
        return None

    window = len(search_lines)
    for i in range(len(content_lines) - window + 1):
        chunk = [ln.strip() for ln in content_lines[i : i + window]]
        if chunk == search_lines:
            before = "".join(content_lines[:i])
            after = "".join(content_lines[i + window :])
            # Preserve the trailing newline shape of the matched region.
            tail = "\n" if content_lines[i + window - 1].endswith("\n") else ""
            return before + replace.rstrip("\n") + tail + after
    return None
