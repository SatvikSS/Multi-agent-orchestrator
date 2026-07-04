"""Tests for the search-replace parser and transactional applier."""

from __future__ import annotations

from pathlib import Path

from repo_surgeon.editing import apply_patches, parse_search_replace
from repo_surgeon.models import Patch
from repo_surgeon.workspace import LocalWorkspace

_BLOCK = """\
calc.py
<<<<<<< SEARCH
    return a + b
=======
    return a + b + 0
>>>>>>> REPLACE
"""


def test_parse_single_block() -> None:
    patches = parse_search_replace(_BLOCK)
    assert len(patches) == 1
    assert patches[0].file_path == "calc.py"
    assert patches[0].search == "    return a + b"
    assert patches[0].replace == "    return a + b + 0"


def test_parse_multiple_blocks_with_fences() -> None:
    text = (
        "```python\nfoo.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n```\n"
        "bar.py\n<<<<<<< SEARCH\ny = 1\n=======\ny = 2\n>>>>>>> REPLACE\n"
    )
    patches = parse_search_replace(text)
    assert [p.file_path for p in patches] == ["foo.py", "bar.py"]


def test_parse_no_blocks_returns_empty() -> None:
    assert parse_search_replace("no edits here") == []


def test_apply_patches_success(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    result = apply_patches(ws, parse_search_replace(_BLOCK))
    assert result.ok is True
    assert result.applied == 1
    assert "a + b + 0" in ws.read_file("calc.py")


def test_apply_patches_is_transactional(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    good = Patch(file_path="calc.py", search="    return a + b", replace="    return 42")
    bad = Patch(file_path="calc.py", search="does not exist", replace="x")
    result = apply_patches(ws, [good, bad])
    assert result.ok is False
    # Neither edit should have been written because one failed.
    assert "return 42" not in ws.read_file("calc.py")
    assert "SEARCH text not found" in result.feedback()


def test_apply_patches_flexible_whitespace(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    # SEARCH indented differently than the file (no leading spaces) still matches.
    patch = Patch(file_path="calc.py", search="return a + b", replace="return a - b")
    result = apply_patches(ws, [patch])
    assert result.ok is True
    assert "a - b" in ws.read_file("calc.py")


def test_apply_empty_patch_list_fails() -> None:
    class _NoWs:
        pass

    result = apply_patches(_NoWs(), [])  # type: ignore[arg-type]
    assert result.ok is False
