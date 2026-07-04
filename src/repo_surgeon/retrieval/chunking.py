"""AST-aware Python chunking for semantic search.

Chunks at function/class granularity using tree-sitter so each chunk is a self-contained
unit (not a mid-function slice), carrying its symbol name and line range — the make-or-break
for code retrieval quality. Falls back to a line-window splitter if tree-sitter is
unavailable, so indexing never hard-fails.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from tree_sitter import Node

logger = logging.getLogger(__name__)

_MAX_CLASS_LINES = 60  # larger classes are split into their methods instead of one chunk
_FALLBACK_WINDOW = 40


class CodeChunk(BaseModel):
    """A retrievable unit of code with its location."""

    model_config = {"frozen": True}

    file_path: str
    symbol: str
    kind: str  # "function" | "class" | "method" | "block"
    start_line: int  # 1-based, inclusive
    end_line: int
    text: str


def chunk_python(file_path: str, source: str) -> list[CodeChunk]:
    """Split one Python file into function/class-level chunks."""
    try:
        return _chunk_with_tree_sitter(file_path, source)
    except Exception as exc:  # noqa: BLE001 - any parser/import issue → safe fallback
        logger.debug("tree-sitter chunking failed for %s (%s); using fallback", file_path, exc)
        return _chunk_fallback(file_path, source)


def _chunk_with_tree_sitter(file_path: str, source: str) -> list[CodeChunk]:
    import tree_sitter_python
    from tree_sitter import Language, Parser

    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(source.encode("utf-8"))
    lines = source.splitlines()

    chunks: list[CodeChunk] = []
    for node in tree.root_node.children:
        if node.type == "function_definition":
            chunks.append(_node_chunk(file_path, node, lines, "function"))
        elif node.type == "class_definition":
            chunks.extend(_class_chunks(file_path, node, lines))
    return chunks


def _class_chunks(file_path: str, class_node: Node, lines: list[str]) -> list[CodeChunk]:
    span = class_node.end_point[0] - class_node.start_point[0] + 1
    if span <= _MAX_CLASS_LINES:
        return [_node_chunk(file_path, class_node, lines, "class")]

    # Large class: index each method separately for sharper retrieval.
    class_name = _identifier(class_node) or "class"
    body = next((c for c in class_node.children if c.type == "block"), None)
    methods = [c for c in (body.children if body else []) if c.type == "function_definition"]
    if not methods:
        return [_node_chunk(file_path, class_node, lines, "class")]
    return [
        _node_chunk(file_path, m, lines, "method", prefix=f"{class_name}.") for m in methods
    ]


def _node_chunk(
    file_path: str, node: Node, lines: list[str], kind: str, *, prefix: str = ""
) -> CodeChunk:
    start = node.start_point[0]  # 0-based
    end = node.end_point[0]
    text = "\n".join(lines[start : end + 1])
    name = _identifier(node) or kind
    return CodeChunk(
        file_path=file_path,
        symbol=f"{prefix}{name}",
        kind=kind,
        start_line=start + 1,
        end_line=end + 1,
        text=text,
    )


def _identifier(node: Node) -> str | None:
    for child in node.children:
        if child.type == "identifier" and child.text is not None:
            return child.text.decode("utf-8")
    return None


def _chunk_fallback(file_path: str, source: str) -> list[CodeChunk]:
    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    for start in range(0, len(lines), _FALLBACK_WINDOW):
        window = lines[start : start + _FALLBACK_WINDOW]
        if not any(line.strip() for line in window):
            continue
        chunks.append(
            CodeChunk(
                file_path=file_path,
                symbol=f"lines {start + 1}-{start + len(window)}",
                kind="block",
                start_line=start + 1,
                end_line=start + len(window),
                text="\n".join(window),
            )
        )
    return chunks
