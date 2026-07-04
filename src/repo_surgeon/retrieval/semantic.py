"""Semantic code search — the writer's 'find related code' tool.

Chunks the repo (AST-aware), embeds each chunk, and answers similarity queries. This is a
supplement to grep + knowledge-base routing, not the localization spine: it earns its keep
for "show me code similar to this" once the agent has a foothold.
"""

from __future__ import annotations

import logging

from repo_surgeon.config import Semantic
from repo_surgeon.llm.embeddings import EmbeddingModel
from repo_surgeon.retrieval.chunking import CodeChunk, chunk_python
from repo_surgeon.retrieval.vector_index import VectorIndex, make_vector_index
from repo_surgeon.workspace.base import Workspace

logger = logging.getLogger(__name__)


class SemanticIndex:
    """A built index over a repo's code chunks; answers `search(query, k)`."""

    def __init__(
        self, chunks: list[CodeChunk], index: VectorIndex, embeddings: EmbeddingModel
    ) -> None:
        self._chunks = chunks
        self._index = index
        self._embeddings = embeddings

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int) -> list[CodeChunk]:
        if not self._chunks or k <= 0:
            return []
        vector = self._embeddings.embed_query(query)
        return [self._chunks[i] for i in self._index.query(vector, k)]


def build_semantic_index(
    workspace: Workspace,
    embeddings: EmbeddingModel,
    cfg: Semantic,
) -> SemanticIndex | None:
    """Chunk + embed the repo's Python files. Returns None if there is nothing to index."""
    chunks: list[CodeChunk] = []
    for path in workspace.list_files(suffix=".py")[: cfg.max_files]:
        try:
            chunks.extend(chunk_python(path, workspace.read_file(path)))
        except (OSError, UnicodeDecodeError):
            continue
    if not chunks:
        return None

    vectors = embeddings.embed_documents([c.text for c in chunks])
    index = make_vector_index(cfg.vector_store)
    index.add(vectors)
    logger.info("Semantic index built: %d chunks (%s store).", len(chunks), cfg.vector_store)
    return SemanticIndex(chunks, index, embeddings)


def related_code_context(index: SemanticIndex | None, query: str, k: int) -> str:
    """Format the top-k similar chunks as a prompt block (empty string if no index)."""
    if index is None:
        return ""
    hits = index.search(query, k)
    if not hits:
        return ""
    blocks = [
        f"----- {h.file_path}:{h.start_line}-{h.end_line} ({h.symbol}) -----\n{h.text}"
        for h in hits
    ]
    header = "Related code elsewhere in the repo (for reference, do not necessarily edit):\n"
    return header + "\n\n".join(blocks)
