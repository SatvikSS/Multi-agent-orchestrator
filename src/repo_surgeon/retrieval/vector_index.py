"""Pluggable vector index behind semantic search.

`InMemoryVectorIndex` (numpy cosine similarity) is the zero-extra-dependency default and
is what the tests exercise — plenty for the hundreds of chunks in a small repo.
`ChromaVectorIndex` is an optional persistent backend enabled by the `chroma` extra.
Both take pre-computed embeddings, so the choice of vector store is independent of the
choice of embeddings model.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Protocol, cast

import numpy as np


class VectorIndex(Protocol):
    """Stores vectors with integer ids and returns nearest ids for a query vector."""

    def add(self, vectors: list[list[float]]) -> None: ...

    def query(self, vector: list[float], k: int) -> list[int]: ...


class InMemoryVectorIndex:
    """Cosine-similarity search over an in-memory numpy matrix of L2-normalized rows."""

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None

    def add(self, vectors: list[list[float]]) -> None:
        if not vectors:
            return
        self._matrix = _normalize(np.asarray(vectors, dtype=np.float32))

    def query(self, vector: list[float], k: int) -> list[int]:
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        q = _normalize(np.asarray([vector], dtype=np.float32))[0]
        scores = self._matrix @ q
        top = np.argsort(scores)[::-1][:k]
        return [int(i) for i in top]


class ChromaVectorIndex:
    """Persistent Chroma-backed index (requires the `chroma` extra)."""

    def __init__(self, collection_name: str = "repo-surgeon") -> None:
        import chromadb  # imported lazily: only when the chroma backend is selected

        client = chromadb.EphemeralClient()
        # Reset a stale collection so each build starts clean.
        with suppress(Exception):
            client.delete_collection(collection_name)
        self._collection = client.create_collection(collection_name)
        self._count = 0

    def add(self, vectors: list[list[float]]) -> None:
        if not vectors:
            return
        ids = [str(i) for i in range(self._count, self._count + len(vectors))]
        self._collection.add(ids=ids, embeddings=vectors)
        self._count += len(vectors)

    def query(self, vector: list[float], k: int) -> list[int]:
        if self._count == 0:
            return []
        result = self._collection.query(query_embeddings=[vector], n_results=min(k, self._count))
        ids = result.get("ids", [[]])[0]
        return [int(i) for i in ids]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return cast(np.ndarray, (matrix / norms).astype(np.float32, copy=False))


def make_vector_index(kind: str) -> VectorIndex:
    if kind == "chroma":
        return ChromaVectorIndex()
    return InMemoryVectorIndex()
