"""Retrieval layer: knowledge-base routing + grep/read navigation tools.

Right-sized for small/medium repos: knowledge-base summaries and hint-driven grep route
to candidate files, then the reader supplies precise context. Semantic search (Phase 5)
plugs in here as an optional similarity tool, not the localization spine.
"""

from repo_surgeon.retrieval.grep import GrepHit, grep
from repo_surgeon.retrieval.knowledge_base import (
    KnowledgeBase,
    build_knowledge_base,
    route_candidates,
)
from repo_surgeon.retrieval.reader import read_numbered
from repo_surgeon.retrieval.semantic import (
    SemanticIndex,
    build_semantic_index,
    related_code_context,
)

__all__ = [
    "GrepHit",
    "KnowledgeBase",
    "SemanticIndex",
    "build_knowledge_base",
    "build_semantic_index",
    "grep",
    "read_numbered",
    "related_code_context",
    "route_candidates",
]
