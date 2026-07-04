"""Tests for AST chunking, the vector index, and semantic search (offline, fake embeddings)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding

from repo_surgeon.config import Semantic
from repo_surgeon.retrieval import build_semantic_index, related_code_context
from repo_surgeon.retrieval.chunking import chunk_python
from repo_surgeon.retrieval.vector_index import InMemoryVectorIndex
from repo_surgeon.workspace import LocalWorkspace

_SAMPLE = '''\
import os


def paginate(items, size):
    """Return a page of items."""
    return items[:size]


def deduplicate(items):
    return list(dict.fromkeys(items))


class Cache:
    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value
'''


# --- chunking ---------------------------------------------------------------


def test_chunk_python_splits_by_symbol() -> None:
    chunks = chunk_python("util.py", _SAMPLE)
    symbols = {c.symbol for c in chunks}
    assert "paginate" in symbols
    assert "deduplicate" in symbols
    # Small class stays as one chunk.
    assert "Cache" in symbols


def test_chunk_carries_line_ranges_and_text() -> None:
    chunks = chunk_python("util.py", _SAMPLE)
    paginate = next(c for c in chunks if c.symbol == "paginate")
    assert paginate.kind == "function"
    assert paginate.start_line < paginate.end_line
    assert "return items[:size]" in paginate.text


def test_large_class_split_into_methods() -> None:
    methods = "\n".join(f"    def m{i}(self):\n        return {i}\n" for i in range(40))
    source = f"class Big:\n{methods}"
    chunks = chunk_python("big.py", source)
    method_syms = {c.symbol for c in chunks if c.kind == "method"}
    assert "Big.m0" in method_syms
    assert "Big.m39" in method_syms


def test_chunk_fallback_on_unparsable(monkeypatch) -> None:
    # Force the tree-sitter path to fail; the fallback should still yield chunks.
    import repo_surgeon.retrieval.chunking as chunking

    monkeypatch.setattr(
        chunking, "_chunk_with_tree_sitter", lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    chunks = chunk_python("util.py", _SAMPLE)
    assert chunks
    assert all(c.kind == "block" for c in chunks)


# --- vector index -----------------------------------------------------------


def test_in_memory_index_returns_nearest() -> None:
    index = InMemoryVectorIndex()
    index.add([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    hits = index.query([1.0, 0.0], k=2)
    assert hits[0] == 0  # identical vector ranks first
    assert 2 in hits  # the near-parallel vector is the second match


def test_empty_index_returns_nothing() -> None:
    assert InMemoryVectorIndex().query([1.0, 0.0], k=3) == []


# --- semantic index end-to-end ----------------------------------------------


def _cfg() -> Semantic:
    return Semantic(enabled=True, embed_provider="fake", top_k=2, max_files=60)


def test_build_and_search(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    (toy_repo / "util.py").write_text(_SAMPLE, encoding="utf-8")
    # Track the new file so list_files(suffix='.py') sees it.
    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=toy_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "add util"], cwd=toy_repo, check=True, capture_output=True
    )

    index = build_semantic_index(ws, DeterministicFakeEmbedding(size=256), _cfg())
    assert index is not None
    assert index.chunk_count >= 3
    hits = index.search("pagination helper", k=2)
    assert len(hits) == 2
    assert all(h.file_path in {"util.py", "calc.py"} for h in hits)


def test_build_returns_none_for_no_python(tmp_path: Path) -> None:
    repo = tmp_path / "empty"
    repo.mkdir()
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.co"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / "README.md").write_text("no code", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "i"], cwd=repo, check=True, capture_output=True)

    ws = LocalWorkspace(repo)
    assert build_semantic_index(ws, DeterministicFakeEmbedding(size=256), _cfg()) is None


def test_related_code_context_formats_block(toy_repo: Path) -> None:
    ws = LocalWorkspace(toy_repo)
    index = build_semantic_index(ws, DeterministicFakeEmbedding(size=256), _cfg())
    block = related_code_context(index, "add two numbers", k=1)
    assert "Related code elsewhere" in block
    assert "calc.py" in block


def test_related_code_context_none_index_is_empty() -> None:
    assert related_code_context(None, "anything", k=3) == ""


def test_build_embeddings_fake() -> None:
    from repo_surgeon.config import AppConfig, Budget, RoleModel, Secrets, Semantic
    from repo_surgeon.llm.embeddings import build_embeddings

    cfg = AppConfig(
        roles={"writer": RoleModel(provider="ollama", model="x")},
        budget=Budget(),
        semantic=Semantic(embed_provider="fake"),
        secrets=Secrets(),
    )
    embeddings = build_embeddings(cfg)
    vec = embeddings.embed_query("hello")
    assert len(vec) == 256


def test_graph_uses_semantic_context(buggy_repo: Path, app_config) -> None:
    """The writer's context is enriched when a semantic index is supplied."""
    from conftest import content_test_runner, make_fake_provider
    from repo_surgeon.graph import build_graph
    from repo_surgeon.issues import LocalIssueSource
    from repo_surgeon.models import RunStatus
    from repo_surgeon.state import RunState

    ws = LocalWorkspace(buggy_repo)
    ws.start_work("Fix add() in calc.py")
    index = build_semantic_index(ws, DeterministicFakeEmbedding(size=256), _cfg())
    assert index is not None

    right = (
        "calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"
    )
    provider = make_fake_provider(
        {
            "summarizer": ["calc"],
            "localizer": ["calc.py"],
            "planner": ["- fix add"],
            "writer": [right],
        }
    )
    graph = build_graph(
        ws,
        app_config,
        llm_provider=provider,
        test_runner=content_test_runner("a + b"),
        semantic_index=index,
    )
    entry: RunState = {
        "issue": LocalIssueSource("Fix add() in calc.py").fetch(),
        "repo_ref": str(buggy_repo),
        "attempts": 0,
        "tokens_spent": 0,
        "cost_usd": 0.0,
        "status": RunStatus.IN_PROGRESS,
        "patches": [],
        "test_results": [],
        "apply_ok": False,
        "writer_feedback": "",
        "notes": [],
    }
    final = graph.invoke(entry, config={"recursion_limit": 50})
    assert final["status"] == RunStatus.RESOLVED
    assert any("+semantic context" in note for note in final["notes"])
