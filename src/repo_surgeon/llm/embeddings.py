"""Switchable embeddings backend for semantic code search.

Mirrors the chat-model factory: one `build_embeddings(config)` returns an object with the
LangChain `Embeddings` interface (`embed_documents`, `embed_query`), chosen by config.
`fake` yields deterministic vectors for offline tests and keyless demos.
"""

from __future__ import annotations

from typing import Protocol

from repo_surgeon.config import AppConfig


class EmbeddingModel(Protocol):
    """The subset of the LangChain Embeddings interface we rely on."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def build_embeddings(config: AppConfig) -> EmbeddingModel:
    """Construct the embeddings backend for `config.semantic`."""
    sem = config.semantic
    if sem.embed_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        config.secrets.require("openai")
        # api_key is accepted at runtime via pydantic alias; mypy doesn't see the alias.
        return OpenAIEmbeddings(model=sem.embed_model, api_key=config.secrets.openai_api_key)  # type: ignore[call-arg]
    if sem.embed_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=sem.embed_model, base_url=config.secrets.ollama_base_url)
    if sem.embed_provider == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=256)

    raise ValueError(f"Unsupported embeddings provider: {sem.embed_provider}")
