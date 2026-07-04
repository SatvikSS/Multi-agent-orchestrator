"""A single factory that builds a chat model for a given agent role.

All four providers are reached through LangChain's `init_chat_model`. OpenRouter is
OpenAI-compatible, so it is wired as the `openai` provider with a custom base_url.
Per-role routing is driven entirely by config, so switching a role's model is a
one-line change in config.yaml.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from repo_surgeon.config import AppConfig, Provider

# A function that builds a chat model for a role. Injected into the graph so tests can
# supply fakes without touching the network. `build_llm` is the production implementation.
LLMProvider = Callable[[AppConfig, str], BaseChatModel]

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_llm(config: AppConfig, role: str) -> BaseChatModel:
    """Return a configured chat model for `role`, validating the provider credential first."""
    routing = config.role(role)
    provider: Provider = routing.provider
    config.secrets.require(provider)

    kwargs: dict[str, Any] = {
        "model": routing.model,
        "temperature": routing.temperature,
    }

    if provider == "openrouter":
        # OpenRouter is OpenAI-compatible: reach it as the openai provider + a custom base_url.
        kwargs.update(
            model_provider="openai",
            base_url=_OPENROUTER_BASE_URL,
            api_key=config.secrets.openrouter_api_key,
        )
    elif provider == "openai":
        kwargs.update(model_provider="openai", api_key=config.secrets.openai_api_key)
    elif provider == "anthropic":
        kwargs.update(model_provider="anthropic", api_key=config.secrets.anthropic_api_key)
    elif provider == "ollama":
        kwargs.update(model_provider="ollama", base_url=config.secrets.ollama_base_url)
    else:  # pragma: no cover - guarded by the Provider Literal
        raise ValueError(f"Unsupported provider: {provider}")

    return cast(BaseChatModel, init_chat_model(**kwargs))
