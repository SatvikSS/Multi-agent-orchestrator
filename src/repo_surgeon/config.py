"""Configuration: secrets from the environment + role→model routing from config.yaml.

Secrets are validated lazily per active role (we only require the key for a provider
that is actually used), so a fully-local Ollama run needs no cloud keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openrouter", "ollama", "anthropic", "openai"]

_DEFAULT_CONFIG_PATH = Path("config.yaml")


class Secrets(BaseSettings):
    """Environment-sourced secrets. All optional; presence checked per active provider."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    github_token: str | None = None

    def require(self, provider: Provider) -> None:
        """Fail fast with a clear message if the key for `provider` is missing."""
        missing = {
            "openrouter": self.openrouter_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "ollama": self.ollama_base_url,  # always present (defaulted)
        }
        if not missing[provider]:
            raise ValueError(
                f"Provider '{provider}' is selected but its credential is not set. "
                f"Add it to your .env (see .env.example)."
            )


class RoleModel(BaseModel):
    """Model routing for a single agent role."""

    model_config = {"frozen": True}

    provider: Provider
    model: str
    temperature: float = 0.0


class Budget(BaseModel):
    """Caps on the self-correction loop."""

    model_config = {"frozen": True}

    max_attempts: int = 4
    max_cost_usd: float = 2.0


class Sandbox(BaseModel):
    """Docker sandbox settings for running the target repo's tests."""

    model_config = {"frozen": True}

    python_version: str = "3.11"
    test_command: str = "python -m pytest -q -rfE -p no:cacheprovider"
    timeout_seconds: int = 300
    mem_limit: str = "512m"
    cpus: float = 2.0
    pids_limit: int = 256
    # Network is disabled while running untrusted test code. Enable only if the target
    # repo's tests genuinely need it (weakens isolation).
    network_enabled: bool = False
    # Best-effort pip install of the repo's requirements.txt when present (build-time only).
    install_requirements: bool = True


class Semantic(BaseModel):
    """Optional semantic code search — the writer's 'find related code' tool.

    Off by default: it needs an embeddings backend and adds indexing cost. When enabled
    it augments the writer's context with the most similar code chunks to the issue/plan.
    """

    model_config = {"frozen": True}

    enabled: bool = False
    embed_provider: Literal["openai", "ollama", "fake"] = "ollama"
    embed_model: str = "nomic-embed-text"
    vector_store: Literal["memory", "chroma"] = "memory"
    top_k: int = 3
    max_files: int = 60


class AppConfig(BaseModel):
    """Fully-resolved application config: role routing + budget + sandbox + secrets."""

    model_config = {"frozen": True}

    roles: dict[str, RoleModel]
    budget: Budget = Field(default_factory=Budget)
    sandbox: Sandbox = Field(default_factory=Sandbox)
    semantic: Semantic = Field(default_factory=Semantic)
    secrets: Secrets = Field(default_factory=Secrets)

    def role(self, name: str) -> RoleModel:
        if name not in self.roles:
            raise KeyError(f"No model routing configured for role '{name}'. Check config.yaml.")
        return self.roles[name]


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load config.yaml + environment secrets into a validated AppConfig."""
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    raw = yaml.safe_load(path.read_text()) or {}
    default_temp = float(raw.get("defaults", {}).get("temperature", 0.0))

    roles: dict[str, RoleModel] = {}
    for name, spec in (raw.get("roles") or {}).items():
        roles[name] = RoleModel(
            provider=spec["provider"],
            model=spec["model"],
            temperature=float(spec.get("temperature", default_temp)),
        )

    budget = Budget(**(raw.get("budget") or {}))
    sandbox = Sandbox(**(raw.get("sandbox") or {}))
    semantic = Semantic(**(raw.get("semantic") or {}))
    return AppConfig(
        roles=roles, budget=budget, sandbox=sandbox, semantic=semantic, secrets=Secrets()
    )
