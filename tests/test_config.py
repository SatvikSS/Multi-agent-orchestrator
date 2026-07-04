"""Tests for config loading and secret validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from repo_surgeon.config import Secrets, load_config


def test_load_config_reads_roles_and_budget(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
defaults:
  temperature: 0.1
roles:
  writer:
    provider: openai
    model: gpt-4o
budget:
  max_attempts: 3
  max_cost_usd: 1.5
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)

    assert cfg.role("writer").provider == "openai"
    assert cfg.role("writer").model == "gpt-4o"
    assert cfg.role("writer").temperature == 0.1  # inherits defaults
    assert cfg.budget.max_attempts == 3
    assert cfg.budget.max_cost_usd == 1.5


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_unknown_role_raises(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "roles:\n  writer:\n    provider: openai\n    model: gpt-4o\n", encoding="utf-8"
    )
    cfg = load_config(cfg_file)
    with pytest.raises(KeyError):
        cfg.role("does-not-exist")


def test_secrets_require_missing_key() -> None:
    secrets = Secrets(openai_api_key=None)
    with pytest.raises(ValueError, match="openai"):
        secrets.require("openai")


def test_secrets_require_present_key() -> None:
    secrets = Secrets(openai_api_key="sk-test")
    secrets.require("openai")  # should not raise


def test_secrets_ollama_always_available() -> None:
    secrets = Secrets()
    secrets.require("ollama")  # base_url is defaulted, so this passes
