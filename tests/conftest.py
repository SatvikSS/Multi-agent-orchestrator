"""Shared pytest fixtures."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from repo_surgeon.config import AppConfig, Budget, RoleModel, Secrets
from repo_surgeon.models import TestResult
from repo_surgeon.sandbox import TestRunner
from repo_surgeon.workspace.base import Workspace


@pytest.fixture
def toy_repo(tmp_path: Path) -> Path:
    """A minimal initialized git repo with one Python file and a passing test."""
    repo = tmp_path / "toy"
    repo.mkdir()

    (repo / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (repo / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("add", "-A")
    git("commit", "-q", "-m", "initial")
    return repo


@pytest.fixture
def buggy_repo(tmp_path: Path) -> Path:
    """A git repo whose add() has a seeded bug (subtracts) and a test that therefore fails."""
    repo = tmp_path / "buggy"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repo / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("add", "-A")
    git("commit", "-q", "-m", "initial")
    return repo


def content_test_runner(needle: str, *, target: str = "calc.py") -> TestRunner:
    """A fake test runner that "passes" iff `target` currently contains `needle`.

    Stands in for the Docker sandbox in fast unit tests: it reflects the applied patch the
    same way a real suite would, without needing Docker.
    """

    def runner(workspace: Workspace) -> TestResult:
        passed = needle in workspace.read_file(target)
        return TestResult(
            passed=passed,
            total=1,
            failed=0 if passed else 1,
            failures=() if passed else ("test_calc.py::test_add",),
            raw_output="fake runner",
        )

    return runner


@pytest.fixture
def app_config() -> AppConfig:
    """A config with all roles routed to a local provider; no cloud keys needed for stubs."""
    role = RoleModel(provider="ollama", model="qwen2.5-coder:7b", temperature=0.0)
    roles = {name: role for name in ("localizer", "planner", "writer", "critic", "summarizer")}
    return AppConfig(roles=roles, budget=Budget(max_attempts=4), secrets=Secrets())


def make_fake_provider(
    responses_by_role: dict[str, list[str]],
) -> Callable[[AppConfig, str], BaseChatModel]:
    """A fake llm_provider: each role returns a persistent fake model cycling its responses.

    Instances are cached per role so a model's response index survives across retries (the
    writer can be given [bad, good] to exercise the self-correction loop).
    """
    cache: dict[str, BaseChatModel] = {}

    def provider(config: AppConfig, role: str) -> BaseChatModel:
        if role not in cache:
            cache[role] = FakeListChatModel(responses=responses_by_role.get(role, [""]))
        return cache[role]

    return provider
