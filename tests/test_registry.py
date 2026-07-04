"""Tests for project discovery and the registry."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from conftest import make_fake_provider
from repo_surgeon.config import AppConfig
from repo_surgeon.registry import discover, load_registry


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPO_SURGEON_HOME", str(tmp_path / "surgeon-home"))


def _make_repo(path: Path, *, origin: str | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@t.co")
    git("config", "user.name", "T")
    (path / "main.py").write_text("x = 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    if origin:
        git("remote", "add", "origin", origin)
    return path


@pytest.fixture
def documents(tmp_path: Path) -> Path:
    """A miniature ~/Documents: repo, plain folder, container, duplicates, empty dir."""
    root = tmp_path / "Documents"
    root.mkdir()

    _make_repo(root / "ontology", origin="https://github.com/org/ontology.git")

    plain = root / "data-scripts"
    plain.mkdir()
    (plain / "clean.py").write_text("pass\n", encoding="utf-8")
    (plain / "raw.csv").write_text("a,b\n", encoding="utf-8")

    container = root / "cpc-models"
    container.mkdir()
    url = "https://github.com/org/chillerPC.git"
    _make_repo(container / "clone-old", origin=url)
    time.sleep(1.1)  # commit timestamps have 1s resolution; make clone-new strictly fresher
    _make_repo(container / "clone-new", origin=url.removesuffix(".git"))

    (root / "empty-folder").mkdir()
    return root


def test_discover_classifies_shapes(documents: Path) -> None:
    registry = discover(documents)
    kinds = {p.name: p.kind for p in registry.projects}
    assert kinds["ontology"] == "repo"
    assert kinds["data-scripts"] == "plain"
    assert kinds["cpc-models"] == "container"
    assert kinds["cpc-models/clone-old"] == "nested_repo"
    assert kinds["cpc-models/clone-new"] == "nested_repo"
    assert "empty-folder" not in kinds


def test_discover_counts_python_files(documents: Path) -> None:
    registry = discover(documents)
    plain = registry.find("data-scripts")
    assert plain is not None
    assert plain.py_files == 1
    assert plain.total_files == 2  # clean.py + raw.csv


def test_duplicates_grouped_and_canonical_chosen(documents: Path) -> None:
    registry = discover(documents)
    old = registry.find("cpc-models/clone-old")
    new = registry.find("cpc-models/clone-new")
    assert old is not None and new is not None
    # Grouped despite the '.git' suffix difference in origin URLs.
    assert old.duplicate_group == new.duplicate_group is not None
    assert new.is_canonical is True  # fresher commit wins
    assert old.is_canonical is False
    # The unrelated repo is not part of any duplicate group.
    ontology = registry.find("ontology")
    assert ontology is not None and ontology.duplicate_group is None


def test_registry_persists_and_loads(documents: Path) -> None:
    discover(documents)
    loaded = load_registry()
    assert loaded is not None
    assert loaded.root == str(documents)
    assert len(loaded.projects) == 5


def test_load_registry_missing_returns_none() -> None:
    assert load_registry() is None


def test_discover_with_summaries(documents: Path, app_config: AppConfig) -> None:
    provider = make_fake_provider({"summarizer": ["A test project doing test things."]})
    registry = discover(documents, llm_provider=provider, app_config=app_config)
    ontology = registry.find("ontology")
    assert ontology is not None
    assert ontology.summary == "A test project doing test things."
    container = registry.find("cpc-models")
    assert container is not None
    assert container.summary == ""  # containers are not summarized


def test_discover_survives_summary_failure(documents: Path, app_config: AppConfig) -> None:
    def broken_provider(config: AppConfig, role: str):
        raise RuntimeError("no key")

    registry = discover(documents, llm_provider=broken_provider, app_config=app_config)
    assert len(registry.projects) == 5  # discovery completes, summaries just empty
    assert all(p.summary == "" for p in registry.projects)
