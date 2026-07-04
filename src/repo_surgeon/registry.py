"""Documents-level project registry: what projects exist and what each one is.

`discover(root)` scans a folder of projects (e.g. ~/Documents) once and records, per
project: its shape (repo / nested repo / plain folder / container), origin remote,
duplicate-clone grouping, rough language stats, and an optional 1-2 sentence LLM
summary. The registry lives at ~/.repo-surgeon/registry.json and powers
`surgeon projects` now and issue→project auto-routing later (item 5).
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from repo_surgeon.config import AppConfig
from repo_surgeon.home import home_dir
from repo_surgeon.llm.factory import LLMProvider
from repo_surgeon.llm.response import content_text
from repo_surgeon.workspace.ignore import JUNK_DIRS
from repo_surgeon.workspace.resolver import normalize_origin

logger = logging.getLogger(__name__)

_REGISTRY_FILE = "registry.json"
_FILE_COUNT_CAP = 2000
_SCAN_DEPTH = 3
_README_NAMES = ("README.md", "README.rst", "README.txt", "readme.md")
_SUMMARY_SYSTEM = (
    "You summarize a software project in 1-2 sentences: what it is for and its main "
    "technology. Be terse and concrete. No preamble."
)

Kind = Literal["repo", "nested_repo", "plain", "container"]


class ProjectEntry(BaseModel):
    """One discovered project."""

    model_config = {"frozen": True}

    name: str
    path: str
    kind: Kind
    container: str | None = None  # parent container path, for nested repos
    origin: str | None = None
    duplicate_group: str | None = None  # normalized origin shared by >1 entries
    is_canonical: bool = True  # freshest clone in its duplicate group
    py_files: int = 0
    total_files: int = 0
    summary: str = ""


class Registry(BaseModel):
    """All projects discovered under one root."""

    model_config = {"frozen": True}

    root: str
    generated_at: str
    projects: list[ProjectEntry]

    def find(self, name: str) -> ProjectEntry | None:
        for p in self.projects:
            if p.name == name:
                return p
        return None


def registry_path() -> Path:
    return home_dir() / _REGISTRY_FILE


def load_registry() -> Registry | None:
    path = registry_path()
    if not path.exists():
        return None
    return Registry(**json.loads(path.read_text(encoding="utf-8")))


def discover(
    root: str | Path,
    *,
    llm_provider: LLMProvider | None = None,
    app_config: AppConfig | None = None,
) -> Registry:
    """Scan `root` for projects, optionally summarize each, and persist the registry."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")

    entries: list[ProjectEntry] = []
    for child in sorted(p for p in base.iterdir() if p.is_dir()):
        if child.name.startswith(".") or child.name in JUNK_DIRS:
            continue
        entries.extend(_classify(child))

    entries = _mark_duplicates(entries)

    if llm_provider is not None and app_config is not None:
        entries = _summarize(entries, llm_provider, app_config)

    registry = Registry(
        root=str(base),
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        projects=entries,
    )
    registry_path().write_text(registry.model_dump_json(indent=2), encoding="utf-8")
    return registry


# --- classification -----------------------------------------------------------


def _classify(folder: Path) -> list[ProjectEntry]:
    """Turn one top-level folder into registry entries."""
    py, total = _count_files(folder)
    if total == 0:
        return []  # empty folder — nothing to register

    if (folder / ".git").exists():
        return [
            ProjectEntry(
                name=folder.name,
                path=str(folder),
                kind="repo",
                origin=_origin(folder),
                py_files=py,
                total_files=total,
            )
        ]

    nested = _find_nested_repos(folder)
    if not nested:
        return [
            ProjectEntry(
                name=folder.name, path=str(folder), kind="plain", py_files=py, total_files=total
            )
        ]

    entries = [
        ProjectEntry(
            name=folder.name,
            path=str(folder),
            kind="container",
            py_files=py,
            total_files=total,
        )
    ]
    for repo in nested:
        npy, ntotal = _count_files(repo)
        entries.append(
            ProjectEntry(
                name=f"{folder.name}/{repo.relative_to(folder)}",
                path=str(repo),
                kind="nested_repo",
                container=str(folder),
                origin=_origin(repo),
                py_files=npy,
                total_files=ntotal,
            )
        )
    return entries


def _find_nested_repos(folder: Path, *, depth: int = 1) -> list[Path]:
    if depth > _SCAN_DEPTH:
        return []
    found: list[Path] = []
    try:
        children = sorted(p for p in folder.iterdir() if p.is_dir())
    except (PermissionError, OSError):
        return []
    for child in children:
        if child.name in JUNK_DIRS or child.name.startswith("."):
            continue
        if (child / ".git").exists():
            found.append(child)  # don't descend into a repo
        else:
            found.extend(_find_nested_repos(child, depth=depth + 1))
    return found


def _count_files(folder: Path) -> tuple[int, int]:
    """(py_files, total_files) under `folder`, junk skipped, capped for speed."""
    py = total = 0
    for path in islice(_iter_files(folder), _FILE_COUNT_CAP):
        total += 1
        if path.suffix == ".py":
            py += 1
    return py, total


def _iter_files(folder: Path) -> Iterator[Path]:
    try:
        entries = sorted(folder.iterdir())
    except (PermissionError, OSError):
        return
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            if entry.name in JUNK_DIRS:
                continue
            yield from _iter_files(entry)
        elif entry.is_file():
            yield entry


def _origin(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=repo, capture_output=True, text=True
    )
    url = result.stdout.strip()
    return url if result.returncode == 0 and url else None


# --- duplicates ----------------------------------------------------------------


def _mark_duplicates(entries: list[ProjectEntry]) -> list[ProjectEntry]:
    """Group git entries by normalized origin; freshest clone in each group is canonical."""
    groups: dict[str, list[int]] = {}
    for i, entry in enumerate(entries):
        if entry.origin:
            groups.setdefault(normalize_origin(entry.origin), []).append(i)

    updated = list(entries)
    for origin_key, indexes in groups.items():
        if len(indexes) < 2:
            continue
        freshest = max(indexes, key=lambda i: _head_commit_time(Path(entries[i].path)))
        for i in indexes:
            updated[i] = entries[i].model_copy(
                update={"duplicate_group": origin_key, "is_canonical": i == freshest}
            )
    return updated


def _head_commit_time(repo: Path) -> int:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct"], cwd=repo, capture_output=True, text=True
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


# --- summaries -------------------------------------------------------------------


def _summarize(
    entries: list[ProjectEntry], llm_provider: LLMProvider, app_config: AppConfig
) -> list[ProjectEntry]:
    """Attach a short LLM summary to each repo/plain entry (containers just list children)."""
    try:
        llm = llm_provider(app_config, "summarizer")
    except Exception as exc:  # noqa: BLE001 - missing key etc.; registry is still useful bare
        logger.warning("Summaries skipped (no usable summarizer): %s", exc)
        return entries
    updated: list[ProjectEntry] = []
    for entry in entries:
        if entry.kind == "container":
            updated.append(entry)
            continue
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=_SUMMARY_SYSTEM),
                    HumanMessage(content=_summary_input(Path(entry.path), entry)),
                ]
            )
            updated.append(entry.model_copy(update={"summary": content_text(response).strip()}))
        except Exception as exc:  # noqa: BLE001 - a failed summary must not sink discovery
            logger.warning("Could not summarize %s: %s", entry.name, exc)
            updated.append(entry)
    return updated


def _summary_input(folder: Path, entry: ProjectEntry) -> str:
    readme = ""
    for name in _README_NAMES:
        candidate = folder / name
        if candidate.is_file():
            readme = candidate.read_text(encoding="utf-8", errors="replace")[:2000]
            break
    listing = ", ".join(
        p.name for p in islice(_iter_files(folder), 50)
    )
    return (
        f"Project '{entry.name}' ({entry.py_files} python files).\n"
        f"README excerpt:\n{readme or '(no README)'}\n\nFiles: {listing}"
    )
