"""Resolve a user-supplied path to the git repository the agents should work in.

Real project folders come in three shapes (all present in the user's ~/Documents):
  - the path IS a git repo → use it;
  - the path is INSIDE a git repo (a subfolder) → walk up to the enclosing repo root;
  - the path is a CONTAINER holding one or more nested repos (e.g. `CPO_v3/` holding
    `CPO_v3/chillerPerformanceCalculator/`) → scan down: one nested repo is used
    directly, several raise AmbiguousWorkspaceError so the caller can let the user pick.

Candidates carry their `origin` remote so duplicate clones of the same repository
(common in this Documents tree) are visible at pick time.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel

_SCAN_DEPTH = 3
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".repo_surgeon_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
}


class RepoCandidate(BaseModel):
    """A git repo discovered under a container folder."""

    model_config = {"frozen": True}

    path: str  # absolute
    rel: str  # relative to the folder the user gave
    origin: str | None = None  # remote URL, for spotting duplicate clones

    def describe(self) -> str:
        origin = f" (origin: {self.origin})" if self.origin else " (no remote)"
        return f"{self.rel}{origin}"


class Resolution(BaseModel):
    """Outcome of resolving a path: the repo root to use and how it was found."""

    model_config = {"frozen": True}

    root: str  # absolute path of the git repo root
    kind: str  # "repo" | "inside_repo" | "nested_repo"
    note: str = ""


class AmbiguousWorkspaceError(RuntimeError):
    """The folder contains several nested repos; the caller must pick one."""

    def __init__(self, folder: Path, candidates: list[RepoCandidate]) -> None:
        self.folder = folder
        self.candidates = candidates
        listing = "\n".join(f"  {i + 1}. {c.describe()}" for i, c in enumerate(candidates))
        duplicates = _duplicate_origins(candidates)
        dup_note = (
            f"\nNote: {len(duplicates)} origin(s) appear more than once — these are "
            f"duplicate clones; prefer the one you actively work in."
            if duplicates
            else ""
        )
        super().__init__(
            f"'{folder}' contains {len(candidates)} git repositories:\n{listing}{dup_note}\n"
            f"Re-run with the specific repo path."
        )


class NoRepoFoundError(RuntimeError):
    """The folder is not a git repo, is not inside one, and contains none."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        super().__init__(
            f"'{folder}' is not a git repository, is not inside one, and contains none "
            f"(searched {_SCAN_DEPTH} levels deep). Use --init-git to let repo-surgeon "
            f"initialize one here (shadow-git, recommended — data files are excluded via "
            f".gitignore), or --staging to work on an isolated copy and get a .patch back."
        )


def resolve_repo(path: str | Path) -> Resolution:
    """Resolve `path` to the git repo root the agents should operate on."""
    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    if _is_repo_root(folder):
        return Resolution(root=str(folder), kind="repo")

    enclosing = _walk_up(folder)
    if enclosing is not None:
        return Resolution(
            root=str(enclosing),
            kind="inside_repo",
            note=f"'{folder.name}' is inside the git repo at {enclosing}; using the repo root.",
        )

    candidates = _scan_down(folder)
    if len(candidates) == 1:
        chosen = candidates[0]
        return Resolution(
            root=chosen.path,
            kind="nested_repo",
            note=f"'{folder.name}' is not a repo itself; using nested repo '{chosen.rel}'.",
        )
    if len(candidates) > 1:
        raise AmbiguousWorkspaceError(folder, candidates)

    raise NoRepoFoundError(folder)


def _is_repo_root(folder: Path) -> bool:
    # .git is a dir in normal repos and a file in worktrees/submodules; both count.
    return (folder / ".git").exists()


def _walk_up(folder: Path) -> Path | None:
    for parent in folder.parents:
        if _is_repo_root(parent):
            return parent
    return None


def _scan_down(folder: Path) -> list[RepoCandidate]:
    """Find nested repos up to _SCAN_DEPTH levels down, without descending into them."""
    found: list[RepoCandidate] = []
    _scan(folder, folder, 1, found)
    found.sort(key=lambda c: c.rel)
    return found


def _scan(base: Path, current: Path, depth: int, found: list[RepoCandidate]) -> None:
    if depth > _SCAN_DEPTH:
        return
    try:
        children = sorted(p for p in current.iterdir() if p.is_dir())
    except (PermissionError, OSError):
        return
    for child in children:
        if child.name in _SKIP_DIRS or child.name.startswith("."):
            continue
        if _is_repo_root(child):
            found.append(
                RepoCandidate(
                    path=str(child),
                    rel=str(child.relative_to(base)),
                    origin=_origin_url(child),
                )
            )
            continue  # a repo's innards are its own; don't surface repos-within-repos
        _scan(base, child, depth + 1, found)


def _origin_url(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    url = result.stdout.strip()
    return url if result.returncode == 0 and url else None


def normalize_origin(url: str) -> str:
    """Canonical form for comparing remotes ('.git' suffix and case differences ignored)."""
    return url.strip().rstrip("/").removesuffix(".git").lower()


def _duplicate_origins(candidates: list[RepoCandidate]) -> set[str]:
    seen: set[str] = set()
    dups: set[str] = set()
    for c in candidates:
        if c.origin is None:
            continue
        key = normalize_origin(c.origin)
        if key in seen:
            dups.add(key)
        seen.add(key)
    return dups
