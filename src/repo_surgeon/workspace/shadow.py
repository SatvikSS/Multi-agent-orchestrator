"""Shadow-git: turn a plain project folder into a git repo, safely and reversibly.

Steps (run only with the user's consent, via --init-git or an interactive prompt):
  1. Write/extend .gitignore so data files, venvs, and oversized files never enter git
     (an existing .gitignore is appended to, never rewritten; the marker keeps it
     idempotent).
  2. `git init` + a baseline commit of the code files. The baseline IS the backup:
     every later state is recoverable, and deleting .git fully undoes shadow-git.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from repo_surgeon.workspace.ignore import (
    DEFAULT_MAX_FILE_MB,
    GITIGNORE_MARKER,
    gitignore_block,
    oversized_files,
)


def init_shadow_git(folder: str | Path, *, max_file_mb: float = DEFAULT_MAX_FILE_MB) -> Path:
    """Initialize git in a plain folder with a data-safe .gitignore and baseline commit."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    if (root / ".git").exists():
        raise ValueError(f"{root} is already a git repository.")

    _extend_gitignore(root, max_file_mb=max_file_mb)
    _git(root, "init", "-q")
    _ensure_identity(root)
    _git(root, "add", "-A")  # .gitignore keeps data/oversized files out
    _git(root, "commit", "-q", "-m", "repo-surgeon baseline (code files only)")
    return root


def _extend_gitignore(root: Path, *, max_file_mb: float) -> None:
    gitignore = root / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    if GITIGNORE_MARKER in existing:
        return  # already initialized once; keep whatever is there
    block = gitignore_block(oversized_files(root, max_file_mb=max_file_mb))
    joiner = "" if not existing or existing.endswith("\n") else "\n"
    gitignore.write_text(f"{existing}{joiner}{block}", encoding="utf-8")


def _ensure_identity(root: Path) -> None:
    """Guarantee commits work even when no global git identity is configured."""
    probe = subprocess.run(
        ["git", "config", "user.email"], cwd=root, capture_output=True, text=True
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        _git(root, "config", "user.email", "repo-surgeon@local")
        _git(root, "config", "user.name", "repo-surgeon")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout
