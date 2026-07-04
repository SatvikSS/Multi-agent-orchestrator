"""Abstract Workspace: every agent talks to this, never to git/GitHub directly.

Concrete implementations (LocalWorkspace, GitHubWorkspace) differ only in how the
code gets on disk and how a fix is delivered (branch+commit vs push+PR). Agents are
identical across both.

Work lifecycle (safety model):
  1. `is_dirty()` — preflight; the runner refuses to start on uncommitted changes.
  2. `start_work(title)` — create and switch to an isolated work branch BEFORE any edit,
     so the user's checked-out branch is never touched.
  3. `deliver(title, body)` — commit the fix on the work branch and return the user to
     where they started; or `abort_work()` — discard edits, return, delete the branch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from repo_surgeon.models import Patch


class DirtyWorkspaceError(RuntimeError):
    """Raised when a run is attempted on a workspace with uncommitted changes."""


def slugify(title: str, *, limit: int = 40) -> str:
    """Turn an issue title into a branch-safe slug."""
    slug = "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)[:limit].strip("-")
    return slug or "fix"


class Workspace(ABC):
    """A checked-out codebase the agents can read, edit, and deliver against."""

    @property
    @abstractmethod
    def root_path(self) -> Path:
        """Absolute path to the working tree on disk."""

    @abstractmethod
    def list_files(self, *, suffix: str | None = None) -> list[str]:
        """Repo-relative paths of tracked source files, optionally filtered by suffix."""

    @abstractmethod
    def read_file(self, rel_path: str) -> str:
        """Return the full text of a repo-relative file."""

    @abstractmethod
    def write_file(self, rel_path: str, content: str) -> None:
        """Overwrite a repo-relative file with `content`."""

    @abstractmethod
    def apply_patch(self, patch: Patch) -> bool:
        """Apply a single search-replace edit. Returns True if it applied cleanly."""

    @abstractmethod
    def is_dirty(self) -> bool:
        """True if tracked files have uncommitted changes (untracked files don't count)."""

    @abstractmethod
    def reset(self) -> None:
        """Discard uncommitted edits to tracked files, restoring the last committed state."""

    @abstractmethod
    def start_work(self, title: str) -> str:
        """Create and switch to a fresh work branch for this run. Returns the branch name."""

    @abstractmethod
    def deliver(self, *, title: str, body: str) -> str:
        """Commit the fix on the work branch and return the user to their original branch.

        Returns a delivery reference (branch name or PR URL).
        """

    @abstractmethod
    def abort_work(self) -> None:
        """Discard edits, return to the original branch, and delete the work branch."""

    def cleanup(self) -> None:
        """Release any temporary resources (e.g. cloned checkouts). No-op by default."""
        return None
