"""Local-folder workspace backed by a git repository on disk.

Safety model: all edits happen on an isolated `repo-surgeon/<slug>` branch created by
`start_work()` before the first edit. On success the fix is committed there and the
user's original branch is checked out again; on failure the branch is deleted and the
tree restored — either way the user's checkout ends exactly where it started.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from repo_surgeon.models import Patch
from repo_surgeon.workspace.base import Workspace, slugify

_CACHE_DIR = ".repo_surgeon_cache/"


class LocalWorkspace(Workspace):
    """A workspace over a local git repo. Delivers a fix as a branch + commit."""

    def __init__(self, path: str | Path) -> None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise NotADirectoryError(f"Repo path is not a directory: {resolved}")
        if not (resolved / ".git").exists():
            raise ValueError(
                f"{resolved} is not a git repository. Run `git init` there first "
                f"(repo-surgeon delivers fixes as a branch + commit)."
            )
        self._root = resolved
        self._work_branch: str | None = None
        self._original_branch: str | None = None
        self._touched: set[str] = set()  # files the agents wrote; only these + tracked edits commit
        self._ensure_cache_excluded()

    @property
    def root_path(self) -> Path:
        return self._root

    def list_files(self, *, suffix: str | None = None) -> list[str]:
        out = self._git("ls-files")
        files = [line for line in out.splitlines() if line.strip()]
        if suffix is not None:
            files = [f for f in files if f.endswith(suffix)]
        return files

    def read_file(self, rel_path: str) -> str:
        target = self._safe_path(rel_path)
        return target.read_text(encoding="utf-8")

    def write_file(self, rel_path: str, content: str) -> None:
        target = self._safe_path(rel_path)
        target.write_text(content, encoding="utf-8")
        self._touched.add(rel_path)

    def apply_patch(self, patch: Patch) -> bool:
        target = self._safe_path(patch.file_path)
        content = target.read_text(encoding="utf-8")
        if patch.search not in content:
            return False
        # Replace only the first occurrence to keep edits precise.
        updated = content.replace(patch.search, patch.replace, 1)
        target.write_text(updated, encoding="utf-8")
        self._touched.add(patch.file_path)
        return True

    def is_dirty(self) -> bool:
        # Untracked files are ignored: the retry loop's reset() only touches tracked files,
        # so untracked work is never at risk and shouldn't block a run.
        return bool(self._git("status", "--porcelain", "--untracked-files=no").strip())

    def reset(self) -> None:
        # Restore tracked files to HEAD; leaves untracked files (e.g. the KB cache) alone.
        self._git("checkout", "--", ".")

    def start_work(self, title: str) -> str:
        if self._work_branch is not None:
            raise RuntimeError(f"Work already in progress on branch '{self._work_branch}'.")
        self._original_branch = self.current_branch()
        branch = self._unique_branch_name(f"repo-surgeon/{slugify(title)}")
        self._git("checkout", "-b", branch)
        self._work_branch = branch
        return branch

    def deliver(self, *, title: str, body: str) -> str:
        # Robustness for callers that skipped start_work (e.g. direct graph invocation).
        if self._work_branch is None:
            self.start_work(title)
        assert self._work_branch is not None and self._original_branch is not None
        allow_empty = not self.has_changes()
        self.commit(f"{title}\n\n{body}", allow_empty=allow_empty)
        branch = self._work_branch
        self._git("checkout", self._original_branch)
        self._work_branch = None
        self._original_branch = None
        return branch

    def abort_work(self) -> None:
        self.reset()
        self._touched.clear()
        if self._work_branch is None:
            return
        assert self._original_branch is not None
        self._git("checkout", self._original_branch)
        self._git("branch", "-D", self._work_branch)
        self._work_branch = None
        self._original_branch = None

    def current_branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD").strip()

    def commit(self, message: str, *, allow_empty: bool = False) -> None:
        # Stage tracked modifications plus files the agents explicitly wrote — never the
        # user's unrelated untracked files (git add -A would sweep those into the commit,
        # and they'd vanish from the working tree on checkout back).
        self._git("add", "-u")
        for rel_path in sorted(self._touched):
            self._git("add", "--", rel_path)
        self._touched.clear()
        args = ["commit", "-m", message]
        if allow_empty:
            args.insert(1, "--allow-empty")
        self._git(*args)

    def has_changes(self) -> bool:
        """True if the working tree has staged or unstaged changes."""
        return bool(self._git("status", "--porcelain").strip())

    # --- internals -------------------------------------------------------

    def _unique_branch_name(self, base: str) -> str:
        name, n = base, 2
        while self._branch_exists(name):
            name = f"{base}-{n}"
            n += 1
        return name

    def _branch_exists(self, name: str) -> bool:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{name}"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def _ensure_cache_excluded(self) -> None:
        """Keep the KB cache out of commits via the repo-local (unshared) exclude file."""
        exclude = self._root / ".git" / "info" / "exclude"
        try:
            existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
            if _CACHE_DIR not in existing:
                exclude.parent.mkdir(parents=True, exist_ok=True)
                joiner = "" if existing.endswith("\n") or not existing else "\n"
                exclude.write_text(f"{existing}{joiner}{_CACHE_DIR}\n", encoding="utf-8")
        except OSError:
            # Non-fatal: worst case the cache dir shows up as untracked noise.
            pass

    def _safe_path(self, rel_path: str) -> Path:
        """Resolve a repo-relative path, refusing anything that escapes the repo root."""
        candidate = (self._root / rel_path).resolve()
        if not candidate.is_relative_to(self._root):
            raise ValueError(f"Path escapes repository root: {rel_path}")
        return candidate

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout
