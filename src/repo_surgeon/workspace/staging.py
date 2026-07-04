"""Staging mode: run the agents on an isolated copy — the original folder is never written.

For plain folders the user doesn't want touched at all: code files (data excluded, so MBs
not GBs) are copied to ~/.repo-surgeon/staging/<name>-<hash>, shadow-git is initialized
there, and the normal branch/test/deliver loop runs on the copy. Delivery produces a
.patch file under ~/.repo-surgeon/patches/; `surgeon apply` applies it back with
per-file backups.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from pathlib import Path

from repo_surgeon.home import subdir
from repo_surgeon.workspace.base import slugify
from repo_surgeon.workspace.ignore import DEFAULT_MAX_FILE_MB, iter_project_files
from repo_surgeon.workspace.local import LocalWorkspace
from repo_surgeon.workspace.shadow import init_shadow_git


class StagingWorkspace(LocalWorkspace):
    """A LocalWorkspace over a fresh staged copy; delivers a .patch instead of a branch."""

    def __init__(self, source: str | Path, *, max_file_mb: float = DEFAULT_MAX_FILE_MB) -> None:
        src = Path(source).expanduser().resolve()
        if not src.is_dir():
            raise NotADirectoryError(f"Not a directory: {src}")
        self._source = src

        stage = subdir("staging") / f"{src.name}-{_short_hash(src)}"
        if stage.exists():
            shutil.rmtree(stage)  # fresh copy every run; staging is disposable
        stage.mkdir(parents=True)

        copied = 0
        for rel in iter_project_files(src, max_file_mb=max_file_mb):
            target = stage / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, target)
            copied += 1
        if copied == 0:
            raise ValueError(f"No code files found to stage in {src}.")

        init_shadow_git(stage, max_file_mb=max_file_mb)
        super().__init__(stage)

    @property
    def source_path(self) -> Path:
        return self._source

    def deliver(self, *, title: str, body: str) -> str:
        """Commit on the staged work branch, then export the fix as a .patch file."""
        branch = super().deliver(title=title, body=body)
        base = self.current_branch()
        diff = self._git("diff", f"{base}..{branch}")
        patch_path = subdir("patches") / f"{self._source.name}-{slugify(title)}.patch"
        patch_path.write_text(diff, encoding="utf-8")
        return str(patch_path)


def apply_patch_file(patch_path: str | Path, target_folder: str | Path) -> list[str]:
    """Apply a staged-run patch to the original folder, backing up affected files first.

    Returns the repo-relative paths of the files the patch touches. Backups land in
    ~/.repo-surgeon/backups/<timestamp>/.
    """
    patch = Path(patch_path).expanduser().resolve()
    target = Path(target_folder).expanduser().resolve()
    if not patch.is_file():
        raise FileNotFoundError(f"Patch not found: {patch}")
    if not target.is_dir():
        raise NotADirectoryError(f"Target is not a directory: {target}")

    touched = _files_in_patch(patch.read_text(encoding="utf-8"))
    if not touched:
        raise ValueError(f"{patch} contains no file changes.")

    backup_dir = subdir("backups") / time.strftime("%Y%m%d-%H%M%S")
    for rel in touched:
        source_file = target / rel
        if source_file.exists():
            backup_target = backup_dir / rel
            backup_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, backup_target)

    # `git apply` works as a plain patch tool even outside a repository.
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch)],
        cwd=target,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Patch did not apply cleanly: {result.stderr.strip()}\n"
            f"Backups (untouched originals) are in {backup_dir}."
        )
    return touched


def _files_in_patch(diff: str) -> list[str]:
    files = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[len("+++ b/") :].strip())
    return sorted(set(files))


def _short_hash(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
