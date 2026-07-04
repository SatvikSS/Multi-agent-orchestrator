"""Shared file-filtering rules for plain (non-git) project folders.

Both shadow-git (.gitignore generation) and staging mode (what to copy) must keep
data artifacts out: the user's project folders run to gigabytes of CSVs, models, and
venvs that must never enter git or be copied around.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

DATA_EXTENSIONS = {
    ".csv", ".tsv", ".parquet", ".feather", ".xlsx", ".xls",
    ".pkl", ".pickle", ".joblib",
    ".h5", ".hdf5", ".npy", ".npz",
    ".pt", ".pth", ".onnx", ".pb", ".safetensors",
    ".db", ".sqlite", ".sqlite3",
    ".zip", ".gz", ".tar", ".rar", ".7z",
    ".log",
}

JUNK_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".ipynb_checkpoints", ".repo_surgeon_cache", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", ".idea", ".vscode",
}

DEFAULT_MAX_FILE_MB = 10.0

# The block appended to .gitignore by shadow-git init; the marker makes it idempotent.
GITIGNORE_MARKER = "# --- repo-surgeon (auto-generated) ---"


def gitignore_block(oversized: list[str]) -> str:
    """Build the .gitignore block: junk dirs, data extensions, explicit oversized files."""
    lines = [GITIGNORE_MARKER]
    lines += sorted(f"{d}/" for d in JUNK_DIRS if d != ".git")
    lines += sorted(f"*{ext}" for ext in DATA_EXTENSIONS)
    lines += [f"/{rel}" for rel in sorted(oversized)]
    return "\n".join(lines) + "\n"


def iter_project_files(
    folder: Path, *, max_file_mb: float = DEFAULT_MAX_FILE_MB
) -> Iterator[str]:
    """Yield repo-relative paths of code/config files worth tracking or copying.

    Skips junk dirs, hidden dirs, data extensions, and files over the size cap.
    """
    for rel in _walk(folder, folder):
        full = folder / rel
        if full.suffix.lower() in DATA_EXTENSIONS:
            continue
        try:
            if full.stat().st_size > max_file_mb * 1024 * 1024:
                continue
        except OSError:
            continue
        yield rel


def oversized_files(folder: Path, *, max_file_mb: float = DEFAULT_MAX_FILE_MB) -> list[str]:
    """Non-data files exceeding the size cap — listed explicitly in .gitignore."""
    result = []
    for rel in _walk(folder, folder):
        full = folder / rel
        if full.suffix.lower() in DATA_EXTENSIONS:
            continue  # already covered by extension patterns
        try:
            if full.stat().st_size > max_file_mb * 1024 * 1024:
                result.append(rel)
        except OSError:
            continue
    return result


def _walk(base: Path, current: Path) -> Iterator[str]:
    try:
        entries = sorted(current.iterdir())
    except (PermissionError, OSError):
        return
    for entry in entries:
        if entry.is_dir():
            if entry.name in JUNK_DIRS or entry.name.startswith("."):
                continue
            yield from _walk(base, entry)
        elif entry.is_file() and not entry.name.startswith("."):
            yield str(entry.relative_to(base))
