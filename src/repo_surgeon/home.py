"""repo-surgeon's own data directory (staging copies, patches, backups, registry).

Defaults to ~/.repo-surgeon; overridable via REPO_SURGEON_HOME (used by tests).
Kept out of the user's project folders by design.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "REPO_SURGEON_HOME"


def home_dir() -> Path:
    base = os.environ.get(_ENV_VAR)
    path = Path(base).expanduser() if base else Path.home() / ".repo-surgeon"
    path.mkdir(parents=True, exist_ok=True)
    return path


def subdir(name: str) -> Path:
    path = home_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
