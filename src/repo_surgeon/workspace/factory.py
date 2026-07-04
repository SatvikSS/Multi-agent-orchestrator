"""Pick the right Workspace implementation from a reference string.

Local paths go through the resolver (walk-up / scan-down for nested repos) before
becoming a LocalWorkspace. Anything shaped like "owner/repo" or a github.com URL
becomes a GitHubWorkspace (Phase 4).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from repo_surgeon.workspace.base import Workspace
from repo_surgeon.workspace.github import GitHubWorkspace
from repo_surgeon.workspace.local import LocalWorkspace
from repo_surgeon.workspace.resolver import resolve_repo

logger = logging.getLogger(__name__)

_GITHUB_URL = re.compile(r"^(https?://)?github\.com/", re.IGNORECASE)
_OWNER_REPO = re.compile(r"^[\w.-]+/[\w.-]+$")


def open_workspace(ref: str, *, github_token: str | None = None) -> Workspace:
    """Resolve `ref` to a concrete Workspace.

    May raise AmbiguousWorkspaceError (several nested repos — caller should let the
    user pick) or NoRepoFoundError (plain folder with no git anywhere).
    """
    expanded = Path(ref).expanduser()
    if expanded.is_dir():
        resolution = resolve_repo(expanded)
        if resolution.note:
            logger.info("%s", resolution.note)
        return LocalWorkspace(resolution.root)
    if _GITHUB_URL.search(ref) or _OWNER_REPO.match(ref):
        return GitHubWorkspace(ref, token=github_token)
    raise ValueError(
        f"Could not interpret repo reference '{ref}'. "
        f"Expected an existing local directory or a GitHub 'owner/repo'."
    )
