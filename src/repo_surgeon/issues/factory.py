"""Pick the right IssueSource from a reference string.

A github.com issue URL or "owner/repo#123" → GitHubIssueSource (Phase 4).
Anything else → LocalIssueSource (a file path or a raw description).
"""

from __future__ import annotations

import re

from repo_surgeon.issues.base import IssueSource
from repo_surgeon.issues.github import GitHubIssueSource
from repo_surgeon.issues.local import LocalIssueSource

_GH_ISSUE_URL = re.compile(r"github\.com/[\w.-]+/[\w.-]+/issues/\d+", re.IGNORECASE)
_GH_SHORTHAND = re.compile(r"^[\w.-]+/[\w.-]+#\d+$")


def open_issue_source(ref: str, *, github_token: str | None = None) -> IssueSource:
    """Resolve `ref` to a concrete IssueSource."""
    if _GH_ISSUE_URL.search(ref) or _GH_SHORTHAND.match(ref):
        return GitHubIssueSource(ref, token=github_token)
    return LocalIssueSource(ref)
