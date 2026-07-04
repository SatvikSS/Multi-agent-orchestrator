"""GitHub issue source: fetch title/body/comments via PyGithub, normalize to an Issue.

Accepts a full issue URL (https://github.com/owner/repo/issues/123) or the shorthand
"owner/repo#123". Works anonymously for public repos (rate-limited); a token from
GITHUB_TOKEN raises the limit and unlocks private repos.
"""

from __future__ import annotations

import re
from itertools import islice
from typing import Any

from repo_surgeon.issues.base import IssueSource, extract_hints
from repo_surgeon.models import Issue

_ISSUE_URL = re.compile(r"github\.com/([\w.-]+/[\w.-]+)/issues/(\d+)", re.IGNORECASE)
_SHORTHAND = re.compile(r"^([\w.-]+/[\w.-]+)#(\d+)$")
_MAX_COMMENTS = 5


def parse_issue_ref(ref: str) -> tuple[str, int] | None:
    """Return (owner/repo, issue_number) if `ref` looks like a GitHub issue reference."""
    match = _ISSUE_URL.search(ref) or _SHORTHAND.match(ref.strip())
    if match is None:
        return None
    return match.group(1), int(match.group(2))


class GitHubIssueSource(IssueSource):
    """Fetch a GitHub issue (plus its first few comments) as a normalized Issue.

    `client` is injectable for tests; production builds a PyGithub client lazily so the
    dependency is only imported when actually fetching from GitHub.
    """

    def __init__(self, ref: str, *, token: str | None = None, client: Any = None) -> None:
        self._ref = ref
        self._token = token
        self._client = client

    def fetch(self) -> Issue:
        parsed = parse_issue_ref(self._ref)
        if parsed is None:
            raise ValueError(
                f"Not a GitHub issue reference: '{self._ref}'. "
                f"Expected an issue URL or 'owner/repo#123'."
            )
        owner_repo, number = parsed

        client = self._client or self._build_client()
        gh_issue = client.get_repo(owner_repo).get_issue(number)

        title = gh_issue.title or f"Issue #{number}"
        parts = [gh_issue.body or ""]
        try:
            for comment in islice(gh_issue.get_comments(), _MAX_COMMENTS):
                if comment.body:
                    parts.append(comment.body)
        except Exception:  # noqa: BLE001 - comments are enrichment, never fatal
            pass
        body = "\n\n".join(p for p in parts if p.strip())

        paths, symbols = extract_hints(f"{title}\n{body}")
        return Issue(
            title=title,
            body=body,
            source=f"github:{owner_repo}#{number}",
            hint_paths=paths,
            hint_symbols=symbols,
        )

    def _build_client(self) -> Any:
        from github import Auth, Github  # imported lazily: only needed for real fetches

        if self._token:
            return Github(auth=Auth.Token(self._token))
        return Github()
