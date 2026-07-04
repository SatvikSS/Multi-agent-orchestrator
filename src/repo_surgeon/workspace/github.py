"""GitHub-repo workspace: clone to a temp checkout, deliver by push + pull request.

Inherits the whole LocalWorkspace edit/branch/safety machinery over the clone; only
acquisition (clone) and delivery (push + PR) differ. Requires push access to the repo
(your own repos or ones you collaborate on) — fork-based flows are future work.

`clone_url` and `github_client` are injectable so tests can run against a local bare
repository with a fake PR client — no network involved.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from repo_surgeon.workspace.local import LocalWorkspace

_REPO_URL = re.compile(r"github\.com[/:]([\w.-]+/[\w.-]+?)(?:\.git)?/?$", re.IGNORECASE)
_OWNER_REPO = re.compile(r"^[\w.-]+/[\w.-]+$")


def parse_repo_ref(ref: str) -> str | None:
    """Return 'owner/repo' if `ref` looks like a GitHub repository reference."""
    match = _REPO_URL.search(ref.strip())
    if match:
        return match.group(1)
    if _OWNER_REPO.match(ref.strip()):
        return ref.strip()
    return None


class GitHubWorkspace(LocalWorkspace):
    """Clone of a GitHub repo; fixes are delivered as a pushed branch + pull request."""

    def __init__(
        self,
        repo: str,
        *,
        token: str | None = None,
        clone_url: str | None = None,
        github_client: Any = None,
    ) -> None:
        full_name = parse_repo_ref(repo)
        if full_name is None:
            raise ValueError(f"Not a GitHub repository reference: '{repo}'.")
        self._full_name = full_name
        self._token = token
        self._client = github_client

        self._tmp = Path(tempfile.mkdtemp(prefix="repo-surgeon-gh-"))
        clone_path = self._tmp / "repo"
        url = clone_url or self._https_url()
        result = subprocess.run(
            ["git", "clone", "-q", url, str(clone_path)], capture_output=True, text=True
        )
        if result.returncode != 0:
            shutil.rmtree(self._tmp, ignore_errors=True)
            raise RuntimeError(
                f"Could not clone {self._full_name}: {_scrub(result.stderr.strip(), self._token)}"
            )

        super().__init__(clone_path)
        self._default_branch = self.current_branch()

    def deliver(self, *, title: str, body: str) -> str:
        """Commit on the work branch, push it, and open a PR. Returns the PR URL.

        Without a token/client a PR cannot be opened; the branch is still pushed and
        its name returned so the user can open the PR manually.
        """
        branch = super().deliver(title=title, body=body)
        self._git("push", "-q", "-u", "origin", branch)

        client = self._client or self._build_client()
        if client is None:
            return branch
        pr = client.get_repo(self._full_name).create_pull(
            title=title, body=body, head=branch, base=self._default_branch
        )
        return str(pr.html_url)

    def cleanup(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    # --- internals -------------------------------------------------------

    def _https_url(self) -> str:
        if self._token:
            return f"https://x-access-token:{self._token}@github.com/{self._full_name}.git"
        return f"https://github.com/{self._full_name}.git"

    def _build_client(self) -> Any:
        if not self._token:
            return None
        from github import Auth, Github  # imported lazily: only needed for real PRs

        return Github(auth=Auth.Token(self._token))


def _scrub(text: str, token: str | None) -> str:
    """Never let a token leak into error messages."""
    return text.replace(token, "***") if token else text
