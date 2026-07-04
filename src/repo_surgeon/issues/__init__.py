"""Issue-source abstraction: normalize GitHub issues and local descriptions to one Issue."""

from repo_surgeon.issues.base import IssueSource
from repo_surgeon.issues.factory import open_issue_source
from repo_surgeon.issues.local import LocalIssueSource

__all__ = ["IssueSource", "LocalIssueSource", "open_issue_source"]
