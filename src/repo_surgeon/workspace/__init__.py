"""Workspace abstraction: a uniform interface over local folders and GitHub repos."""

from repo_surgeon.workspace.base import Workspace
from repo_surgeon.workspace.factory import open_workspace
from repo_surgeon.workspace.local import LocalWorkspace

__all__ = ["Workspace", "LocalWorkspace", "open_workspace"]
