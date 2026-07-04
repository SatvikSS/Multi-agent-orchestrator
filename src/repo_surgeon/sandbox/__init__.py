"""Sandbox layer: run the target repo's tests in an isolated Docker container."""

from __future__ import annotations

from collections.abc import Callable

from repo_surgeon.config import Sandbox
from repo_surgeon.models import TestResult
from repo_surgeon.sandbox.docker_runner import run_pytest
from repo_surgeon.sandbox.report import parse_pytest_output
from repo_surgeon.workspace.base import Workspace

# A function that runs the tests for a workspace and returns the result. Injected into the
# graph so tests can supply a fake without Docker; `make_docker_runner` is the production one.
TestRunner = Callable[[Workspace], TestResult]

__all__ = ["TestRunner", "make_docker_runner", "parse_pytest_output", "run_pytest"]


def make_docker_runner(cfg: Sandbox) -> TestRunner:
    """Build the production test runner: run pytest in Docker against the workspace root."""

    def runner(workspace: Workspace) -> TestResult:
        return run_pytest(workspace.root_path, cfg)

    return runner
