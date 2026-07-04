"""Run a target repo's test suite inside a hardened, ephemeral Docker container.

Design (SWE-bench-aligned): a base image with pytest is built once (network on, at build
time). Tests then run from that image with the repo mounted read-only, **network disabled**,
CPU/memory/pids capped, and a wall-clock timeout — so untrusted, LLM-generated code cannot
reach the network, exhaust the host, or mutate the working tree.
"""

from __future__ import annotations

import hashlib
import io
import shlex
import tempfile
from pathlib import Path

import docker
from docker.errors import ImageNotFound

from repo_surgeon.config import Sandbox
from repo_surgeon.models import TestResult
from repo_surgeon.sandbox.report import parse_pytest_output

_MOUNT = "/workspace"


def run_pytest(repo_path: Path, cfg: Sandbox) -> TestResult:
    """Run the configured test command against `repo_path` in a sandboxed container."""
    client = docker.from_env()
    image = _ensure_image(client, repo_path, cfg)

    container = client.containers.run(
        image=image,
        command=shlex.split(cfg.test_command),
        volumes={str(repo_path): {"bind": _MOUNT, "mode": "ro"}},
        working_dir=_MOUNT,
        network_disabled=not cfg.network_enabled,
        mem_limit=cfg.mem_limit,
        nano_cpus=int(cfg.cpus * 1_000_000_000),
        pids_limit=cfg.pids_limit,
        environment={"PYTHONDONTWRITEBYTECODE": "1"},
        tmpfs={"/tmp": ""},
        detach=True,
    )
    try:
        try:
            status = container.wait(timeout=cfg.timeout_seconds)
        except Exception:  # noqa: BLE001 - docker/requests timeout surfaces in several shapes
            container.kill()
            return TestResult(
                passed=False,
                failures=(f"test run exceeded {cfg.timeout_seconds}s timeout",),
                raw_output="timeout",
            )
        logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
        return parse_pytest_output(logs, int(status.get("StatusCode", 1)))
    finally:
        container.remove(force=True)


def _ensure_image(client: docker.DockerClient, repo_path: Path, cfg: Sandbox) -> str:
    """Return an image tag with pytest (+ repo requirements if present) installed."""
    base = f"repo-surgeon-sandbox:py{cfg.python_version.replace('.', '')}"
    base_dockerfile = (
        f"FROM python:{cfg.python_version}-slim\nRUN pip install --no-cache-dir pytest\n"
    )
    _build_if_missing(client, tag=base, dockerfile=base_dockerfile)

    requirements = repo_path / "requirements.txt"
    if not (cfg.install_requirements and requirements.exists()):
        return base

    # Per-repo image that layers the requirements on top of the base, cached by content hash.
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()[:12]
    repo_tag = f"repo-surgeon-sandbox:req-{digest}"
    try:
        client.images.get(repo_tag)
        return repo_tag
    except ImageNotFound:
        pass

    with tempfile.TemporaryDirectory() as ctx:
        ctx_path = Path(ctx)
        (ctx_path / "requirements.txt").write_bytes(requirements.read_bytes())
        (ctx_path / "Dockerfile").write_text(
            f"FROM {base}\nCOPY requirements.txt .\n"
            f"RUN pip install --no-cache-dir -r requirements.txt\n",
            encoding="utf-8",
        )
        client.images.build(path=str(ctx_path), tag=repo_tag, rm=True)
    return repo_tag


def _build_if_missing(client: docker.DockerClient, *, tag: str, dockerfile: str) -> None:
    try:
        client.images.get(tag)
    except ImageNotFound:
        client.images.build(fileobj=io.BytesIO(dockerfile.encode("utf-8")), tag=tag, rm=True)
