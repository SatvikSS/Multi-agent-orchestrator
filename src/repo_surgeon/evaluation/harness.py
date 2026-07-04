"""Run the orchestrator over a benchmark of seeded-bug cases and compute metrics.

A case is a directory holding a small broken repo plus a `meta.json` with the issue text
and a failing test. The harness copies each case to a fresh temp git repo, runs the full
pipeline (real Docker test execution by default), and records whether the suite went green
along with attempts and wall-time. The headline metric is the resolved rate.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel

from repo_surgeon.config import AppConfig, load_config
from repo_surgeon.llm.factory import LLMProvider, build_llm
from repo_surgeon.models import RunStatus
from repo_surgeon.runner import run_issue
from repo_surgeon.sandbox import TestRunner

logger = logging.getLogger(__name__)

_META = "meta.json"
_SKIP = {_META, "__pycache__", ".git"}


class EvalCase(BaseModel):
    """One benchmark case: a template repo directory plus its issue."""

    model_config = {"frozen": True}

    name: str
    path: str
    issue: str


class EvalResult(BaseModel):
    """Outcome of running one case."""

    model_config = {"frozen": True}

    name: str
    resolved: bool
    produced_patch: bool
    attempts: int
    wall_seconds: float
    status: str
    error: str | None = None


class EvalSummary(BaseModel):
    """Aggregate metrics over a run."""

    model_config = {"frozen": True}

    total: int
    resolved: int
    resolved_rate: float
    produced_patch_rate: float
    avg_attempts: float
    avg_wall_seconds: float
    results: list[EvalResult]


def builtin_cases_dir() -> Path:
    return Path(__file__).parent / "cases"


def load_cases(cases_dir: str | Path) -> list[EvalCase]:
    """Load every case (a subdir with a meta.json) under `cases_dir`."""
    base = Path(cases_dir).expanduser().resolve()
    cases: list[EvalCase] = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        meta_file = sub / _META
        if not meta_file.is_file():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        cases.append(EvalCase(name=sub.name, path=str(sub), issue=meta["issue"]))
    return cases


def run_eval(
    cases: list[EvalCase],
    *,
    config: AppConfig | None = None,
    llm_provider: LLMProvider = build_llm,
    test_runner: TestRunner | None = None,
) -> EvalSummary:
    """Run every case and aggregate the results."""
    cfg = config or load_config()
    results = [_run_case(c, cfg, llm_provider, test_runner) for c in cases]
    return _summarize(results)


def _run_case(
    case: EvalCase,
    cfg: AppConfig,
    llm_provider: LLMProvider,
    test_runner: TestRunner | None,
) -> EvalResult:
    import tempfile

    with tempfile.TemporaryDirectory(prefix=f"repo-surgeon-eval-{case.name}-") as tmp:
        repo = Path(tmp) / "repo"
        _materialize(Path(case.path), repo)

        start = time.perf_counter()
        try:
            final = run_issue(
                case.issue,
                str(repo),
                config=cfg,
                llm_provider=llm_provider,
                test_runner=test_runner,
            )
            elapsed = time.perf_counter() - start
            return EvalResult(
                name=case.name,
                resolved=final.get("status") == RunStatus.RESOLVED,
                produced_patch=bool(final.get("patches")),
                attempts=int(final.get("attempts", 0)),
                wall_seconds=round(elapsed, 2),
                status=str(final.get("status", RunStatus.FAILED)),
            )
        except Exception as exc:  # noqa: BLE001 - a crashing case must not sink the whole run
            logger.warning("Eval case '%s' errored: %s", case.name, exc)
            return EvalResult(
                name=case.name,
                resolved=False,
                produced_patch=False,
                attempts=0,
                wall_seconds=round(time.perf_counter() - start, 2),
                status="error",
                error=str(exc),
            )


def _materialize(template: Path, dest: Path) -> None:
    """Copy the case files (minus meta.json) into a fresh git repo with a baseline commit."""
    dest.mkdir(parents=True)
    for item in template.iterdir():
        if item.name in _SKIP:
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(item, target)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=dest, check=True, capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "eval@repo-surgeon.local")
    git("config", "user.name", "repo-surgeon-eval")
    git("add", "-A")
    git("commit", "-q", "-m", "eval baseline")


def _summarize(results: list[EvalResult]) -> EvalSummary:
    total = len(results)
    resolved = sum(1 for r in results if r.resolved)
    produced = sum(1 for r in results if r.produced_patch)
    return EvalSummary(
        total=total,
        resolved=resolved,
        resolved_rate=round(resolved / total, 3) if total else 0.0,
        produced_patch_rate=round(produced / total, 3) if total else 0.0,
        avg_attempts=round(sum(r.attempts for r in results) / total, 2) if total else 0.0,
        avg_wall_seconds=round(sum(r.wall_seconds for r in results) / total, 2) if total else 0.0,
        results=results,
    )
