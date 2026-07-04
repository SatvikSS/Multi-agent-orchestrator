"""Evaluation harness: run the orchestrator over a benchmark and report metrics."""

from repo_surgeon.evaluation.harness import (
    EvalCase,
    EvalResult,
    EvalSummary,
    builtin_cases_dir,
    load_cases,
    run_eval,
)

__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalSummary",
    "builtin_cases_dir",
    "load_cases",
    "run_eval",
]
