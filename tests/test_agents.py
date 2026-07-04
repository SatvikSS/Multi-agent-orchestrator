"""Tests for the individual agents with fake chat models."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from repo_surgeon.agents.localizer import localize
from repo_surgeon.agents.planner import make_plan
from repo_surgeon.agents.writer import write_patches
from repo_surgeon.issues import LocalIssueSource


def _issue():
    return LocalIssueSource("Fix add() in calc.py").fetch()


def test_localizer_returns_valid_choice() -> None:
    llm = FakeListChatModel(responses=["calc.py"])
    chosen = localize(_issue(), ["calc.py", "test_calc.py"], "context", llm)
    assert chosen == ["calc.py"]


def test_localizer_ignores_invalid_and_falls_back() -> None:
    llm = FakeListChatModel(responses=["not_a_real_file.py"])
    chosen = localize(_issue(), ["calc.py", "test_calc.py"], "context", llm)
    assert chosen == ["calc.py"]  # falls back to the top candidate


def test_localizer_empty_candidates() -> None:
    llm = FakeListChatModel(responses=["anything"])
    assert localize(_issue(), [], "context", llm) == []


def test_planner_returns_text() -> None:
    llm = FakeListChatModel(responses=["  - change add to subtract  "])
    assert make_plan(_issue(), "context", llm) == "- change add to subtract"


def test_writer_parses_patches() -> None:
    block = "calc.py\n<<<<<<< SEARCH\na + b\n=======\na - b\n>>>>>>> REPLACE"
    llm = FakeListChatModel(responses=[block])
    patches = write_patches(_issue(), "plan", "context", llm)
    assert len(patches) == 1
    assert patches[0].file_path == "calc.py"
    assert patches[0].replace == "a - b"
