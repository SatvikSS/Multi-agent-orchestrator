"""Tests for issue→project routing over the registry."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from repo_surgeon.issues import LocalIssueSource
from repo_surgeon.registry import ProjectEntry, Registry
from repo_surgeon.routing import pick_with_llm, route_issue


def _entry(name: str, path: str = "/tmp/x", **overrides) -> ProjectEntry:
    defaults = {"name": name, "path": path, "kind": "repo", "py_files": 10, "total_files": 20}
    defaults.update(overrides)
    return ProjectEntry(**defaults)


def _registry(*entries: ProjectEntry) -> Registry:
    return Registry(root="/tmp", generated_at="2026-07-03T00:00:00+00:00", projects=list(entries))


def test_name_match_ranks_first() -> None:
    registry = _registry(
        _entry("ontology"),
        _entry("cpo-dashboard"),
        _entry("digital-twin"),
    )
    issue = LocalIssueSource("The ontology loader crashes on empty input").fetch()
    ranked = route_issue(issue, registry)
    assert ranked[0].entry.name == "ontology"
    assert ranked[0].score > 0


def test_camel_case_names_match_word_split_issues() -> None:
    registry = _registry(
        _entry("CPO_v3/chillerPerformanceCalculator", kind="nested_repo"),
        _entry("cpo_dashboard"),
    )
    issue = LocalIssueSource("The chiller performance calculator returns wrong COP").fetch()
    ranked = route_issue(issue, registry)
    assert ranked[0].entry.name == "CPO_v3/chillerPerformanceCalculator"
    assert ranked[0].score >= 9  # chiller + performance + calculator


def test_version_suffix_disambiguates() -> None:
    registry = _registry(_entry("cpo_dashboard"), _entry("cpo_dashboard_v2"))
    issue = LocalIssueSource("cpo dashboard v2 chart does not refresh").fetch()
    ranked = route_issue(issue, registry)
    assert ranked[0].entry.name == "cpo_dashboard_v2"


def test_summary_keywords_contribute() -> None:
    registry = _registry(
        _entry("proj-a", summary="Chiller performance calculator for HVAC plants."),
        _entry("proj-b", summary="Airflow DAGs for billing exports."),
    )
    issue = LocalIssueSource("chiller performance numbers look wrong").fetch()
    ranked = route_issue(issue, registry)
    assert ranked[0].entry.name == "proj-a"


def test_hint_path_bonus(tmp_path: Path) -> None:
    proj_a = tmp_path / "a"
    proj_b = tmp_path / "b"
    (proj_a / "src").mkdir(parents=True)
    proj_b.mkdir()
    (proj_a / "src" / "pager.py").write_text("pass\n", encoding="utf-8")
    registry = _registry(
        _entry("alpha", path=str(proj_a)),
        _entry("beta", path=str(proj_b)),
    )
    issue = LocalIssueSource("Crash in pager.py when listing items").fetch()
    ranked = route_issue(issue, registry)
    assert ranked[0].entry.name == "alpha"
    assert ranked[0].score >= 5


def test_containers_and_stale_duplicates_excluded() -> None:
    registry = _registry(
        _entry("cpc-models", kind="container"),
        _entry("cpc-models/old-clone", kind="nested_repo", duplicate_group="g", is_canonical=False),
        _entry("cpc-models/new-clone", kind="nested_repo", duplicate_group="g", is_canonical=True),
    )
    issue = LocalIssueSource("cpc models clone issue").fetch()
    ranked = route_issue(issue, registry)
    names = [r.entry.name for r in ranked]
    assert "cpc-models" not in names
    assert "cpc-models/old-clone" not in names
    assert "cpc-models/new-clone" in names


def test_no_score_falls_back_to_alphabetical_head() -> None:
    registry = _registry(_entry("zeta"), _entry("alpha"))
    issue = LocalIssueSource("completely unrelated words qqq").fetch()
    ranked = route_issue(issue, registry)
    assert [r.entry.name for r in ranked] == ["alpha", "zeta"]
    assert all(r.score == 0 for r in ranked)


def test_empty_registry_returns_empty() -> None:
    issue = LocalIssueSource("anything").fetch()
    assert route_issue(issue, _registry()) == []


def test_pick_with_llm_selects_by_number() -> None:
    issue = LocalIssueSource("pick the second").fetch()
    candidates = route_issue(issue, _registry(_entry("alpha"), _entry("beta")))
    llm = FakeListChatModel(responses=["2"])
    assert pick_with_llm(issue, candidates, llm).entry.name == "beta"


def test_pick_with_llm_invalid_answer_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    issue = LocalIssueSource("pick something").fetch()
    candidates = route_issue(issue, _registry(_entry("alpha"), _entry("beta")))
    llm = FakeListChatModel(responses=["neither of these"])
    assert pick_with_llm(issue, candidates, llm) == candidates[0]


def test_pick_with_llm_single_candidate_short_circuits() -> None:
    issue = LocalIssueSource("only one").fetch()
    candidates = route_issue(issue, _registry(_entry("alpha")))
    llm = FakeListChatModel(responses=["999"])  # never consulted
    assert pick_with_llm(issue, candidates, llm).entry.name == "alpha"