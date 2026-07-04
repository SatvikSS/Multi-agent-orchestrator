"""Streamlit demo: type a problem, watch the agents fix it live.

Run locally (it needs filesystem access to your projects + Docker):
    uv run --extra ui streamlit run src/repo_surgeon/ui/app.py
or simply:  surgeon ui
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from repo_surgeon.config import load_config
from repo_surgeon.models import RunStatus
from repo_surgeon.registry import load_registry
from repo_surgeon.runner import run_issue
from repo_surgeon.state import RunState

load_dotenv()  # provider keys + LangSmith config from .env

st.set_page_config(page_title="repo-surgeon", page_icon="🩺", layout="centered")
st.title("🩺 repo-surgeon")
st.caption("Multi-agent coding orchestrator — describe a problem, get a verified fix on a branch.")

registry = load_registry()
project_paths: dict[str, str] = {}
if registry:
    project_paths = {
        f"{p.name}  ·  {p.kind}": p.path
        for p in registry.projects
        if p.kind != "container" and p.is_canonical
    }

issue = st.text_area(
    "Problem statement",
    placeholder="e.g. median() in stats.py returns the wrong value for even-length lists",
    height=110,
)

col1, col2 = st.columns(2)
with col1:
    picked = st.selectbox(
        "Project",
        options=["(auto-route from the problem)"] + sorted(project_paths),
    )
with col2:
    manual = st.text_input("…or a repo path / GitHub owner/repo", placeholder="~/Documents/my-proj")

no_verify = st.checkbox(
    "Skip test verification", help="For docs/config tasks or repos without a test suite."
)

if st.button("Run repo-surgeon", type="primary", disabled=not issue.strip()):
    repo = manual.strip() or project_paths.get(picked, "")
    if not repo:
        st.error("Pick a project, enter a repo path, or ensure the registry has projects.")
        st.stop()

    trace = st.container()
    seen: list[str] = []

    def on_event(state: RunState) -> None:
        notes = state.get("notes", [])
        for note in notes[len(seen) :]:
            trace.markdown(f"- {note}")
        seen[:] = notes

    with st.status("Orchestrating…", expanded=True):
        try:
            final = run_issue(
                issue,
                repo,
                config=load_config(),
                require_tests=not no_verify,
                on_event=on_event,
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
            st.error(f"Run failed: {exc}")
            st.stop()

    status = final.get("status", RunStatus.FAILED)
    if status == RunStatus.RESOLVED:
        st.success(f"Resolved — delivered on `{final.get('delivery_ref', '?')}`")
    else:
        st.warning(f"Finished with status: {status.value} (no fix delivered)")

    patches = final.get("patches", [])
    if patches:
        with st.expander(f"Proposed edits ({len(patches)})"):
            for p in patches:
                st.code(f"# {p.file_path}\n- {p.search}\n+ {p.replace}", language="diff")
