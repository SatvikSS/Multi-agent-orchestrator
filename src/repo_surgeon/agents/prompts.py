"""System prompts and human-message builders for the agents."""

from __future__ import annotations

from repo_surgeon.models import Issue

LOCALIZER_SYSTEM = (
    "You are a code localizer. Given an issue and candidate source files, decide which "
    "file(s) contain the code that must change. Respond with file paths only, one per line, "
    "most likely first. Choose only from the provided paths. No explanation."
)

PLANNER_SYSTEM = (
    "You are a planning agent. Given an issue and the relevant code, write a short, concrete "
    "plan for the fix as 2-5 bullet points naming the exact function(s) and change required. "
    "Do not write code."
)

WRITER_SYSTEM = (
    "You are a precise code-editing agent. Produce edits ONLY as SEARCH/REPLACE blocks in "
    "exactly this format:\n\n"
    "path/to/file.py\n"
    "<<<<<<< SEARCH\n"
    "the exact existing lines to find\n"
    "=======\n"
    "the replacement lines\n"
    ">>>>>>> REPLACE\n\n"
    "Rules: the SEARCH text must match the current file exactly (whitespace included); keep "
    "each block minimal; use the file path shown in the context; output nothing but blocks.\n"
    "To create a new file OR replace a file's entire contents, leave the SEARCH section "
    "empty (put the full new file body in the REPLACE section)."
)


def localizer_prompt(issue: Issue, candidate_paths: list[str], context: str) -> str:
    return (
        f"Issue: {issue.title}\n\n{issue.body}\n\n"
        f"Candidate files:\n" + "\n".join(candidate_paths) + "\n\n"
        f"File contents:\n{context}"
    )


def planner_prompt(issue: Issue, context: str) -> str:
    return f"Issue: {issue.title}\n\n{issue.body}\n\nRelevant code:\n{context}"


def writer_prompt(issue: Issue, plan: str, context: str, feedback: str = "") -> str:
    feedback_block = f"\n\nPrevious attempt failed:\n{feedback}\n" if feedback else ""
    return (
        f"Issue: {issue.title}\n\n{issue.body}\n\n"
        f"Fix plan:\n{plan}\n\n"
        f"Current code (edit against this exact text):\n{context}"
        f"{feedback_block}"
    )
