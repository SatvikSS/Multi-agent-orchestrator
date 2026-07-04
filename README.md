# repo-surgeon

A multi-agent coding orchestrator built on **LangGraph**. It takes a problem statement
(a GitHub issue or a local description) plus a target codebase (a **local folder** or a
**GitHub repo**) and runs a self-correcting agent pipeline to produce a fix:

```
locate → plan → write (search-replace edits) → test (Docker sandbox) → critic → deliver
```

The critic loops back to the writer/planner until the repo's own test suite passes or a
budget (attempts / cost) is exhausted.

## Status

🚧 Under active development. Working today:

- **Full agent pipeline**: knowledge-base routing → localizer → planner → writer
  (search-replace edits) → **Docker-sandboxed pytest** → critic self-correction loop
- **Any project shape**: git repos, subfolders, containers with nested repos
  (duplicate-clone detection), and plain folders (consented shadow-git or no-touch
  staging + `.patch` delivery)
- **GitHub end-to-end**: issue URL in → clone → fix → push → **pull request out**
- **Safety rails**: dirty-tree guard, work-branch isolation, crash cleanup, untracked
  files never committed
- **Fleet awareness**: `surgeon discover` registry + issue→project auto-routing
  (`surgeon run -i "..."` with no `--repo`)
- **LangSmith tracing** via `LANGCHAIN_TRACING_V2=true`
- **Semantic code search** (AST-chunked, embeddings-backed) as the writer's "related
  code" tool — optional, `semantic.enabled`
- **Evaluation harness**: `surgeon eval` runs a benchmark of seeded-bug cases and reports
  resolved-rate + metrics (real pytest in Docker)

Remaining: Streamlit demo, CI, README diagram.

## Design highlights

- **Provider-agnostic**: switch between OpenRouter, Ollama (local), Anthropic, and OpenAI
  per agent role via `config.yaml`.
- **Source-agnostic**: `Workspace` and `IssueSource` abstractions make "local folder today,
  GitHub repo tomorrow" a non-event.
- **Right-sized retrieval**: knowledge-base summaries route to a repo/folder, then
  ripgrep + read pinpoint the code; embeddings are a similarity *tool*, not the spine.

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env   # fill in the keys you use
uv run surgeon run --issue "fix the off-by-one in paginate()" --repo ./path/to/project
```

## Development

```bash
uv run pytest        # tests (LLM calls mocked)
uv run ruff check .  # lint
uv run mypy src      # types
```
