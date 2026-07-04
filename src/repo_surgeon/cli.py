"""Typer CLI: `surgeon run --issue <ref> --repo <path>`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from repo_surgeon import __version__
from repo_surgeon.config import AppConfig, load_config
from repo_surgeon.evaluation import EvalSummary, builtin_cases_dir, load_cases, run_eval
from repo_surgeon.home import subdir
from repo_surgeon.issues import open_issue_source
from repo_surgeon.issues.github import parse_issue_ref
from repo_surgeon.llm.factory import build_llm
from repo_surgeon.models import RunStatus
from repo_surgeon.registry import discover as discover_projects
from repo_surgeon.registry import load_registry
from repo_surgeon.retrieval import build_knowledge_base
from repo_surgeon.routing import pick_with_llm, route_issue
from repo_surgeon.runner import run_issue
from repo_surgeon.workspace import open_workspace
from repo_surgeon.workspace.base import DirtyWorkspaceError, Workspace
from repo_surgeon.workspace.resolver import (
    AmbiguousWorkspaceError,
    NoRepoFoundError,
    resolve_repo,
)
from repo_surgeon.workspace.shadow import init_shadow_git
from repo_surgeon.workspace.staging import StagingWorkspace, apply_patch_file

# Export .env into the process environment so LangSmith tracing (LANGCHAIN_*) and
# provider SDKs that read os.environ see the values, not just pydantic-settings.
load_dotenv()

app = typer.Typer(
    name="surgeon",
    help="Multi-agent coding orchestrator: resolve an issue into a fix.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

_STATUS_STYLE = {
    RunStatus.RESOLVED: "bold green",
    RunStatus.FAILED: "bold red",
    RunStatus.BUDGET_EXHAUSTED: "bold yellow",
}


def _pick_candidate(exc: AmbiguousWorkspaceError, repo: str) -> str:
    console.print(f"[yellow]'{repo}' contains {len(exc.candidates)} git repositories:[/yellow]")
    for i, cand in enumerate(exc.candidates, start=1):
        console.print(f"  [cyan]{i}[/cyan]. {cand.describe()}")
    choice = int(typer.prompt("Which repo should I work in? (number)", type=int))
    if not 1 <= choice <= len(exc.candidates):
        console.print("[bold red]Invalid choice.[/bold red]")
        raise typer.Exit(code=1) from exc
    return exc.candidates[choice - 1].path


def _handle_plain_folder(
    repo: str, *, init_git: bool, staging: bool
) -> tuple[str, Workspace | None]:
    """Decide how to work in a folder with no git: shadow-git, staging, or ask."""
    if init_git and staging:
        console.print("[bold red]--init-git and --staging are mutually exclusive.[/bold red]")
        raise typer.Exit(code=1)
    if not init_git and not staging:
        console.print(f"[yellow]'{repo}' is not a git repository.[/yellow]")
        console.print("  [cyan]1[/cyan]. Initialize git here (recommended — data files excluded)")
        console.print("  [cyan]2[/cyan]. Staging mode (work on an isolated copy, get a .patch)")
        console.print("  [cyan]3[/cyan]. Abort")
        choice = int(typer.prompt("How should I proceed? (number)", type=int))
        init_git, staging = choice == 1, choice == 2
        if not init_git and not staging:
            raise typer.Exit(code=0)
    if init_git:
        root = init_shadow_git(repo)
        console.print(f"[green]Initialized shadow-git in {root} (baseline commit created).[/green]")
        return str(root), None
    workspace = StagingWorkspace(repo)
    console.print(f"[dim]Staging copy created at {workspace.root_path}; original untouched.[/dim]")
    return repo, workspace


def _route_issue_to_project(issue_ref: str, cfg: AppConfig, *, assume_yes: bool) -> str:
    """No --repo given: rank registry projects for this issue and confirm with the user."""
    registry = load_registry()
    if registry is None:
        console.print(
            "[bold red]No --repo given and no registry found.[/bold red] Run "
            "[cyan]surgeon discover --root ~/Documents[/cyan] once, or pass --repo."
        )
        raise typer.Exit(code=1)

    issue = open_issue_source(issue_ref, github_token=cfg.secrets.github_token).fetch()
    candidates = route_issue(issue, registry)
    if not candidates:
        console.print("[bold red]The registry has no routable projects.[/bold red]")
        raise typer.Exit(code=1)

    try:
        best = pick_with_llm(issue, candidates, build_llm(cfg, "localizer"))
    except Exception:  # noqa: BLE001 - no key / no model: scoring order is fine
        best = candidates[0]

    default_index = candidates.index(best) + 1
    console.print(f"[bold]Routing issue:[/bold] {issue.title}")
    for i, cand in enumerate(candidates, start=1):
        marker = "→" if i == default_index else " "
        summary = cand.entry.summary or "no summary — run discover with --summarize"
        console.print(
            f" {marker} [cyan]{i}[/cyan]. {cand.entry.name} "
            f"[dim](score {cand.score}; {summary})[/dim]"
        )
    if assume_yes:
        choice = default_index
    else:
        choice = int(typer.prompt("Which project? (number)", default=default_index))
    if not 1 <= choice <= len(candidates):
        console.print("[bold red]Invalid choice.[/bold red]")
        raise typer.Exit(code=1)
    chosen = candidates[choice - 1].entry
    console.print(f"[green]Working in:[/green] {chosen.path}")
    return chosen.path


def _resolve_repo_interactive(
    repo: str, *, init_git: bool = False, staging: bool = False
) -> tuple[str, Workspace | None]:
    """Resolve a repo ref to (path, optional pre-built workspace), interacting as needed."""
    if not Path(repo).expanduser().is_dir():
        return repo, None  # GitHub refs pass through untouched
    try:
        resolution = resolve_repo(repo)
    except AmbiguousWorkspaceError as exc:
        return _pick_candidate(exc, repo), None
    except NoRepoFoundError:
        return _handle_plain_folder(repo, init_git=init_git, staging=staging)
    if resolution.note:
        console.print(f"[dim]{resolution.note}[/dim]")
    return resolution.root, None


@app.command()
def run(
    issue: Annotated[
        str,
        typer.Option("--issue", "-i", help="GitHub issue ref, a description, or a file path."),
    ],
    repo: Annotated[
        str | None,
        typer.Option(
            "--repo",
            "-r",
            help="Local project folder or a GitHub 'owner/repo'. "
            "Omit to auto-route via the registry.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config.yaml (defaults to ./config.yaml)."),
    ] = None,
    allow_dirty: Annotated[
        bool,
        typer.Option(
            "--allow-dirty",
            help="Run even if the repo has uncommitted changes (they may be lost on retries).",
        ),
    ] = False,
    no_verify: Annotated[
        bool,
        typer.Option(
            "--no-verify",
            help="Deliver a cleanly-applied patch without running tests (doc/config tasks, "
            "or repos without a test suite). Skips the Docker sandbox.",
        ),
    ] = False,
    assume_yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Accept the top routing suggestion without prompting."),
    ] = False,
    init_git: Annotated[
        bool,
        typer.Option(
            "--init-git",
            help="If the folder is not a git repo, initialize shadow-git (data files excluded).",
        ),
    ] = False,
    staging: Annotated[
        bool,
        typer.Option(
            "--staging",
            help="If the folder is not a git repo, work on an isolated copy and emit a .patch.",
        ),
    ] = False,
) -> None:
    """Run the orchestrator on one issue against one repo."""
    cfg = load_config(config_path)
    console.print(Panel.fit(f"[bold]repo-surgeon[/bold] v{__version__}", border_style="cyan"))

    if repo is None:
        gh_issue = parse_issue_ref(issue)
        if gh_issue is not None:
            repo = gh_issue[0]  # a GitHub issue URL implies its own repo
            console.print(f"[dim]GitHub issue implies repo '{repo}'.[/dim]")
        else:
            repo = _route_issue_to_project(issue, cfg, assume_yes=assume_yes)
    repo, workspace = _resolve_repo_interactive(repo, init_git=init_git, staging=staging)
    try:
        final = run_issue(
            issue,
            repo,
            config=cfg,
            allow_dirty=allow_dirty,
            require_tests=not no_verify,
            workspace=workspace,
        )
    except DirtyWorkspaceError as exc:
        console.print(f"[bold red]Refusing to run:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("\n[bold]Run trace[/bold]")
    for note in final.get("notes", []):
        console.print(f"  • {note}")

    status = final.get("status", RunStatus.FAILED)
    style = _STATUS_STYLE.get(status, "bold red")
    console.print(f"\nStatus: [{style}]{status.value}[/{style}]")
    if final.get("delivery_ref"):
        console.print(f"Delivered: [cyan]{final['delivery_ref']}[/cyan]")
        if final["delivery_ref"].endswith(".patch"):
            console.print(
                f"Apply it with: [cyan]surgeon apply --patch '{final['delivery_ref']}' "
                f"--repo '{repo}'[/cyan]"
            )


@app.command()
def apply(
    patch: Annotated[
        Path,
        typer.Option("--patch", "-p", help="Path to a .patch produced by a staging-mode run."),
    ],
    repo: Annotated[
        Path,
        typer.Option("--repo", "-r", help="The original project folder to apply the patch to."),
    ],
) -> None:
    """Apply a staging-mode patch to the original folder (with per-file backups)."""
    try:
        touched = apply_patch_file(patch, repo)
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as exc:
        console.print(f"[bold red]Apply failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Patch applied to {repo}.[/green] Files changed:")
    for rel in touched:
        console.print(f"  • {rel}")


@app.command()
def index(
    repo: Annotated[
        str,
        typer.Option("--repo", "-r", help="Local project folder to index."),
    ],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config.yaml (defaults to ./config.yaml)."),
    ] = None,
) -> None:
    """Build (or rebuild) the knowledge-base summaries for a repo."""
    cfg = load_config(config_path)
    repo, workspace = _resolve_repo_interactive(repo)
    workspace = workspace or open_workspace(repo, github_token=cfg.secrets.github_token)
    kb = build_knowledge_base(workspace, build_llm, cfg, force=True)
    console.print(
        f"Indexed [cyan]{len(kb.summaries)}[/cyan] of {kb.total_files} Python files "
        f"in [cyan]{repo}[/cyan]."
    )


@app.command()
def discover(
    root: Annotated[
        Path,
        typer.Option("--root", "-r", help="Folder containing your projects (e.g. ~/Documents)."),
    ],
    summarize: Annotated[
        bool,
        typer.Option(
            "--summarize/--no-summarize",
            help="Generate a short LLM summary per project (needs a provider key).",
        ),
    ] = True,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config.yaml (defaults to ./config.yaml)."),
    ] = None,
) -> None:
    """Scan a projects folder and build the registry (~/.repo-surgeon/registry.json)."""
    if summarize:
        cfg = load_config(config_path)
        registry = discover_projects(root, llm_provider=build_llm, app_config=cfg)
    else:
        registry = discover_projects(root)
    dup_count = sum(1 for p in registry.projects if p.duplicate_group and not p.is_canonical)
    console.print(
        f"Discovered [cyan]{len(registry.projects)}[/cyan] projects under {registry.root} "
        f"({dup_count} stale duplicate clones flagged). Run [cyan]surgeon projects[/cyan] to list."
    )


@app.command()
def projects() -> None:
    """List all projects known to the registry."""
    registry = load_registry()
    if registry is None:
        console.print(
            "[yellow]No registry yet.[/yellow] Run "
            "[cyan]surgeon discover --root ~/Documents[/cyan] first."
        )
        raise typer.Exit(code=1)

    table = Table(
        title=f"Projects under {registry.root} (scanned {registry.generated_at})",
        expand=True,
    )
    table.add_column("Project", style="cyan", overflow="fold", ratio=2)
    table.add_column("Kind", width=11)
    table.add_column(".py", justify="right", width=5)
    table.add_column("Dup", justify="center", width=3)
    table.add_column("Summary", overflow="fold", ratio=3)
    for p in registry.projects:
        if not p.duplicate_group:
            dup = ""
        else:
            dup = "[green]✓[/green]" if p.is_canonical else "[red]≈[/red]"
        table.add_row(p.name, p.kind, str(p.py_files), dup, p.summary or "[dim]—[/dim]")
    console.print(table)
    if any(p.duplicate_group for p in registry.projects):
        console.print(
            "[dim]Dup column: ✓ = freshest clone of its origin, ≈ = stale duplicate.[/dim]"
        )


@app.command(name="eval")
def evaluate(
    cases_dir: Annotated[
        Path | None,
        typer.Option("--cases", help="Directory of eval cases (default: built-in benchmark)."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="Run only the first N cases."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config.yaml (defaults to ./config.yaml)."),
    ] = None,
) -> None:
    """Run the benchmark and report the resolved rate (needs a provider key + Docker)."""
    cfg = load_config(config_path)
    cases = load_cases(cases_dir or builtin_cases_dir())
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        console.print("[yellow]No eval cases found.[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"Running [cyan]{len(cases)}[/cyan] eval case(s)…")
    summary = run_eval(cases, config=cfg)
    _print_eval_summary(summary)

    out = subdir("eval-results") / "latest.json"
    out.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[dim]Full results written to {out}[/dim]")


def _print_eval_summary(summary: EvalSummary) -> None:
    table = Table(title="Evaluation results")
    table.add_column("Case", style="cyan")
    table.add_column("Resolved", justify="center")
    table.add_column("Attempts", justify="right")
    table.add_column("Seconds", justify="right")
    for r in summary.results:
        mark = "[green]✓[/green]" if r.resolved else "[red]✗[/red]"
        table.add_row(r.name, mark, str(r.attempts), f"{r.wall_seconds:.1f}")
    console.print(table)
    pct = f"{summary.resolved_rate * 100:.0f}%"
    console.print(
        f"\n[bold]Resolved {summary.resolved}/{summary.total} ({pct})[/bold] · "
        f"produced a patch {summary.produced_patch_rate * 100:.0f}% · "
        f"avg {summary.avg_attempts:.1f} attempts · avg {summary.avg_wall_seconds:.1f}s/case"
    )


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
