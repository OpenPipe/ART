import json
from pathlib import Path
import socket
from typing import Any, AsyncIterator

from dotenv import load_dotenv
import typer

load_dotenv()

app = typer.Typer()


SKILL_NAMES = ["train-sft", "train-rl"]

WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
WANDB_INFERENCE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


def _get_skill_path(skill_name: str) -> Path:
    """Find a skill file, checking installed package first, then repo root."""
    # Installed from wheel: art/skills/ in site-packages
    pkg_path = Path(__file__).parent / "skills" / skill_name / "SKILL.md"
    if pkg_path.exists():
        return pkg_path
    # Development: .agents/skills/ in repo root
    dev_path = (
        Path(__file__).parent.parent.parent
        / ".agents"
        / "skills"
        / skill_name
        / "SKILL.md"
    )
    if dev_path.exists():
        return dev_path
    raise FileNotFoundError(f"Skill '{skill_name}' not found")


def _chat_with_skill(skill_name: str) -> None:
    """Run an interactive chat session using a skill as the system prompt."""
    import os
    import sys

    from openai import OpenAI

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        typer.echo(
            "Error: WANDB_API_KEY environment variable is required.\n"
            "Get your key at https://wandb.ai/authorize",
            err=True,
        )
        raise typer.Exit(1)

    try:
        skill_path = _get_skill_path(skill_name)
    except FileNotFoundError:
        typer.echo(
            f"Error: Skill '{skill_name}' not found in this installation.", err=True
        )
        raise typer.Exit(1)

    skill_content = skill_path.read_text()
    # Strip YAML frontmatter
    if skill_content.startswith("---"):
        end = skill_content.find("---", 3)
        if end != -1:
            skill_content = skill_content[end + 3 :].strip()

    cli_preamble = (
        "IMPORTANT: You are in a plain text chat. You have NO tools. "
        "You cannot search files, read files, run scripts, validate data, or execute code. "
        "Never say 'Let me search...', 'Scanning...', 'Found N files...', or 'Valid! N rows' — "
        "you did not do any of these things. "
        "If a step requires running code or using tools, skip it entirely and move on.\n\n"
    )

    client = OpenAI(base_url=WANDB_INFERENCE_BASE_URL, api_key=api_key)
    messages: list = [{"role": "system", "content": cli_preamble + skill_content}]

    typer.echo(f"ART {skill_name} wizard (powered by {WANDB_INFERENCE_MODEL})")
    typer.echo("Type 'quit' to exit.\n")

    # Send an initial empty user message to kick off the wizard
    messages.append({"role": "user", "content": "Hi, let's get started."})

    while True:
        try:
            stream = client.chat.completions.create(
                model=WANDB_INFERENCE_MODEL,
                messages=messages,
                stream=True,
            )
            assistant_message = ""
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    assistant_message += delta
            typer.echo("")  # newline after streaming
            messages.append({"role": "assistant", "content": assistant_message})
        except Exception as e:
            typer.echo(f"\nError from inference API: {e}", err=True)
            raise typer.Exit(1)

        try:
            user_input = typer.prompt("\nYou")
        except (KeyboardInterrupt, EOFError):
            typer.echo("\nExiting.")
            break

        if user_input.strip().lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})


@app.command()
def train_sft() -> None:
    """Interactive wizard to create an SFT training script."""
    _chat_with_skill("train-sft")


@app.command()
def train_rl() -> None:
    """Interactive wizard to create an RL training script."""
    _chat_with_skill("train-rl")


@app.command()
def install_skills(
    path: Path = typer.Argument(
        default=Path("."), help="Project directory to install skills into"
    ),
) -> None:
    """Install ART agent skills for Claude Code and OpenAI Codex.

    Copies bundled SKILL.md files into .claude/skills/ and .agents/skills/
    in the target project directory.

    Examples:
        art install-skills
        art install-skills /path/to/my-project
    """
    import shutil

    target = path.resolve()
    destinations = [
        target / ".claude" / "skills",
        target / ".agents" / "skills",
    ]

    installed = []
    for dest_root in destinations:
        for skill_name in SKILL_NAMES:
            try:
                src = _get_skill_path(skill_name)
            except FileNotFoundError:
                continue
            dest_dir = dest_root / skill_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_dir / "SKILL.md")
            installed.append(str(dest_dir / "SKILL.md"))

    typer.echo(f"Installed {len(installed)} skill files into {target}:")
    for f in installed:
        typer.echo(f"  {f}")
    typer.echo(
        "\nUse /train-sft and /train-rl in Claude Code or OpenAI Codex to get started."
    )


@app.command()
def migrate(
    path: Path = typer.Argument(
        ..., help="Path to model dir, project dir, or trajectories dir"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be migrated without making changes",
    ),
    keep_jsonl: bool = typer.Option(
        False, "--keep-jsonl", help="Keep original JSONL files after conversion"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print progress for each file"
    ),
) -> None:
    """
    Migrate trajectory files from JSONL to Parquet format.

    This command converts old .jsonl trajectory files to the new .parquet format,
    which provides ~25x compression and ~20x faster queries.

    Examples:
        art migrate /path/to/.art/project/models/my-model
        art migrate /path/to/.art/project --dry-run
        art migrate /path/to/trajectories --keep-jsonl --verbose
    """
    from .utils.trajectory_migration import (
        migrate_model_dir,
        migrate_trajectories_dir,
    )

    if not path.exists():
        typer.echo(f"Error: Path does not exist: {path}", err=True)
        raise typer.Exit(1)

    # Determine what kind of path this is
    if (path / "trajectories").exists():
        # This is a model directory
        typer.echo(f"Migrating model directory: {path}")
        result = migrate_model_dir(
            path,
            delete_originals=not keep_jsonl,
            dry_run=dry_run,
            progress_callback=lambda f: typer.echo(f"  {f}") if verbose else None,
        )
    elif path.name == "trajectories" or any(path.glob("*/[0-9]*.jsonl")):
        # This is a trajectories directory
        typer.echo(f"Migrating trajectories directory: {path}")
        result = migrate_trajectories_dir(
            path,
            delete_originals=not keep_jsonl,
            dry_run=dry_run,
            progress_callback=lambda f: typer.echo(f"  {f}") if verbose else None,
        )
    elif (path / "models").exists():
        # This is a project directory
        typer.echo(f"Migrating project directory: {path}")
        from .utils.trajectory_migration import MigrationResult

        result = MigrationResult()
        models_dir = path / "models"
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                if verbose:
                    typer.echo(f"Processing model: {model_dir.name}")
                model_result = migrate_model_dir(
                    model_dir,
                    delete_originals=not keep_jsonl,
                    dry_run=dry_run,
                    progress_callback=lambda f: typer.echo(f"    {f}")
                    if verbose
                    else None,
                )
                result = result + model_result
    else:
        typer.echo(
            f"Error: Could not determine path type. Expected a model, project, or trajectories directory.",
            err=True,
        )
        raise typer.Exit(1)

    # Print summary
    if dry_run:
        typer.echo(f"\n[DRY RUN] Would migrate {result.files_migrated} files")
        if result.bytes_before > 0:
            typer.echo(
                f"  Estimated space savings: {result.space_saved / 1024 / 1024:.1f} MB"
            )
    else:
        typer.echo(f"\nMigrated {result.files_migrated} files")
        if result.files_skipped > 0:
            typer.echo(f"Skipped {result.files_skipped} files")
        if result.bytes_before > 0 and result.bytes_after > 0:
            typer.echo(
                f"Space saved: {result.space_saved / 1024 / 1024:.1f} MB ({result.compression_ratio:.1f}x compression)"
            )

    if result.errors:
        typer.echo(f"\nErrors ({len(result.errors)}):", err=True)
        for error in result.errors[:10]:
            typer.echo(f"  {error}", err=True)
        if len(result.errors) > 10:
            typer.echo(f"  ... and {len(result.errors) - 10} more errors", err=True)
        raise typer.Exit(1)


@app.command()
def run(host: str = "0.0.0.0", port: int = 7999) -> None:
    """Run the ART CLI."""

    from fastapi import Body, FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import pydantic
    import uvicorn

    from . import dev
    from .errors import ARTError
    from .local import LocalBackend
    from .model import Model, TrainableModel
    from .trajectories import TrajectoryGroup
    from .types import TrainConfig

    # check if port is available
    def is_port_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) != 0

    if not is_port_available(port):
        print(
            f"Port {port} is already in use, possibly because the ART server is already running."
        )
        return

    # Reset the custom __new__ and __init__ methods for TrajectoryGroup
    def __new__(cls, *args: Any, **kwargs: Any) -> TrajectoryGroup:
        return pydantic.BaseModel.__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return pydantic.BaseModel.__init__(self, *args, **kwargs)

    TrajectoryGroup.__new__ = __new__  # type: ignore
    TrajectoryGroup.__init__ = __init__  # ty:ignore[invalid-assignment]

    backend = LocalBackend()
    app = FastAPI()

    # Add exception handler for ARTError
    @app.exception_handler(ARTError)
    async def art_error_handler(request: Request, exc: ARTError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    app.get("/healthcheck")(lambda: {"status": "ok"})
    app.post("/close")(backend.close)
    app.post("/register")(backend.register)
    app.post("/_get_step")(backend._get_step)

    @app.post("/_delete_checkpoint_files")
    async def _delete_checkpoint_files(
        model: TrainableModel = Body(...),
        steps_to_keep: list[int] = Body(...),
    ):
        await backend._delete_checkpoint_files(model, steps_to_keep)

    @app.post("/_prepare_backend_for_training")
    async def _prepare_backend_for_training(
        model: TrainableModel,
        config: dev.OpenAIServerConfig | None = Body(None),
    ):
        return await backend._prepare_backend_for_training(model, config)

    # Note: /_log endpoint removed - logging now handled by frontend (Model.log())

    @app.post("/_train_model")
    async def _train_model(
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
        verbose: bool = Body(False),
    ) -> StreamingResponse:
        async def stream() -> AsyncIterator[str]:
            async for result in backend._train_model(
                model, trajectory_groups, config, dev_config, verbose
            ):
                yield json.dumps(result) + "\n"

        return StreamingResponse(stream())

    # Wrap in function with Body(...) to ensure FastAPI correctly interprets
    # all parameters as body parameters
    @app.post("/_experimental_pull_from_s3")
    async def _experimental_pull_from_s3(
        model: Model = Body(...),
        s3_bucket: str | None = Body(None),
        prefix: str | None = Body(None),
        verbose: bool = Body(False),
        delete: bool = Body(False),
    ):
        await backend._experimental_pull_from_s3(
            model=model,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
            delete=delete,
        )

    @app.post("/_experimental_push_to_s3")
    async def _experimental_push_to_s3(
        model: Model = Body(...),
        s3_bucket: str | None = Body(None),
        prefix: str | None = Body(None),
        verbose: bool = Body(False),
        delete: bool = Body(False),
    ):
        await backend._experimental_push_to_s3(
            model=model,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
            delete=delete,
        )

    uvicorn.run(app, host=host, port=port, loop="asyncio")
