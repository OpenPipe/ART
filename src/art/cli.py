from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from fastapi import Request
from fastapi.responses import JSONResponse
import json
import pydantic
import socket
import typer
from typing import Any, AsyncIterator
import uvicorn

from . import dev
from .local import LocalBackend
from .model import Model, TrainableModel, list_trash, restore_from_trash, empty_trash, delete_model
from .trajectories import TrajectoryGroup, AsyncTrajectoryGroup
from .types import TrainConfig
from .utils.deploy_model import LoRADeploymentProvider
from .errors import ARTError

app = typer.Typer()


@app.command()
def run(host: str = "0.0.0.0", port: int = 7999) -> None:
    """Run the ART CLI."""

    # check if port is available
    def is_port_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) != 0

    if not is_port_available(port):
        print(
            f"Port {port} is already in use, possibly because the ART server is already running."
        )
        return

    

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
    app.post("/_delete_checkpoints")(backend._delete_checkpoints)

    @app.post("/_prepare_backend_for_training")
    async def _prepare_backend_for_training(
        model: TrainableModel,
        config: dev.OpenAIServerConfig | None = Body(None),
    ):
        return await backend._prepare_backend_for_training(model, config)

    @app.post("/_log")
    async def _log(
        model: Model,
        trajectory_groups: list[TrajectoryGroup],
        split: str = Body("val"),
    ):
        await backend._log(model, trajectory_groups, split)

    @app.post("/_train_model")
    async def _train_model(
        model: TrainableModel,
        trajectory_groups: list[AsyncTrajectoryGroup],
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

    @app.post("/_experimental_deploy")
    async def _experimental_deploy(
        deploy_to: LoRADeploymentProvider = Body(...),
        model: TrainableModel = Body(...),
        step: int | None = Body(None),
        s3_bucket: str | None = Body(None),
        prefix: str | None = Body(None),
        verbose: bool = Body(False),
        pull_s3: bool = Body(True),
        wait_for_completion: bool = Body(True),
    ):
        return await backend._experimental_deploy(
            deploy_to=deploy_to,
            model=model,
            step=step,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
            pull_s3=pull_s3,
            wait_for_completion=wait_for_completion,
        )

    uvicorn.run(app, host=host, port=port, loop="asyncio")

@app.command()
def delete(project_name: str, model_name: str):
    """Move a model to the trash."""
    try:
        delete_model(project_name, model_name)
        print(f"Model '{model_name}' in project '{project_name}' moved to trash.")
    except FileNotFoundError as e:
        print(e)

@app.command()
def list_trashed_models(project_name: str):
    """List models in the trash."""
    trashed_models = list_trash(project_name)
    if not trashed_models:
        print("Trash is empty.")
    else:
        print("Models in trash:")
        for model_name in trashed_models:
            print(f"- {model_name}")

@app.command()
def restore_model(project_name: str, model_name: str):
    """Restore a model from the trash."""
    try:
        restore_from_trash(project_name, model_name)
        print(f"Model '{model_name}' in project '{project_name}' restored.")
    except FileNotFoundError as e:
        print(e)

@app.command()
def empty_trashed_models(project_name: str):
    """Permanently delete all models in the trash."""
    empty_trash(project_name)
    print("Trash has been emptied.")