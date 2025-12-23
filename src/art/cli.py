import json
import socket
from typing import Any, AsyncIterator

import pydantic
import typer
import uvicorn
from dotenv import load_dotenv
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import dev
from .errors import ARTError
from .local import LocalBackend
from .model import Model, TrainableModel, list_trash, restore_from_trash, empty_trash, delete_model
from .trajectories import TrajectoryGroup
from .types import TrainConfig
from .utils.deployment import (
    Provider,
    TogetherDeploymentConfig,
    WandbDeploymentConfig,
)

load_dotenv()

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

    
    # Reset the custom __new__ and __init__ methods for TrajectoryGroup
    def __new__(cls, *args: Any, **kwargs: Any) -> TrajectoryGroup:
        return pydantic.BaseModel.__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return pydantic.BaseModel.__init__(self, *args, **kwargs)

    TrajectoryGroup.__new__ = __new__  # type: ignore
    TrajectoryGroup.__init__ = __init__


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

@app.command()
def delete_model(project_name: str, model_name: str):
    """Move a model to the trash."""
    try:
        delete_model(project_name, model_name)
        print(f"Model '{model_name}' in project '{project_name}' moved to trash.")
    except FileNotFoundError as e:
        print(e)

@app.command()
def list_trash(project_name: str):
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
def empty_trash(project_name: str):
    """Permanently delete all models in the trash."""
    empty_trash(project_name)
    print("Trash has been emptied.")
