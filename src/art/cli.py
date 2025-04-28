from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import pydantic
import socket
import typer
from typing import Any, AsyncIterator
import uvicorn

from . import dev
from .local import LocalAPI
from .model import TrainableModel
from .trajectories import TrajectoryGroup
from .types import TrainConfig

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

    api = LocalAPI()
    app = FastAPI()
    app.get("/healthcheck")(lambda: {"status": "ok"})
    app.post("/register")(api.register)
    app.post("/_log")(api._log)
    app.post("/_prepare_backend_for_training")(api._prepare_backend_for_training)
    app.post("/_get_step")(api._get_step)
    app.post("/_delete_checkpoints")(api._delete_checkpoints)

    @app.post("/_train_model")
    async def _train_model(
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
    ) -> StreamingResponse:
        async def stream() -> AsyncIterator[str]:
            async for result in api._train_model(
                model, trajectory_groups, config, dev_config
            ):
                yield json.dumps(result) + "\n"

        return StreamingResponse(stream())

    uvicorn.run(app, host=host, port=port)
