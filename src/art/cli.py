from fastapi import FastAPI
import typer
import uvicorn

from .local import LocalAPI

app = typer.Typer()


@app.command()
def run(host: str = "0.0.0.0", port: int = 2218) -> None:
    """Run the ART CLI."""
    api = LocalAPI()
    app = FastAPI()
    app.post("/register")(api.register)
    app.post("/_log")(api._log)
    app.post("/_prepare_backend_for_training")(api._prepare_backend_for_training)
    app.post("/_get_step")(api._get_step)
    app.post("/_delete_checkpoints")(api._delete_checkpoints)
    app.post("/_train_model")(api._train_model)
    uvicorn.run(app, host=host, port=port)
