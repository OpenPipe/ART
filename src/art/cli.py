import typer

app = typer.Typer()


@app.command()
def run() -> None:
    """Run the ART CLI."""
    print("Hello, world!")