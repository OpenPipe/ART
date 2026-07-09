from pathlib import Path


def file_init_method(tmp_path: Path, name: str) -> str:
    path = tmp_path / f"{name}.dist"
    path.unlink(missing_ok=True)
    return f"file://{path}"
