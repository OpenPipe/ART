import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires CUDA devices")
