from abc import ABC, abstractmethod
from typing import Literal

Provider = Literal["daytona", "modal"]


class Sandbox(ABC):
    """
    Base class for all sandboxes.

    Provides a common interface for all sandboxes, as well as shared logic and functionality.
    """

    provider: Provider

    @abstractmethod
    async def exec(self, command: str, timeout: int) -> tuple[int, str]:
        raise NotImplementedError

    async def apply_patch(self, patch: str, timeout: int) -> None:
        raise NotImplementedError

    async def run_tests(self, tests: list[str], timeout: int) -> tuple[int, int]:
        raise NotImplementedError
