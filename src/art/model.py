from contextlib import asynccontextmanager, _AsyncGeneratorContextManager
from dataclasses import dataclass
from openai import AsyncOpenAI
from types import TracebackType
from typing import Any, Callable, Coroutine, Generator, Iterable, TYPE_CHECKING

from .config.model import ModelConfig
from .config.openai_server import OpenAIServerConfig
from .openai import patch_openai
from .types import BaseModel, Trajectory, TuneConfig, Verbosity


if TYPE_CHECKING:
    from .api import API


@dataclass
class ClientWrapper:
    get_client: Callable[[], Coroutine[None, None, AsyncOpenAI]]
    client: AsyncOpenAI | None = None

    async def __aenter__(self) -> AsyncOpenAI:
        self.client = await self.get_client()
        return self.client

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.client is not None:
            await self.client.close()

    def __await__(self) -> Generator[Any, Any, AsyncOpenAI]:
        return self.get_client().__await__()


@dataclass
class Model:
    api: "API"
    name: str
    base_model: BaseModel
    _config: ModelConfig | None = None

    def openai_client(
        self,
        estimated_completion_tokens: int = 1024,
        tool_use: bool = False,
        verbosity: Verbosity = 1,
    ) -> ClientWrapper:
        """
        Context manager for an OpenAI client to a managed inference service.

        Args:
            estimated_completion_tokens: Estimated completion tokens per request.
            tool_use: Whether to enable tool use.
            verbosity: Verbosity level.

        Yields:
            AsyncOpenAI: An asynchronous OpenAI client.

        Example:
            async with model.openai_client() as client:
                chat_completion = await client.chat.completions.create(
                    model=model.name,
                    messages=[{"role": "user", "content": "Hello, world!"}],
                )
        """
        return self._openai_client(
            estimated_completion_tokens, tool_use, verbosity, _config=None
        )

    def _openai_client(
        self,
        estimated_completion_tokens: int = 1024,
        tool_use: bool = False,
        verbosity: Verbosity = 1,
        _config: OpenAIServerConfig | None = None,
    ) -> ClientWrapper:
        """
        Private method for the context manager for an OpenAI client to a managed inference service.

        Args:
            estimated_completion_tokens: Estimated completion tokens per request.
            tool_use: Whether to enable tool use.
            verbosity: Verbosity level.
            _config: An OpenAIServerConfig object. May be subject to breaking changes at any time.
                Use at your own risk.

        Yields:
            AsyncOpenAI: An asynchronous OpenAI client.

        Example:
            async with model.openai_client() as client:
                chat_completion = await client.chat.completions.create(
                    model=model.name,
                    messages=[{"role": "user", "content": "Hello, world!"}],
                )
        """

        async def get_client() -> AsyncOpenAI:
            client, semaphore = await self.api._get_openai_client(
                self, estimated_completion_tokens, tool_use, verbosity, _config
            )
            return patch_openai(client, semaphore, self.api._close_openai_client)

        return ClientWrapper(get_client=get_client)

    async def get_iteration(self) -> int:
        """
        Get the model's current training iteration.
        """
        return await self.api._get_iteration(self)

    async def clear_iterations(
        self,
        benchmark: str = "val/reward",
        benchmark_smoothing: float = 1.0,
        verbosity: Verbosity = 1,
    ) -> None:
        """
        Delete all but the latest and best iteration checkpoints.

        Args:
            benchmark: The benchmark to use to determine the best iteration.
            benchmark_smoothing: Smoothing factor (0-1) that controls how much to reduce
                variance when determining the best iteration. Defaults to 1.0 (no smoothing).
            verbosity: Verbosity level.
        """
        await self.api._clear_iterations(
            self, benchmark, benchmark_smoothing, verbosity
        )

    async def log(
        self,
        trajectory_groups: Iterable[Iterable[Trajectory | BaseException]],
        split: str = "val",
    ) -> None:
        """
        Log the model's performance for an evaluation batch of trajectory groups.

        Args:
            trajectory_groups: A batch of trajectory groups.
            split: The evaluation's split. Defaults to "val".
        """
        await self.api._log(self, [list(group) for group in trajectory_groups], split)

    async def tune(
        self,
        trajectory_groups: Iterable[Iterable[Trajectory | BaseException]],
        config: TuneConfig = TuneConfig(),
    ) -> None:
        """
        Reinforce fine-tune the model with a batch of trajectory groups.

        Args:
            trajectory_groups: A batch of trajectory groups.
            config: Fine-tuning specific configuration with options for the optimizer,
                loss, other hyperparameters, logging, etc.
        """
        await self.api._tune_model(
            self, [list(group) for group in trajectory_groups], config
        )
