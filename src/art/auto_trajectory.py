import contextvars
import json
from typing import Any, AsyncIterator, Coroutine, Iterator, Literal, overload

import httpx._models
from openai import OpenAI
from openai._streaming import Stream
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from .openai import consume_sync_chat_completion_stream
from .trajectories import History, Trajectory


@overload
def auto_trajectory(*, required: Literal[True]) -> Trajectory: ...


@overload
def auto_trajectory(*, required: Literal[False] = False) -> Trajectory | None: ...


def auto_trajectory(*, required: bool = False) -> Trajectory | None:
    context = auto_trajectory_context_var.get(None)
    if context is None:
        if required:
            raise RuntimeError(
                "No auto trajectory in context. `auto_trajectory(required=True)` must be called in a `capture_auto_trajectory(...)` scope."
            )
        return None
    return context.trajectory


async def capture_auto_trajectory(coroutine: Coroutine[Any, Any, Any]) -> Trajectory:
    with AutoTrajectoryContext():
        await coroutine
        trajectory = auto_trajectory_context_var.get().trajectory
        trajectory.finish()
        return trajectory


class AutoTrajectoryContext:
    def __init__(self) -> None:
        self.trajectory = Trajectory(
            messages_and_choices=[],
            reward=0.0,
        )
        self.openai_client = OpenAI(api_key="")

    def __enter__(self) -> None:
        self.token = auto_trajectory_context_var.set(self)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        auto_trajectory_context_var.reset(self.token)

    def handle_httpx_response(self, response: httpx._models.Response) -> None:
        try:
            request_content = json.loads(getattr(response.request, "_content", b""))
            messages = request_content["messages"]
            tools = request_content.get("tools", None)
            setattr(response, "_content", getattr(response, "_content_so_far", b""))
            print(getattr(response, "_content"))
            if request_content.get("stream", False):
                choice = consume_sync_chat_completion_stream(
                    Stream(
                        cast_to=ChatCompletionChunk,
                        response=response,
                        client=self.openai_client,
                    )
                ).choices[0]
            else:
                choice = Choice(
                    **json.loads(getattr(response, "_content"))["choices"][0]
                )
            history: Trajectory | History = self.trajectory
            history_index = -1
            while True:
                history_messages = history.messages()
                if history_messages == messages[: len(history_messages)] and (
                    history.tools == tools
                    or (history_messages == [] and history.tools is None)
                ):
                    break
                history_index += 1
                try:
                    history = self.trajectory.additional_histories[history_index]
                except IndexError:
                    history = History(messages_and_choices=[])
                    self.trajectory.additional_histories.append(history)
                    break
            history.messages_and_choices.extend(
                messages[len(history.messages_and_choices) :]
            )
            history.messages_and_choices.append(choice)
            history.tools = tools
        except:
            pass


auto_trajectory_context_var: contextvars.ContextVar[AutoTrajectoryContext] = (
    contextvars.ContextVar("auto_trajectory_context")
)


def patch_httpx() -> None:
    original_iter_bytes = httpx._models.Response.iter_bytes
    original_aiter_bytes = httpx._models.Response.aiter_bytes
    original_close = httpx._models.Response.close
    original_aclose = httpx._models.Response.aclose

    def patched_iter_bytes(
        self: httpx._models.Response, chunk_size: int | None = None
    ) -> Iterator[bytes]:
        for chunk in original_iter_bytes(self, chunk_size):
            setattr(
                self, "_content_so_far", getattr(self, "_content_so_far", b"") + chunk
            )
            yield chunk

    async def patched_aiter_bytes(
        self: httpx._models.Response, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        async for chunk in original_aiter_bytes(self, chunk_size):
            setattr(
                self, "_content_so_far", getattr(self, "_content_so_far", b"") + chunk
            )
            yield chunk

    def patched_close(self: httpx._models.Response) -> None:
        original_close(self)
        if context := auto_trajectory_context_var.get(None):
            context.handle_httpx_response(self)

    async def patched_aclose(self: httpx._models.Response) -> None:
        await original_aclose(self)
        if context := auto_trajectory_context_var.get(None):
            context.handle_httpx_response(self)

    httpx._models.Response.iter_bytes = patched_iter_bytes
    httpx._models.Response.aiter_bytes = patched_aiter_bytes
    httpx._models.Response.close = patched_close
    httpx._models.Response.aclose = patched_aclose


patch_httpx()
