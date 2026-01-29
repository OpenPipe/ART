import contextvars
import json
import logging
from typing import Any, AsyncIterator, Coroutine, Iterator, Literal, overload

import httpx._models
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from .openai import init_chat_completion, update_chat_completion
from .trajectories import History, Trajectory

logger = logging.getLogger(__name__)


def parse_sse_to_chat_completion(content: bytes) -> ChatCompletion:
    """Parse SSE (Server-Sent Events) content and build a ChatCompletion.

    This handles the case where streaming responses have already been consumed
    and we need to reconstruct the ChatCompletion from buffered bytes.
    """
    chat_completion: ChatCompletion | None = None

    # Parse SSE format: each line starting with "data: " contains JSON
    for line in content.decode("utf-8").split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]  # Remove "data: " prefix
        if data == "[DONE]":
            continue

        chunk_data = json.loads(data)
        chunk = ChatCompletionChunk(**chunk_data)
        if chat_completion is None:
            chat_completion = init_chat_completion(chunk)
        update_chat_completion(chat_completion, chunk)

    if chat_completion is None:
        raise ValueError("No valid chat completion chunks found in SSE content")

    return chat_completion


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
    with AutoTrajectoryContext() as trajectory:
        await coroutine
        return trajectory


class AutoTrajectoryContext:
    def __init__(self) -> None:
        self.trajectory = Trajectory(
            messages_and_choices=[],
            reward=0.0,
        )

    def __enter__(self) -> Trajectory:
        self.token = auto_trajectory_context_var.set(self)
        return self.trajectory

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        auto_trajectory_context_var.reset(self.token)
        self.trajectory.finish()

    def handle_httpx_response(self, response: httpx._models.Response) -> None:
        # Get buffered content (set by patched aiter_bytes/iter_bytes)
        content = getattr(response, "_content_so_far", b"")
        if not content:
            # No content captured, nothing to process
            return

        try:
            request_content = json.loads(getattr(response.request, "_content", b""))
        except (json.JSONDecodeError, AttributeError):
            # Not a JSON request body, skip
            return

        messages = request_content.get("messages")
        if messages is None:
            # Not a chat completion request
            return

        tools = request_content.get("tools", None)

        try:
            if request_content.get("stream", False):
                # Parse SSE content directly from buffered bytes
                chat_completion = parse_sse_to_chat_completion(content)
                choice = chat_completion.choices[0]
            else:
                choice = Choice(**json.loads(content)["choices"][0])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Failed to parse response content: {e}")
            return

        # Find the appropriate history to add this response to
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

    httpx._models.Response.iter_bytes = patched_iter_bytes  # ty:ignore[invalid-assignment]
    httpx._models.Response.aiter_bytes = patched_aiter_bytes  # ty:ignore[invalid-assignment]
    httpx._models.Response.close = patched_close  # ty:ignore[invalid-assignment]
    httpx._models.Response.aclose = patched_aclose  # ty:ignore[invalid-assignment]


patch_httpx()
