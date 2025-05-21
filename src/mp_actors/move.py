import asyncio
from dataclasses import dataclass
import inspect
import multiprocessing as mp
import nest_asyncio
import os
import setproctitle
import sys
from tblib import pickling_support
from typing import Any, AsyncGenerator, cast, TypeVar
import uuid
from concurrent.futures import ThreadPoolExecutor

from .traceback import streamline_tracebacks

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

nest_asyncio.apply()


T = TypeVar("T")

# Special ID to signal shutdown
_SHUTDOWN_ID = "__shutdown__"

def move_to_child_process(
    obj: T, log_file: str | None = None, process_name: str | None = None
) -> T:
    """
    Move an object to a child process and return a proxy to it.

    This function creates a proxy object that runs in a separate process. Method calls
    on the proxy are forwarded to a pickled copy of the original object in the child
    process.

    Args:
        obj: The object to move to a child process.
        log_file: Optional path to a file where stdout/stderr from the child process
                 will be redirected. If None, output goes to the parent process.
        process_name: Optional name for the child process.

    Returns:
        A proxy object that forwards method calls to the original object in the child process.
        The proxy has the same interface as the original object.
    """
    return cast(T, Proxy(obj, log_file, process_name))


@dataclass
class Request:
    id: str
    method_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    send_value: Any = None


@dataclass
class Response:
    id: str
    result: Any
    exception: Exception | None


class Proxy:
    def __init__(
        self, obj: object, log_file: str | None = None, process_name: str | None = None
    ) -> None:
        self._obj = obj
        self._requests = mp.Queue()
        self._responses = mp.Queue()
        self._process = mp.Process(
            target=_target,
            args=(obj, self._requests, self._responses, log_file, process_name),
        )
        self._process.start()
        # dedicated executor for queue.get calls
        self._executor = ThreadPoolExecutor()
        self._futures: dict[str, asyncio.Future] = {}
        self._handle_responses_task = asyncio.create_task(self._handle_responses())

    async def _handle_responses(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            response: Response = await loop.run_in_executor(
                self._executor, self._responses.get
            )
            # check for shutdown signal
            if response.id == _SHUTDOWN_ID:
                break
            # normal processing
            future = self._futures.pop(response.id, None)
            if future is None:
                continue
            if response.exception:
                future.set_exception(response.exception)
            else:
                future.set_result(response.result)

    @streamline_tracebacks()
    def __getattr__(self, name: str) -> Any:
        # For attributes that aren't methods, get them directly
        if not hasattr(self._obj, name):
            raise AttributeError(
                f"{type(self._obj).__name__} has no attribute '{name}'"
            )

        async def get_response(
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            id: uuid.UUID | None = None,
            send_value: Any | None = None,
        ) -> Any:
            request = Request(str(id or uuid.uuid4()), name, args, kwargs, send_value)
            self._futures[request.id] = asyncio.Future()
            self._requests.put_nowait(request)
            return await self._futures[request.id]

        # Check if it's a method or property
        attr = getattr(self._obj, name)
        if inspect.isasyncgenfunction(attr):
            # Return an async generator wrapper function
            @streamline_tracebacks()
            async def async_gen_wrapper(
                *args: Any, **kwargs: Any
            ) -> AsyncGenerator[Any, Any]:
                try:
                    id = uuid.uuid4()
                    send_value = None
                    while True:
                        send_value = yield await get_response(
                            args, kwargs, id, send_value
                        )
                        args, kwargs = (), {}
                except StopAsyncIteration:
                    return

            return async_gen_wrapper
        elif asyncio.iscoroutinefunction(attr):
            # Return an async wrapper function
            @streamline_tracebacks()
            async def async_method_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await get_response(args, kwargs)

            return async_method_wrapper
        elif callable(attr):
            # Return a regular function wrapper
            @streamline_tracebacks()
            def method_wrapper(*args: Any, **kwargs: Any) -> Any:
                return asyncio.run(get_response(args, kwargs))

            return method_wrapper
        else:
            # For non-callable attributes, get them directly
            return asyncio.run(get_response(tuple(), dict()))

    def close(self):
        # signal the response loop to exit
        self._responses.put_nowait(Response(_SHUTDOWN_ID, None, None))
        # wait for the handler to finish
        if hasattr(self, "_handle_responses_task"):
            # give it a moment to break
            try:
                asyncio.get_event_loop().run_until_complete(self._handle_responses_task)
            except Exception:
                pass

        # terminate child process and force kill if needed
        if hasattr(self, "_process"):
            self._process.terminate()
            try:
                self._process.join(timeout=1)
            except Exception:
                pass
            if self._process.is_alive():
                # Python 3.7+: force kill
                try:
                    self._process.kill()
                except AttributeError:
                    # fallback: os.kill
                    os.kill(self._process.pid, 9)
                self._process.join()

        # shutdown executor cleanly
        self._executor.shutdown(wait=True)

        # close and cancel queue feeder threads
        self._responses.close()
        self._responses.cancel_join_thread()
        self._requests.close()
        self._requests.cancel_join_thread()


def _target(
    obj: object,
    requests: mp.Queue,
    responses: mp.Queue,
    log_file: str | None = None,
    process_name: str | None = None,
) -> None:
    if process_name:
        setproctitle.setproctitle(process_name)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        sys.stdout = sys.stderr = open(log_file, "a", buffering=1)
    asyncio.run(_handle_requests(obj, requests, responses))


async def _handle_requests(
    obj: object, requests: mp.Queue, responses: mp.Queue
) -> None:
    generators: dict[str, AsyncGenerator[Any, Any]] = {}
    while True:
        request: Request = await asyncio.get_event_loop().run_in_executor(
            None, requests.get
        )
        asyncio.create_task(_handle_request(obj, request, responses, generators))


async def _handle_request(
    obj: object,
    request: Request,
    responses: mp.Queue,
    generators: dict[str, AsyncGenerator[Any, Any]],
) -> None:
    try:
        result_or_callable = getattr(obj, request.method_name)
        if inspect.isasyncgenfunction(result_or_callable):
            if not request.id in generators:
                generators[request.id] = result_or_callable(
                    *request.args, **request.kwargs
                )
            result = await generators[request.id].asend(request.send_value)
        elif callable(result_or_callable):
            result_or_coro = result_or_callable(*request.args, **request.kwargs)
            if asyncio.iscoroutine(result_or_coro):
                result = await result_or_coro
            else:
                result = result_or_coro
        else:
            result = result_or_callable
        response = Response(request.id, result, None)
    except Exception as e:
        pickling_support.install(e)
        response = Response(request.id, None, e)
    responses.put_nowait(response)
