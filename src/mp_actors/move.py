import asyncio
import atexit
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
import inspect
import itertools as it
import multiprocessing as mp
import os
import queue
import sys
import threading
from typing import Any, AsyncGenerator, TypeVar, cast
import weakref

import nest_asyncio
import setproctitle
from tblib import pickling_support

from .traceback import streamline_tracebacks

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

nest_asyncio.apply()


T = TypeVar("T")

# Special ID to signal shutdown
_SHUTDOWN_ID = -1
_LIVE_PROXIES: weakref.WeakSet[Any] = weakref.WeakSet()
_ATEXIT_REGISTERED = False


def _close_all_live_proxies() -> None:
    # Best-effort cleanup for callers that forget close_proxy().
    for proxy in list(_LIVE_PROXIES):
        try:
            close = getattr(proxy, "close", None)
            if callable(close):
                close()
        except BaseException:
            pass


def _register_proxy_for_atexit(proxy: "Proxy") -> None:
    global _ATEXIT_REGISTERED
    _LIVE_PROXIES.add(proxy)
    if not _ATEXIT_REGISTERED:
        atexit.register(_close_all_live_proxies)
        _ATEXIT_REGISTERED = True


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


def close_proxy(proxy: object) -> None:
    """
    After moving an object to a child process, you can use this function to close
    the proxy object and terminate the child process.

    Args:
        proxy: The proxy object to close.
    """
    getattr(proxy, "close", lambda: None)()


@dataclass
class Request:
    id: int
    method_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    send_value: Any = None


@dataclass
class Response:
    id: int
    result: Any
    exception: Exception | None


class Proxy:
    def __init__(
        self, obj: object, log_file: str | None = None, process_name: str | None = None
    ) -> None:
        self._obj = obj
        self._process_name = process_name
        self._requests = mp.Queue()
        self._responses = mp.Queue()
        self._process = mp.Process(
            target=_target,
            args=(obj, self._requests, self._responses, log_file, process_name),
        )
        self._process.start()
        self._futures: dict[int, Future] = {}
        self._futures_lock = threading.Lock()
        self._dead_process_error: RuntimeError | None = None
        self._closing = False
        self._closed = False
        self._next_id = it.count(1).__next__
        self._dispatcher = threading.Thread(
            target=self._dispatch_responses, name="mp-actors-dispatch", daemon=True
        )
        self._dispatcher.start()
        _register_proxy_for_atexit(self)

    def _process_error(self) -> RuntimeError:
        exit_code = self._process.exitcode
        name = f" '{self._process_name}'" if self._process_name else ""
        if exit_code is None:
            return RuntimeError(f"Child process{name} died unexpectedly")
        if exit_code < 0:
            return RuntimeError(
                f"Child process{name} was killed by signal {-exit_code}"
            )
        return RuntimeError(f"Child process{name} exited with code {exit_code}")

    def _fail_pending(self, error: Exception) -> None:
        with self._futures_lock:
            pending = list(self._futures.values())
            self._futures.clear()
        for future in pending:
            if not future.done():
                try:
                    future.set_exception(error)
                except InvalidStateError:
                    pass

    def _dispatch_responses(self) -> None:
        while True:
            try:
                response: Response = self._responses.get(timeout=0.1)
            except queue.Empty:
                if self._closing:
                    break
                if self._dead_process_error is None and not self._process.is_alive():
                    self._dead_process_error = self._process_error()
                    self._fail_pending(self._dead_process_error)
                continue
            except Exception:
                break
            if response.id == _SHUTDOWN_ID:
                break
            with self._futures_lock:
                future = self._futures.pop(response.id, None)
            if future is None:
                continue
            try:
                if response.exception:
                    future.set_exception(response.exception)
                else:
                    future.set_result(response.result)
            except InvalidStateError:
                pass

    @streamline_tracebacks()
    def __getattr__(self, name: str) -> Any:
        # For attributes that aren't methods, get them directly
        if not hasattr(self._obj, name):
            raise AttributeError(
                f"{type(self._obj).__name__} has no attribute '{name}'"
            )

        def response_future(
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            id: int | None = None,
            send_value: Any | None = None,
        ) -> Future:
            request = Request(
                id=id if id is not None else self._next_id(),
                method_name=name,
                args=args,
                kwargs=kwargs,
                send_value=send_value,
            )
            future: Future = Future()
            with self._futures_lock:
                if self._dead_process_error:
                    raise self._dead_process_error
                if self._closing:
                    raise RuntimeError("Proxy is closing")
                self._futures[request.id] = future
            try:
                self._requests.put_nowait(request)
            except BaseException:
                with self._futures_lock:
                    self._futures.pop(request.id, None)
                raise
            return future

        # Check if it's a method or property
        attr = getattr(self._obj, name)
        if inspect.isasyncgenfunction(attr):
            # Return an async generator wrapper function
            @streamline_tracebacks()
            async def async_gen_wrapper(
                *args: Any, **kwargs: Any
            ) -> AsyncGenerator[Any, Any]:
                try:
                    id = self._next_id()
                    send_value = None
                    while True:
                        send_value = yield await asyncio.wrap_future(
                            response_future(args, kwargs, id, send_value)
                        )
                        args, kwargs = (), {}
                except StopAsyncIteration:
                    return

            return async_gen_wrapper
        elif asyncio.iscoroutinefunction(attr):
            # Return an async wrapper function
            @streamline_tracebacks()
            async def async_method_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await asyncio.wrap_future(response_future(args, kwargs))

            return async_method_wrapper
        elif callable(attr):
            # Return a regular function wrapper
            @streamline_tracebacks()
            def method_wrapper(*args: Any, **kwargs: Any) -> Any:
                return response_future(args, kwargs).result()

            return method_wrapper
        else:
            # For non-callable attributes, get them directly
            return response_future(tuple(), dict()).result()

    def close(self):
        if self._closed:
            return
        self._closed = True
        _LIVE_PROXIES.discard(self)
        self._closing = True
        self._fail_pending(RuntimeError("Proxy is closing"))

        # terminate child process and force kill if needed
        self._process.terminate()
        self._process.join(timeout=1)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=1)

        # close and cancel queue feeder threads
        self._responses.close()
        self._responses.cancel_join_thread()
        self._requests.close()
        self._requests.cancel_join_thread()
        self._dispatcher.join(timeout=1)


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
    generators: dict[int, AsyncGenerator[Any, Any]] = {}
    while True:
        request: Request = await asyncio.get_event_loop().run_in_executor(
            None, requests.get
        )
        asyncio.create_task(
            _handle_request(obj, request, responses, generators)
        ).add_done_callback(lambda t: None if t.cancelled() else t.exception())


async def _handle_request(
    obj: object,
    request: Request,
    responses: mp.Queue,
    generators: dict[int, AsyncGenerator[Any, Any]],
) -> None:
    try:
        result_or_callable = getattr(obj, request.method_name)
        if inspect.isasyncgenfunction(result_or_callable):
            if request.id not in generators:
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
