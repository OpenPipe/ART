import functools
import inspect
import sys
from types import TracebackType
from typing import Any, Callable, TypeVar, cast

T = TypeVar("T", bound=Callable[..., Any])


def streamline_tracebacks() -> Callable[[T], T]:
    def decorator(func: T) -> T:
        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise e.with_traceback(streamlined_traceback())

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                raise e.with_traceback(streamlined_traceback())

        return cast(T, async_wrapper if is_async else wrapper)

    return decorator


def streamlined_traceback() -> TracebackType | None:
    traceback = sys.exc_info()[2]
    if traceback is None:
        return None
    child_traceback, _ = get_child_traceback(traceback)
    return child_traceback


def get_child_traceback(
    traceback: TracebackType,
) -> tuple[TracebackType, bool]:
    if not traceback.tb_next:
        return traceback, False
    next_traceback, mp_actor_code = get_child_traceback(traceback.tb_next)
    if mp_actor_code:
        return next_traceback, True
    elif "/mp_actors/" in traceback.tb_frame.f_code.co_filename:
        return next_traceback, True
    return traceback, False
