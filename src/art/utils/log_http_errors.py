from functools import wraps
import httpx


def log_http_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            # raise a new exception with the status code, url, and "detail" key if it exists
            try:
                detail = e.response.json().get("detail", None)
            except Exception:
                # if we can't parse the response as json, just raise the original exception
                raise e
            raise Exception(
                f"[HTTP {e.response.status_code}] {e.request.url} {detail}"
            ) from e

    return wrapper
