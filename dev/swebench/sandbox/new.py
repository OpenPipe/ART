import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import daytona_sdk
import modal
from tenacity import (
    retry,
    retry_if_exception_message,
    stop_after_attempt,
    wait_exponential,
)

from .daytona import DaytonaSandbox
from .modal import ModalSandbox
from .sandbox import Provider, Sandbox

daytona = daytona_sdk.AsyncDaytona()
modal_app_task: asyncio.Task[modal.App] | None = None

# Enable Modal output to see image build logs
modal.enable_output()


# Retry decorator for sandbox creation
create_sandbox_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_message(match="No available runners|timeout|504 Gateway"),
)


@asynccontextmanager
async def new_sandbox(
    *, image: str, provider: Provider, timeout: int = 60
) -> AsyncIterator[Sandbox]:
    """
    Context manager for a new sandbox.

    Args:
        image: The image to use for the sandbox.
        provider: The provider to use for the sandbox: "daytona" or "modal".

    Returns:
        A context manager that yields a sandbox object.

    Example:
        ```python
        async with new_sandbox(image=instance["image_name"], provider="daytona") as sandbox:
            failed, passed = await sandbox.eval(instance["FAIL_TO_PASS"])
        ```
    """
    if provider == "daytona":
        global daytona

        @create_sandbox_retry
        async def create_daytona_sandbox():
            global daytona
            try:
                return await daytona.create(
                    daytona_sdk.CreateSandboxFromImageParams(image=image),
                    timeout=timeout,
                )
            except daytona_sdk.DaytonaError as e:
                if "Event loop is closed" in str(e):
                    await daytona.close()
                    daytona = daytona_sdk.AsyncDaytona()
                raise

        try:
            sandbox = await create_daytona_sandbox()
            yield DaytonaSandbox(sandbox)
        finally:
            try:
                await sandbox.delete()
            except Exception:
                pass
    else:
        global modal_app_task
        if modal_app_task is None:
            modal_app_task = asyncio.create_task(
                modal.App.lookup.aio("swebench", create_if_missing=True)
            )
        app = await modal_app_task
        sandbox = await modal.Sandbox.create.aio(
            app=app, image=modal.Image.from_registry(image), timeout=timeout
        )
        try:
            yield ModalSandbox(sandbox)
        finally:
            await sandbox.terminate.aio()
