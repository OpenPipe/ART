import asyncio
from contextlib import asynccontextmanager
import daytona_sdk
import modal
from typing import AsyncIterator

from .daytona import DaytonaSandbox
from .modal import ModalSandbox
from .sandbox import Provider, Sandbox

daytona = daytona_sdk.AsyncDaytona()
modal_app_task: asyncio.Task[modal.App] | None = None


@asynccontextmanager
async def new_sandbox(*, image: str, provider: Provider) -> AsyncIterator[Sandbox]:
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
        sandbox = await daytona.create(
            daytona_sdk.CreateSandboxFromImageParams(image=image)
        )
        try:
            yield DaytonaSandbox(sandbox)
        finally:
            await sandbox.delete()
    else:
        global modal_app_task
        if modal_app_task is None:
            modal_app_task = asyncio.create_task(
                modal.App.lookup.aio("swebench", create_if_missing=True)
            )
        app = await modal_app_task
        sandbox = await modal.Sandbox.create.aio(
            app=app, image=modal.Image.from_registry(image)
        )
        try:
            yield ModalSandbox(sandbox)
        finally:
            await sandbox.terminate.aio()
