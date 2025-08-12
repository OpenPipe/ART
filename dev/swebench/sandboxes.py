import asyncio

import modal


async def terminate_sandboxes() -> None:
    sandboxes: list[modal.Sandbox] = []
    async for sandbox in modal.Sandbox.list.aio(
        app_id=modal.App.lookup("swe-rex", create_if_missing=True).app_id
    ):
        sandboxes.append(sandbox)
    _ = await asyncio.gather(*[sandbox.terminate.aio() for sandbox in sandboxes])
