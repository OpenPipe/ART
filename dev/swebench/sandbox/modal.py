import modal

from .sandbox import Provider, Sandbox


class ModalSandbox(Sandbox):
    """
    Modal sandbox.

    Wraps a Modal sandbox with the shared Sandbox interface.
    """

    provider: Provider = "modal"

    def __init__(self, sandbox: modal.Sandbox) -> None:
        self._sandbox = sandbox

    async def exec(self, command: str, timeout: int) -> tuple[int, str]:
        process = await self._sandbox.exec.aio(
            "/bin/sh", "-c", command, timeout=timeout
        )
        exit_code = await process.wait.aio()
        stdout = await process.stdout.read.aio()
        return exit_code, stdout
