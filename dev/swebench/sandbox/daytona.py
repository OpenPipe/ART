import daytona_sdk

from .sandbox import Provider, Sandbox


class DaytonaSandbox(Sandbox):
    """
    Daytona sandbox.

    Wraps a Daytona sandbox with the shared Sandbox interface.
    """

    provider: Provider = "daytona"

    def __init__(self, sandbox: daytona_sdk.AsyncSandbox) -> None:
        self._sandbox = sandbox

    async def exec(self, command: str, timeout: int) -> tuple[int, str]:
        result = await self._sandbox.process.exec(command, timeout=timeout)
        return int(result.exit_code), result.result
