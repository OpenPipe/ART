from dotenv import load_dotenv
import pytest

from .new import new_sandbox
from .sandbox import Provider

load_dotenv()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["daytona", "modal"])
async def test_sandbox(provider: Provider) -> None:
    async with new_sandbox(image="python:3.10", provider=provider) as sandbox:
        code, stdout = await sandbox.exec("echo 'Hello, world!'", 10)
        assert code == 0
        assert stdout == "Hello, world!\n"
