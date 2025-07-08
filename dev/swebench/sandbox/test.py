from dotenv import load_dotenv
import pytest

from ..instances import as_instances_iter, get_filtered_swe_smith_instances_df
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


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["daytona", "modal"])
@pytest.mark.parametrize("instance_idx", range(8))
async def test_run_tests(provider: Provider, instance_idx: int) -> None:
    instance = next(
        get_filtered_swe_smith_instances_df()
        .pipe(lambda df: df.tail(-instance_idx) if instance_idx > 0 else df)
        .pipe(as_instances_iter)
    )
    async with new_sandbox(image=instance["image_name"], provider=provider) as sandbox:
        await sandbox.apply_patch(instance["patch"], 10)
        failed, passed = await sandbox.run_tests(instance["FAIL_TO_PASS"], 60)
        assert failed == len(instance["FAIL_TO_PASS"])
        assert passed == 0
        failed, passed = await sandbox.run_tests(instance["PASS_TO_PASS"], 60)
        assert failed == 0
        assert passed == len(instance["PASS_TO_PASS"])
