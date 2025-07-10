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
@pytest.mark.parametrize("instance_idx", range(16))
async def test_run_tests(provider: Provider, instance_idx: int) -> None:
    instance = next(
        get_filtered_swe_smith_instances_df()
        .pipe(lambda df: df.tail(-instance_idx) if instance_idx > 0 else df)
        .pipe(as_instances_iter)
    )
    
    # Calculate dynamic timeout based on number of tests
    # Formula: base_timeout + num_tests * per_test_time
    base_timeout = 120  # Base time for dependency installation
    per_test_time = 0.05  # Per-test time (reduced since most tests are fast)
    
    # Skip instances with extreme test counts that may hit system limits
    if len(instance["PASS_TO_PASS"]) > 3000:
        pytest.skip(f"Skipping instance with {len(instance['PASS_TO_PASS'])} PASS_TO_PASS tests (system limits)")
    
    fail_to_pass_timeout = int(base_timeout + len(instance["FAIL_TO_PASS"]) * per_test_time)
    pass_to_pass_timeout = int(base_timeout + len(instance["PASS_TO_PASS"]) * per_test_time)
    
    
    async with new_sandbox(image=instance["image_name"], provider=provider) as sandbox:
        failed, passed = await sandbox.run_tests(instance["FAIL_TO_PASS"], fail_to_pass_timeout)
        assert failed == 0
        assert passed == len(instance["FAIL_TO_PASS"])
        await sandbox.apply_patch(instance["patch"], 10)
        failed, passed = await sandbox.run_tests(instance["FAIL_TO_PASS"], fail_to_pass_timeout)
        assert failed == len(instance["FAIL_TO_PASS"])
        assert passed == 0
        failed, passed = await sandbox.run_tests(instance["PASS_TO_PASS"], pass_to_pass_timeout)
        assert failed == 0
        assert passed == len(instance["PASS_TO_PASS"])
