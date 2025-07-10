from abc import ABC, abstractmethod
from typing import Literal

Provider = Literal["daytona", "modal"]


class Sandbox(ABC):
    """
    Base class for all sandboxes.

    Provides a common interface for all sandboxes, as well as shared logic and functionality.
    """

    provider: Provider

    @abstractmethod
    async def exec(self, command: str, timeout: int) -> tuple[int, str]:
        raise NotImplementedError

    async def apply_patch(self, patch: str, timeout: int) -> None:
        # Write patch to file using heredoc
        exit_code, output = await self.exec(
            f"cat > /tmp/patch.txt << 'EOF'\n{patch}\nEOF", timeout
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to write patch: {output}")

        # Apply the patch in the /testbed directory
        exit_code, output = await self.exec(
            "cd /testbed && patch -p1 < /tmp/patch.txt", timeout
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to apply patch: {output}")

    async def run_tests(self, tests: list[str], timeout: int) -> tuple[int, int]:
        import re

        # First, ensure uv is installed
        exit_code, _ = await self.exec("which uv", timeout)
        if exit_code != 0:
            await self.exec(
                "curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet", timeout
            )

        # Set up uv environment
        uv_cmd = "UV_SYSTEM_PYTHON=true PATH=$HOME/.local/bin:$PATH uv"

        # Install pytest first
        await self.exec(f"{uv_cmd} pip install -q pytest", timeout)
        
        # Try to install the project itself if it has a setup.py or pyproject.toml
        # This will install all project dependencies
        setup_exists = await self.exec("test -f /testbed/setup.py && echo exists", timeout)
        pyproject_exists = await self.exec("test -f /testbed/pyproject.toml && echo exists", timeout)
        if setup_exists[1].strip() == "exists" or pyproject_exists[1].strip() == "exists":
            await self.exec(f"cd /testbed && {uv_cmd} pip install -q -e . 2>/dev/null", timeout)

        # Write test list in chunks to avoid command length limits
        # First, clear any existing file
        await self.exec("rm -f /tmp/tests.txt", timeout)
        
        # Write tests in chunks
        chunk_size = 50  # Write 50 tests at a time
        for i in range(0, len(tests), chunk_size):
            chunk = tests[i:i + chunk_size]
            test_chunk = "\n".join(chunk)
            exit_code, output = await self.exec(
                f"cat >> /tmp/tests.txt << 'EOF'\n{test_chunk}\nEOF", timeout
            )
            if exit_code != 0:
                raise RuntimeError(f"Failed to write test chunk: {output}")

        # Create a Python script to run pytest with proper handling of special characters
        pytest_script = """
import sys
sys.path.insert(0, '/testbed')

with open('/tmp/tests.txt', 'r') as f:
    tests = [line.strip() for line in f if line.strip()]

import pytest
args = ['-v', '-o', 'addopts=', '--tb=short', '--no-header'] + tests
exit_code = pytest.main(args)
sys.exit(exit_code)
"""
        exit_code, output = await self.exec(
            f"cat > /tmp/run_pytest.py << 'EOF'\n{pytest_script}\nEOF", timeout
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to write pytest script: {output}")

        # Run the tests with retry logic for missing dependencies
        max_retries = 20
        for attempt in range(max_retries):
            exit_code, output = await self.exec(
                "cd /testbed && python /tmp/run_pytest.py 2>&1", timeout
            )

            # Check for missing dependencies and try to install them
            if "ModuleNotFoundError" in output or "ImportError" in output:
                # Extract missing module names using a single pattern
                missing_modules = list(
                    set(re.findall(r"No module named [']([^']+)[']", output))
                )

                if missing_modules and attempt < max_retries - 1:
                    # Try to install missing modules
                    for module in missing_modules:
                        # Handle special cases where import name differs from package name
                        package_name = module
                        if module == "OpenSSL":
                            package_name = "pyOpenSSL"
                        elif module == "yaml":
                            package_name = "pyyaml"
                        elif module == "cv2":
                            package_name = "opencv-python"
                        
                        await self.exec(f"{uv_cmd} pip install -q {package_name}", timeout)

                    # Retry the test
                    continue

            # No more dependency issues or max retries reached
            break

        # Parse results - look for FAILED and PASSED (with leading space like in pytest output)
        # Note: We only count test results, not pytest framework messages
        # Test results appear in format like "tests/test_file.py::test_name FAILED"
        failed_count = output.count(" FAILED")
        passed_count = output.count(" PASSED")
        
        # Handle edge case: if pytest exits with code 4 (collection errors) and no tests ran,
        # we should count the requested tests as failures since they couldn't be executed
        if exit_code == 4 and failed_count == 0 and passed_count == 0:
            # Check if there were collection errors preventing tests from running
            if "ERROR collecting" in output or "ImportError" in output or "ModuleNotFoundError" in output:
                # Count all requested tests as failures since they couldn't run
                failed_count = len(tests)

        return failed_count, passed_count
