import re
import shlex
from typing import TypedDict

from instances import Instance
from swerex.exceptions import CommandTimeoutError
from swerex.runtime.abstract import AbstractRuntime, BashAction


class EvalResult(TypedDict):
    num_failed_f2p: int
    num_passed_f2p: int
    num_failed_p2p: int
    num_passed_p2p: int


async def eval_instance(
    instance: Instance, runtime: AbstractRuntime, timeout: float = 240.0
) -> EvalResult:
    """
    Evaluate a problem instance using a SWE-ReX runtime by running the fail-to-pass
    and pass-to-pass tests.

    Args:
        instance: The instance to evaluate.
        runtime: The runtime to use.
        timeout: The timeout for the tests. (default: 240.0)

    Returns:
        The evaluation result.
    """
    num_failed_f2p, num_passed_f2p = await eval_tests(
        instance["FAIL_TO_PASS"], runtime, timeout
    )
    num_failed_p2p, num_passed_p2p = await eval_tests(
        instance["PASS_TO_PASS"], runtime, timeout
    )
    return EvalResult(
        num_failed_f2p=num_failed_f2p,
        num_passed_f2p=num_passed_f2p,
        num_failed_p2p=num_failed_p2p,
        num_passed_p2p=num_passed_p2p,
    )


async def eval_tests(
    tests: list[str], runtime: AbstractRuntime, timeout: float
) -> tuple[int, int]:
    if not tests:
        return 0, 0

    # We batch tests to avoid exceeding the max command length
    base_cmd = "cd /testbed && python -m pytest "
    max_len = 16384
    results = []
    batch, batch_len = [], 0
    for test in tests + [None]:  # Sentinel to trigger final batch
        if test is None or (
            batch and batch_len + len(shlex.quote(test or "")) + 1 > max_len
        ):
            if batch:
                command = f"{base_cmd}{' '.join(map(shlex.quote, batch))}"
                try:
                    observation = await runtime.run_in_session(
                        BashAction(command=command, check="silent", timeout=timeout)
                    )
                    if lines := observation.output.splitlines():
                        s = lines[-1]
                        failed = (
                            int(m.group(1))
                            if (m := re.search(r"(\d+)\s+failed", s))
                            else 0
                        )
                        passed = (
                            int(m.group(1))
                            if (m := re.search(r"(\d+)\s+passed", s))
                            else 0
                        )
                        results.append((failed, passed))
                except CommandTimeoutError:
                    results.append((len(batch), 0))
                except Exception as e:
                    print(f"Error running tests: {e}")
                    results.append((len(batch), 0))
                batch, batch_len = [], 0
        if test:
            batch.append(test)
            batch_len += len(shlex.quote(test)) + 1

    return (
        (sum(f for f, _ in results), sum(p for _, p in results)) if results else (0, 0)
    )
