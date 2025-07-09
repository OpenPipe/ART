Current goal is to get the tests passing. Example:

`uv run pytest sandbox/test.py -n 18 -v`

To investigate a single test, run:

`uv run pytest sandbox/test.py::test_run_tests[0-daytona] -v`

Logic should be as simple as possible and general, not specific to anyone test case.

Logic that can be shared between providers should be implemented in the `sandbox.sandbox.Sandbox` class.

Logic that is specific to a provider should be implemented in the `sandbox.daytona.DaytonaSandbox` and `sandbox.modal.ModalSandbox` classes.

At this point all tests are passing, so now we need to evaluate the following questions:

1. Is the implementation correct? Is there any evidence of inappropriate "reward hacking" to get the tests to pass? If there is, we need to address that first and foremost and then try to get tests passing properly. Take as long as necessary if this is an actual issue.
2. Is everything in the implementation strictly necessary? Remove every mitigation and fallback that is not necessary. The best way to do this is to try removing mitigations to see if the tests still pass. Keep working on this until you are confident that the implementation is correct and absolutely minimal.
3. Is there a `safe_exec` method that we could use to run code in the sandbox that shares any of the remaining necessary mitigations left in `apply_patch` and `run_tests`? Abstracting this out may be desirable and useful for when we add additional sandbox functions.

Thanks for your help in advance, it is sincerely appreciated.