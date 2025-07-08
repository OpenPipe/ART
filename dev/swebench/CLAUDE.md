Current goal is to get the tests passing. Example:

`uv run pytest sandbox/test.py -n 18 -v`

Investigate a single test:

`uv run pytest sandbox/test.py::test_run_tests[0-daytona] -v`

You'll need to implement the `apply_patch` and `run_tests` methods in the `sandbox.sandbox.Sandbox` class.

Look at ./daytona.py for previous work at getting tests to pass.

Logic should be as simple as possible and general, not specific to anyone test case.

Logic that can be shared between providers should be implemented in the `sandbox.sandbox.Sandbox` class.

Logic that is specific to a provider should be implemented in the `sandbox.daytona.DaytonaSandbox` and `sandbox.modal.ModalSandbox` classes.