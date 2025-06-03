import art
import asyncio
import json
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from langfuse.types import SpanLevel
import litellm
import logging
from logging import Handler, LogRecord
import modal
from pathlib import Path
from pydantic import BaseModel
import re
import requests
from requests import adapters as requests_adapters
from requests.exceptions import SSLError
import shlex
from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
from sweagent.run.hooks.abstract import RunHook
from sweagent.run.hooks.apply_patch import SaveApplyPatchHook
from sweagent.run.run_replay import RunReplay
from sweagent.run.run_single import RunSingle, RunSingleConfig
from sweagent.types import AgentRunResult
from swebench.harness.modal_eval.run_evaluation_modal import app, run_instance_modal
from swebench.harness.test_spec.test_spec import make_test_spec
from swerex.deployment.modal import ModalDeployment
from swerex.exceptions import CommandTimeoutError
from swerex.runtime.abstract import BashAction
from typing import Any, Literal, overload

from config import get_config
from eval import eval_instance
from instances import Instance

# Add Langfuse callbacks for SWE-agent litellm calls
litellm.success_callback.append("langfuse")
litellm.failure_callback.append("langfuse")

# Suppress urllib3 retry warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


# # Custom filter to suppress swerex and rex-deploy related critical logs
# class SuppressSwerexLogsFilter(logging.Filter):
#     def filter(self, record):
#         # Suppress logs from rex-deploy loggers
#         if record.name.startswith("rex-deploy"):
#             return False
#         # Suppress swerex exception logs
#         if "swerex.exceptions" in record.getMessage() or "swerex.exceptions" in str(
#             record.exc_info
#         ):
#             return False
#         # Suppress CommandTimeoutError and BashIncorrectSyntaxError logs
#         if any(
#             error in record.getMessage()
#             for error in [
#                 "CommandTimeoutError",
#                 "BashIncorrectSyntaxError",
#                 "pexpect.exceptions.TIMEOUT",
#             ]
#         ):
#             return False
#         return True


# # Apply the filter to the root logger to catch all logs
# logging.getLogger().addFilter(SuppressSwerexLogsFilter())

# Disable printing the patch message to reduce log noise
SaveApplyPatchHook._print_patch_message = lambda *args, **kwargs: None


class ModelConfig(BaseModel):
    completion_kwargs: dict[str, Any] = {}
    max_input_tokens: int | None = None
    per_instance_cost_limit: float = 0.0
    system_prompt_suffix: str = ""
    xml_function_calling: bool = False


@overload
async def rollout(
    model: art.Model[ModelConfig],
    instance: Instance,
    completion_kwargs: dict[str, Any] | None = None,
    replay_trajectory_path: Path | None = None,
    return_run_single: Literal[False] = False,
    run_in_thread: bool = True,
) -> art.Trajectory: ...


@overload
async def rollout(
    model: art.Model[ModelConfig],
    instance: Instance,
    *,
    completion_kwargs: dict[str, Any] | None = None,
    replay_trajectory_path: Path | None = None,
    return_run_single: Literal[True],
    run_in_thread: bool = True,
) -> tuple[art.Trajectory, RunSingle]: ...


@observe(capture_input=False, capture_output=False)
async def rollout(
    model: art.Model[ModelConfig],
    instance: Instance,
    completion_kwargs: dict[str, Any] | None = None,
    replay_trajectory_path: Path | None = None,
    return_run_single: bool = False,
    reward_power: float = 1.0,
    run_in_thread: bool = True,
) -> art.Trajectory | tuple[art.Trajectory, RunSingle]:
    trajectory = art.Trajectory(messages_and_choices=[], reward=0.0)
    config = get_config(model, instance, completion_kwargs)
    if run_in_thread:
        run_single = await asyncio.to_thread(RunSingle.from_config, config)
    else:
        run_single = RunSingle.from_config(config)
    assert isinstance(run_single.agent, DefaultAgent)
    run_single.agent.logger.propagate = False
    run_single.agent.logger.handlers = []
    run_single.agent.logger.addHandler(
        LangfuseHandler(langfuse_context.get_current_trace_id() or "")
    )
    patch_get_model_requery_history(run_single.agent)
    if replay_trajectory_path:
        run_replay = RunReplay(
            traj_path=replay_trajectory_path,
            deployment=run_single.env.deployment,
            output_dir=Path("replays"),
        )
        run_replay._create_actions_file()
        run_single = run_replay._get_run_single()
        run_single.agent.replay_config = RunSingleConfig(  # type: ignore
            agent=run_replay.config.agent,
            problem_statement=run_single.problem_statement,  # type: ignore
            env=run_replay.config.env,
        )
    assert isinstance(config.agent, DefaultAgentConfig)
    trajectory = art.Trajectory(
        messages_and_choices=[], reward=0.0, tools=config.agent.tools.tools
    )
    if isinstance(run_single.env.deployment, ModalDeployment):
        run_single.add_hook(PatchRuntimeRunHook(run_single.env.deployment))
    if not instance["use_swebench_modal_harness"]:
        run_single.add_hook(
            RewardRunHook(instance, trajectory, run_single, reward_power)
        )
    try:
        if run_in_thread:
            await asyncio.to_thread(run_single.run)
        else:
            run_single.run()
    except RuntimeError as e:
        if not "Container process terminated" in str(e):
            raise e
        print(e)
    except SSLError as ssl_error:
        print(ssl_error)
    finally:
        try:
            if isinstance(run_single.env.deployment, ModalDeployment):
                await run_single.env.deployment.stop()
        except:
            pass
    if instance["use_swebench_modal_harness"]:
        await update_trajectory_with_swebench_modal_harness(
            instance, trajectory, run_single, reward_power
        )
    assert isinstance(run_single.agent, DefaultAgent)
    trajectory.messages_and_choices = run_single.agent.history
    if return_run_single:
        return trajectory, run_single
    else:
        return trajectory


class LangfuseHandler(Handler):
    """
    Custom handler to forward logs to Langfuse
    """

    def __init__(self, trace_id: str) -> None:
        self.langfuse = Langfuse()
        self.trace_id = trace_id
        super().__init__()

    def emit(self, record: LogRecord) -> None:
        if record.levelname in ["DEBUG", "INFO"]:
            return
        levels: dict[str, SpanLevel] = {
            "WARNING": "WARNING",
            "ERROR": "ERROR",
            "CRITICAL": "ERROR",
        }
        self.langfuse.event(
            trace_id=self.trace_id,
            name="agent-logger",
            level=levels[record.levelname],
            status_message=record.getMessage(),
        )
        self.langfuse.flush()


def patch_get_model_requery_history(agent: DefaultAgent) -> None:
    get_model_requery_history = agent.get_model_requery_history

    def _get_model_requery_history(
        error_template: str, *, output: str, **kwargs: str | int | float | bool | None
    ) -> list[dict[str, str]]:
        history = get_model_requery_history(error_template, output=output, **kwargs)
        agent.history = history
        return history

    agent.get_model_requery_history = _get_model_requery_history


class PatchRuntimeRunHook(RunHook):
    """
    Custom run hook to patch the runtime of the deployment
    """

    def __init__(self, deployment: ModalDeployment) -> None:
        self.deployment = deployment

    def on_instance_start(self, *args: Any, **kwargs: Any) -> None:
        runtime = self.deployment.runtime
        session = requests.Session()
        retry = requests_adapters.Retry(
            total=5,  # Increased from 3
            backoff_factor=1,  # Increased from 0.1, using int instead of float
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"],
            # Also retry on SSL errors
            raise_on_status=False,
        )
        adapter = requests_adapters.HTTPAdapter(
            max_retries=retry,
            pool_connections=10,
            pool_maxsize=10,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        def _request(
            endpoint: str, request: BaseModel | None, output_class: Any
        ) -> Any:
            response = session.post(
                f"{runtime._api_url}/{endpoint}",
                json=request.model_dump() if request else None,
                headers=runtime._headers,
            )
            runtime._handle_response_errors(response)
            return output_class(**response.json())

        runtime._request = _request

        stop = self.deployment.stop

        async def _stop() -> None:
            if self.deployment._sandbox is not None:
                sandbox_id = self.deployment._sandbox.object_id
            await stop()
            if sandbox_id:
                try:
                    sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
                    await sandbox.terminate.aio()
                except Exception as e:
                    print(e)

        self.deployment.stop = _stop

        # # Patch the runtime to use longer default timeouts for commands
        # original_run_in_session = runtime.run_in_session

        # def patched_run_in_session(action):
        #     if hasattr(action, "timeout") and action.timeout == 30.0:
        #         action.timeout = 120.0
        #     return original_run_in_session(action)

        # runtime.run_in_session = patched_run_in_session


class RewardRunHook(RunHook):
    """
    Custom run hook to update a trajectory with test results while the environment is still running
    """

    def __init__(
        self,
        instance: Instance,
        trajectory: art.Trajectory,
        run_single: RunSingle,
        reward_power: float,
    ) -> None:
        self.instance = instance
        self.trajectory = trajectory
        self.run_single = run_single
        self.reward_power = reward_power

    def on_instance_completed(self, *, result: AgentRunResult) -> None:
        # TODO: Address potential reward hacking
        # An agent could potentially modify the tests to pass
        # without actually addressing the issue.
        update_trajectory(
            self.trajectory,
            self.instance,
            self.reward_power,
            **asyncio.run(
                eval_instance(self.instance, self.run_single.env.deployment.runtime)
            ),
        )


async def update_trajectory_with_swebench_modal_harness(
    instance: Instance,
    trajectory: art.Trajectory,
    run_single: RunSingle,
    reward_power: float,
) -> None:
    """
    Update a trajectory with test results from the SWE-bench modal harness
    """
    async with app.run():
        output = await run_instance_modal.remote.aio(
            test_spec=make_test_spec(instance),  # type: ignore
            pred={
                "model_name_or_path": "model_name",
                "model_patch": run_single.agent.info["submission"],  # type: ignore
                "instance_id": instance["instance_id"],
            },
            run_id="run_id",
            timeout=1800,
        )
    tests_status = json.loads(output.report_json_str)[instance["instance_id"]][
        "tests_status"
    ]
    update_trajectory(
        trajectory,
        instance,
        reward_power,
        num_failed_f2p=len(tests_status["FAIL_TO_PASS"]["failure"]),
        num_passed_f2p=len(tests_status["FAIL_TO_PASS"]["success"]),
        num_failed_p2p=len(tests_status["PASS_TO_PASS"]["failure"]),
        num_passed_p2p=len(tests_status["PASS_TO_PASS"]["success"]),
    )


def update_trajectory(
    trajectory: art.Trajectory,
    instance: Instance,
    reward_power: float,
    num_failed_f2p: int,
    num_passed_f2p: int,
    num_failed_p2p: int,
    num_passed_p2p: int,
) -> None:
    """
    Update a trajectory with instance test results
    """
    # Penalize missing or errored tests
    num_missing = max(
        len(instance["FAIL_TO_PASS"])
        + len(instance["PASS_TO_PASS"])
        - (num_failed_f2p + num_passed_f2p + num_failed_p2p + num_passed_p2p),
        0,
    )
    # Max reward (1.0) occurs when all failing tests pass, no passing tests regress, and no tests are missing or errored.
    # A zero reward (0.0) reflects the status quo with no net change in test outcomes.
    # Negative rewards indicate more tests fail after the rollout than before.
    net_change = num_passed_f2p - num_failed_p2p - num_missing
    trajectory.reward = (
        (net_change / len(instance["FAIL_TO_PASS"])) ** reward_power
        if net_change > 0
        else net_change / max(len(instance["PASS_TO_PASS"]), 1)
    )
