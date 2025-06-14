import art
import asyncio
import json
from langfuse.decorators import observe
import modal
from pathlib import Path
from pydantic import BaseModel
import requests
from requests import adapters as requests_adapters
from requests.exceptions import ConnectTimeout, SSLError
from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
from sweagent.run.hooks.abstract import RunHook
from sweagent.run.run_replay import RunReplay
from sweagent.run.run_single import RunSingle, RunSingleConfig
from sweagent.types import AgentRunResult
from swebench.harness.modal_eval.run_evaluation_modal import app, run_instance_modal
from swebench.harness.test_spec.test_spec import make_test_spec
from swerex.deployment.modal import ModalDeployment
from swerex.exceptions import CommandTimeoutError
from typing import Any, Literal, overload

from config import get_config
from eval import eval_instance
from logs import setup_agent_logger
from instances import Instance
from run import run


class ModelConfig(BaseModel):
    completion_kwargs: dict[str, Any] = {}
    max_input_tokens: int | None = None
    per_instance_call_limit: int = 0
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
@art.retry(
    max_attempts=2,
    exceptions=(modal.exception.SandboxTimeoutError,),
)
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
    run_single = await run(RunSingle.from_config, run_in_thread, config)
    assert isinstance(run_single.agent, DefaultAgent)
    setup_agent_logger(run_single.agent)
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
        messages_and_choices=[],
        reward=0.0,
        tools=(
            config.agent.tools.tools if not model.config.xml_function_calling else None
        ),
    )
    if isinstance(run_single.env.deployment, ModalDeployment):
        run_single.add_hook(PatchRuntimeRunHook(run_single.env.deployment))
    if not instance["use_swebench_modal_harness"]:
        run_single.add_hook(
            RewardRunHook(instance, trajectory, run_single, reward_power)
        )
    try:
        await run(run_single.run, run_in_thread)
    except modal.exception.RemoteError as e:
        print(instance["instance_id"])
        print(e)
    except ConnectionError as e:
        print(e)
    except ConnectTimeout as e:
        print(e)
    except CommandTimeoutError as e:
        print(e)
    except RuntimeError as e:
        if not "Container process terminated" in str(e):
            raise e
        print(e)
    except SSLError as ssl_error:
        print(ssl_error)
    except TimeoutError as e:
        if not "Runtime did not start within" in str(e):
            raise e
        print(e)
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
            else:
                sandbox_id = None
            await stop()
            if sandbox_id:
                try:
                    sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
                    await sandbox.terminate.aio()
                except Exception as e:
                    print(e)

        self.deployment.stop = _stop


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

    def on_instance_start(self, *args: Any, **kwargs: Any) -> None:
        eval_result = asyncio.run(
            eval_instance(self.instance, self.run_single.env.deployment.runtime)
        )
        if (
            eval_result["num_failed_f2p"] == 0
            and eval_result["num_passed_f2p"] == 0
            and eval_result["num_failed_p2p"] == 0
            and eval_result["num_passed_p2p"] == 0
        ):
            print(
                f"eval_result: {eval_result}, instance_id: {self.instance['instance_id']}"
            )

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
    # calculate the following (clamped) metrics:
    # progress towards fixing failing tests
    # failure to fix failing tests
    # maintenance of passing tests
    # regression of passing tests
    progress = clamp(num_passed_f2p / len(instance["FAIL_TO_PASS"]), 0.0, 1.0)
    failure = clamp(num_failed_f2p / len(instance["FAIL_TO_PASS"]), 0.0, 1.0)
    maintenance = clamp(
        num_passed_p2p / max(len(instance["PASS_TO_PASS"]), 1), 0.0, 1.0
    )
    regression = clamp(num_failed_p2p / max(len(instance["PASS_TO_PASS"]), 1), 0.0, 1.0)
    # reconcile metrics pessimistically
    progress, failure = min(progress, 1 - failure), max(failure, 1 - progress)
    maintenance, regression = (
        min(maintenance, 1 - regression),
        max(regression, 1 - maintenance),
    )
    # determine if the instance was successfully resolved
    resolved = (
        num_failed_f2p == 0
        and num_passed_f2p == len(instance["FAIL_TO_PASS"])
        and num_failed_p2p == 0
        and num_passed_p2p == len(instance["PASS_TO_PASS"])
    )
    # calculate reward and save metrics
    trajectory.reward = (
        0.45 * maintenance + 0.45 * progress**reward_power + 0.1 * resolved
    )
    trajectory.metrics["progress"] = progress
    trajectory.metrics["maintenance"] = maintenance
    trajectory.metrics["resolved"] = resolved


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))
