import art
import asyncio
import json
from langfuse.decorators import observe
import litellm
from pathlib import Path
from pydantic import BaseModel
import re
from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
from sweagent.run.hooks.abstract import RunHook
from sweagent.run.run_replay import RunReplay
from sweagent.run.run_single import RunSingle, RunSingleConfig
from sweagent.types import AgentRunResult
from swebench.harness.modal_eval.run_evaluation_modal import app, run_instance_modal
from swebench.harness.test_spec.test_spec import make_test_spec
from swerex.runtime.abstract import BashAction
from typing import Any, Literal, overload

from config import get_config
from instances import Instance


litellm.success_callback.append("langfuse")
litellm.failure_callback.append("langfuse")


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


@observe(capture_output=False)
async def rollout(
    model: art.Model[ModelConfig],
    instance: Instance,
    completion_kwargs: dict[str, Any] | None = None,
    replay_trajectory_path: Path | None = None,
    return_run_single: bool = False,
    run_in_thread: bool = True,
) -> art.Trajectory | tuple[art.Trajectory, RunSingle]:
    trajectory = art.Trajectory(messages_and_choices=[], reward=0.0)
    config = get_config(model, instance, completion_kwargs)
    if run_in_thread:
        run_single = await asyncio.to_thread(RunSingle.from_config, config)
    else:
        run_single = RunSingle.from_config(config)
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
    if not instance.get("test_patch"):
        run_single.add_hook(RewardRunHook(instance, trajectory, run_single))
    if run_in_thread:
        await asyncio.to_thread(run_single.run)
    else:
        run_single.run()
    if instance.get("test_patch"):
        await update_trajectory_with_swebench_modal_harness(
            instance, trajectory, run_single
        )
    assert isinstance(run_single.agent, DefaultAgent)
    trajectory.messages_and_choices = run_single.agent.history
    if return_run_single:
        return trajectory, run_single
    else:
        return trajectory


class RewardRunHook(RunHook):
    def __init__(
        self, instance: Instance, trajectory: art.Trajectory, run_single: RunSingle
    ) -> None:
        self.instance = instance
        self.trajectory = trajectory
        self.run_single = run_single

    def on_instance_completed(self, *, result: AgentRunResult) -> None:
        # TODO: Address potential reward hacking
        # An agent could potentially modify the tests to pass
        # without actually addressing the issue.
        num_failed_f2p, num_passed_f2p = self._get_test_results(
            self.instance["FAIL_TO_PASS"]
        )
        num_failed_p2p, num_passed_p2p = self._get_test_results(
            self.instance["PASS_TO_PASS"]
        )
        update_trajectory(
            self.trajectory,
            self.instance,
            num_failed_f2p,
            num_passed_f2p,
            num_failed_p2p,
            num_passed_p2p,
        )

    def _get_test_results(self, tests: list[str]) -> tuple[int, int]:
        observation = asyncio.run(
            self.run_single.env.deployment.runtime.run_in_session(
                BashAction(
                    command=f"cd /testbed && python -m pytest {' '.join(tests)}",
                    check="silent",
                )
            )
        )
        summary_line = observation.output.splitlines()[-1]
        failed_match = re.search(r"(\d+)\s+failed", summary_line)
        passed_match = re.search(r"(\d+)\s+passed", summary_line)
        num_failed = int(failed_match.group(1)) if failed_match else 0
        num_passed = int(passed_match.group(1)) if passed_match else 0
        return num_failed, num_passed


async def update_trajectory_with_swebench_modal_harness(
    instance: Instance, trajectory: art.Trajectory, run_single: RunSingle
) -> None:
    async with app.run():
        output = await run_instance_modal.remote.aio(
            test_spec=make_test_spec(instance),  # type: ignore
            pred={
                "model_name_or_path": "model_name",
                "model_patch": run_single.agent.info["submission"],  # type: ignore
                "instance_id": instance["instance_id"],
            },
            run_id="run_id",
            timeout=1200,
        )
    tests_status = json.loads(output.report_json_str)[instance["instance_id"]][
        "tests_status"
    ]
    update_trajectory(
        trajectory,
        instance,
        num_failed_f2p=len(tests_status["FAIL_TO_PASS"]["failure"]),
        num_passed_f2p=len(tests_status["FAIL_TO_PASS"]["success"]),
        num_failed_p2p=len(tests_status["PASS_TO_PASS"]["failure"]),
        num_passed_p2p=len(tests_status["PASS_TO_PASS"]["success"]),
    )


def update_trajectory(
    trajectory: art.Trajectory,
    instance: Instance,
    num_failed_f2p: int,
    num_passed_f2p: int,
    num_failed_p2p: int,
    num_passed_p2p: int,
) -> None:
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
    trajectory.reward = (num_passed_f2p - num_failed_p2p - num_missing) / len(
        instance["FAIL_TO_PASS"]
    )
