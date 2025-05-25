import art
import asyncio
from pathlib import Path
from pydantic import BaseModel
import re
from sweagent.agent.agents import DefaultAgent
from sweagent.run.hooks.abstract import RunHook
from sweagent.run.run_replay import RunReplay
from sweagent.run.run_single import RunSingle, RunSingleConfig
from sweagent.types import AgentRunResult
from swerex.runtime.abstract import BashAction

from config import get_config
from instances import Instance


class ModelConfig(BaseModel):
    max_input_tokens: int | None = None
    per_instance_cost_limit: float = 0.0
    xml_function_calling: bool = False


async def rollout(
    model: art.Model[ModelConfig],
    instance: Instance,
    replay_trajectory_path: Path | None = None,
) -> tuple[art.Trajectory, RunSingle]:
    trajectory = art.Trajectory(messages_and_choices=[], reward=0.0)
    run_single = await asyncio.to_thread(
        RunSingle.from_config, get_config(model, instance)
    )
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
    assert isinstance(run_single.agent, DefaultAgent)
    if not instance.get("test_patch"):
        run_single.add_hook(RewardRunHook(instance, trajectory, run_single))
    await asyncio.to_thread(run_single.run)
    trajectory.messages_and_choices = run_single.agent.history
    return trajectory, run_single


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
        # Penalize missing or errored tests
        num_missing = max(
            len(self.instance["FAIL_TO_PASS"])
            + len(self.instance["PASS_TO_PASS"])
            - (num_failed_f2p + num_passed_f2p + num_failed_p2p + num_passed_p2p),
            0,
        )
        # Max reward (1.0) occurs when all failing tests pass, no passing tests regress, and no tests are missing or errored.
        # A zero reward (0.0) reflects the status quo with no net change in test outcomes.
        # Negative rewards indicate more tests fail after the rollout than before.
        self.trajectory.reward = (num_passed_f2p - num_failed_p2p - num_missing) / len(
            self.instance["FAIL_TO_PASS"]
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
