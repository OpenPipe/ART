from typing import List
import json
from litellm import acompletion
from textwrap import dedent
from pydantic import BaseModel, Field
from rich import print
import art
import weave


class RolloutScore(BaseModel):
    rollout_id: str = Field(description="The id of the rollout being scored.")
    explanation: str = Field(
        description="A short explanation of why you gave this score."
    )
    score: float = Field(description="A score between 0 and 1.")


class GroupJudgeResponse(BaseModel):
    scores: List[RolloutScore] = Field(description="The scores for each rollout.")


DEFAULT_RUBRIC = dedent(
    """         
        - A rollout that achieves its goal should always get a significantly higher score than a rollout that does not achieve its goal.
        - A rollout that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a rollout that achieves its goal less efficiently.
        - If one rollout is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
        - You may give some partial credit for a rollout that makes progress towards its goal but does not complete it.
"""
)


class GroupJudge:
    """LLM-based judge for groups of trajectories.

    Parameters
    ----------
    judge_model: str, default "openai/o3"
        The model that will be used to score the trajectories.
    rubric: str, default :data:`DEFAULT_RUBRIC`
        A replacement *rubric* that will overwrite the default bullet list
        under the "Grading standards:" section of :data:`DEFAULT_RUBRIC`.
        If *None*, the built-in grading standards are kept intact.
    """

    def __init__(
        self,
        project: str,
        judge_model: str | art.Model = "openai/o3",
        rubric: str = DEFAULT_RUBRIC,
    ):
        self.project = project  # store for later use
        self.judge_model = judge_model
        self.rubric = rubric

    @weave.op()
    async def judge(
        self, trajectories: list[art.Trajectory], *, debug: bool = False
    ) -> list[art.Trajectory]:
        """Score every trajectory in *trajectories* and write the score to `traj.reward`."""

        if not trajectories:
            return trajectories

        for traj in trajectories:
            if len(traj.additional_histories) > 0:
                raise ValueError(
                    "Additional histories are not supported for the GroupJudge yet."
                )

        # Gather the message lists for each trajectory so we can detect any
        # common prefix messages that appear at the start of *every* trajectory.
        message_lists: list[list] = []
        for traj in trajectories:
            message_lists.append(traj.messages())

        # Determine the length of the longest common prefix shared by all trajectories.
        common_prefix_len = 0
        for i, msg in enumerate(message_lists[0]):
            if all(msg_list[i] == msg for msg_list in message_lists):
                common_prefix_len += 1
            else:
                break

        # If there is a non-empty common prefix, serialize it inside a <context>
        # tag so the judge model only sees it once, saving tokens.
        user_text = ""
        if common_prefix_len > 0:
            common_prefix_messages = message_lists[0][:common_prefix_len]
            user_text += (
                "<context>\n" + json.dumps(common_prefix_messages) + "\n</context>\n\n"
            )

        # Serialize the remainder of each rollout *without* the common prefix.
        serialized_rollouts: List[str] = []
        for idx, (traj, full_messages) in enumerate(
            zip(trajectories, message_lists), start=1
        ):
            # Preserve the original reward for later inspection.
            traj.metrics["independent_reward"] = traj.reward

            trimmed_messages = full_messages[common_prefix_len:]
            serialized_rollouts.append(
                f'<rollout id="{idx}">\n'
                + json.dumps(trimmed_messages)
                + "\n</rollout>"
            )

        user_text += "Rollouts:\n\n" + "\n\n".join(serialized_rollouts)

        judge_prompt = dedent(
            f"""
            All of the rollouts below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgement of the agent's goal.

            Grading standards:
            {self.rubric}
            """
        )

        messages = [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": user_text},
        ]

        completion_params = {}
        if isinstance(self.judge_model, art.Model):
            completion_params = self.judge_model.litellm_completion_params()
        else:
            completion_params["model"] = self.judge_model

        print("model is", self.judge_model)
        response = await acompletion(
            # **completion_params,
            model=self.judge_model,
            messages=messages,
            response_format=GroupJudgeResponse,
            caching=True,
        )

        first_choice = response.choices[0]  # type: ignore[attr-defined]

        if debug:
            raw_content = first_choice.message.content or "{}"  # type: ignore[attr-defined]

            try:
                print("\n[GroupJudge] Pretty-printed LLM choice JSON:")
                print(json.loads(raw_content))
            except json.JSONDecodeError as e:
                print(f"[GroupJudge] Could not parse choice content as JSON: {e}")
                print(f"[GroupJudge] Raw choice content: {raw_content}")

        content = first_choice.message.content or "{}"  # type: ignore[attr-defined]
        parsed = GroupJudgeResponse.model_validate_json(content)
        assert len(parsed.scores) == len(trajectories)

        for traj, score in zip(trajectories, parsed.scores):
            traj.metrics["group_judge_score"] = score.score
            traj.reward = (
                score.score
                if traj.metrics.get("failed_format_validation", 0) == 0
                else 0
            )
            traj.log(f"Judge group explanation: {score.explanation}")

        return trajectories
