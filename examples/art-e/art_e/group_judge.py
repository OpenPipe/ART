from art_e.rollout import ProjectTrajectory
from typing import List
import json
from litellm import acompletion
from textwrap import dedent
from tqdm.asyncio import tqdm
from pydantic import BaseModel, Field
from rich import print
import art
import weave

weave.init(project_name="email_agent")


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
    """LLM-based judge for groups of rollouts.

    Parameters
    ----------
    judge_model: str, default "openai/o3"
        The model that will be used to score the rollouts.
    rubric: str, default :data:`DEFAULT_RUBRIC`
        A replacement *rubric* that will overwrite the default bullet list
        under the "Grading standards:" section of :data:`DEFAULT_RUBRIC`.
        If *None*, the built-in grading standards are kept intact.
    """

    def __init__(
        self,
        project: str,
        judge_model: str | art.TrainableModel = "openai/o3",
        judge_model_use_base_model: bool = False,
        rubric: str = DEFAULT_RUBRIC,
    ):
        self.project = project  # store for later use
        self.judge_model = judge_model
        self.judge_model_use_base_model = judge_model_use_base_model
        self.rubric = rubric

    @weave.op()
    async def judge(
        self, rollouts: list[ProjectTrajectory], *, debug: bool = False
    ) -> list[ProjectTrajectory]:
        """Score every trajectory in *rollouts* and write the score to `traj.reward`."""

        if not rollouts:
            return rollouts

        for traj in rollouts:
            if len(traj.additional_histories) > 0:
                raise ValueError(
                    "Additional histories are not supported for the GroupJudge yet."
                )

        # Gather the message lists for each rollout so we can detect any
        # common prefix messages that appear at the start of *every* rollout.
        message_lists: list[list] = []
        for traj in rollouts:
            message_lists.append(traj.messages())

        # Determine the length of the longest common prefix shared by all rollouts.
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
            zip(rollouts, message_lists), start=1
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
            if self.judge_model_use_base_model:
                # When using base_model, we still need the model's inference configuration
                # (api_key, base_url) but we override the model name to use the base model
                completion_params["model"] = (
                    f"hosted_vllm/{self.judge_model.base_model}"
                )
        else:
            completion_params["model"] = self.judge_model

        print("judge completion_params", completion_params)
        response = await acompletion(
            **completion_params,
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
        assert len(parsed.scores) == len(rollouts)

        for traj, score in zip(rollouts, parsed.scores):
            traj.metrics["group_judge_score"] = score.score
            traj.reward = (
                score.score
                if traj.metrics.get("failed_format_validation", 0) == 0
                else 0
            )
            traj.log(f"Judge group explanation: {score.explanation}")

        return rollouts


if __name__ == "__main__":
    import asyncio

    async def main():
        """Run a quick smoke-test: generate one rollout per model and judge them."""
        from dotenv import load_dotenv
        import art
        from art_e.project_types import ProjectPolicyConfig
        from art_e.data.query_iterators import load_synthetic_queries
        from art_e.rollout import rollout

        load_dotenv()

        MODEL_CONFIGS = [
            "openai/gpt-4o",
            "openai/gpt-4.1",
            "openai/o4-mini",
            "openai/o3",
        ]

        # Create the four models we want to benchmark.
        models: list[art.Model] = []
        for litellm_name in MODEL_CONFIGS:
            models.append(
                art.Model(
                    name=litellm_name,
                    inference_model_name=litellm_name,
                    project="email_agent",
                    config=ProjectPolicyConfig(),
                )
            )

        # Grab the first N test-set scenarios and evaluate them one by one so we
        # can inspect the results serially.
        scenarios = load_synthetic_queries(split="test", limit=1)

        for scenario_idx, scenario in enumerate(scenarios, start=1):
            print(f"\n\n=== Scenario {scenario_idx} / {len(scenarios)} ===")

            # Generate one rollout per model for the current scenario.
            rollouts = await tqdm.gather(
                *[rollout(model, scenario) for model in models],
                desc=f"Rollouts for scenario {scenario_idx}",
            )

            print("Independent rewards (before judging):")
            for m, t in zip(models, rollouts):
                print(f"  {m.name:10s}: {t.reward:.3f}")

            art_model = art.TrainableModel[ProjectPolicyConfig](
                name="willcb/Qwen3-32B",
                base_model="willcb/Qwen3-32B",
                # inference_model_name="hosted_vllm/willcb/Qwen3-32B",
                project="email_agent",
                config=ProjectPolicyConfig(),
            )
            art_model.inference_api_key = "default"
            art_model.inference_base_url = "http://localhost:8000/v1"

            judge = GroupJudge(
                project="email_agent",
                judge_model=art_model,
                judge_model_use_base_model=True,
                # judge_model="openrouter/qwen/qwen3-14b",
                # judge_model="openai/o3",
            )
            judged_rollouts = await judge.judge(rollouts, debug=True)

            print("\nJudge-group rewards:")
            for m, t in zip(models, judged_rollouts):
                print(f"  {m.name:10s}: {t.reward:.3f}")

    asyncio.run(main())
