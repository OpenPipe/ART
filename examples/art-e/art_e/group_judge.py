from art_e.rollout import ProjectTrajectory
from art_e.project_types import TrainingConfig
from typing import List, Literal
import json
from litellm import acompletion
from textwrap import dedent
from tqdm.asyncio import tqdm
from pydantic import BaseModel, Field
from rich import print


class Issue(BaseModel):
    label: str = Field(description="A short label for the issue.")
    explanation: str = Field(
        description="A human-readable, actionable explanation of the issue."
    )
    severity: Literal["minor", "major", "fatal"] = Field(
        description="The severity of the issue: 'minor', 'major', or 'fatal'."
    )


class RolloutScore(BaseModel):
    rollout_id: str = Field(description="The id of the rollout being scored.")
    explanation: str = Field(
        description="A short explanation of why you gave this score."
    )
    score: float = Field(description="A score between 0 and 1.")
    issues: List[str] = Field(
        description="The list of labels for each issue identified for this rollout, including any new ones."
    )


class JudgeGroupResponse(BaseModel):
    new_issues: List[Issue] = Field(
        description="Any new issues identified on the rollouts in this group. Do not include issues that already exist in the list of existing issues."
    )
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
        judge_model: str = "openai/o3",
        rubric: str = DEFAULT_RUBRIC,
        initial_issues: List[Issue] = [
            Issue(
                label="looping",
                explanation="The assistant repeats itself unnnecessarily but is able to recover.",
                severity="minor",
            ),
            Issue(
                label="fatal_looping",
                explanation="The assistant began repeating itself and is unable to recover.",
                severity="fatal",
            ),
        ],
    ):
        self.project = project  # store for later use
        self.judge_model = judge_model
        self.rubric = rubric
        self.all_issues = initial_issues

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

        # First, gather the message lists for each rollout so we can detect any
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
        common_prefix_messages: list = []
        if common_prefix_len > 0:
            common_prefix_messages = message_lists[0][:common_prefix_len]
            user_text += (
                "<context>\n" + json.dumps(common_prefix_messages) + "\n</context>\n\n"
            )

        # Now serialize the remainder of each rollout *without* the common prefix.
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

        # if debug:
        #     print("\n[GroupJudge] Rollout metrics:")
        #     for idx, traj in enumerate(rollouts, start=1):
        #         print(f"\nRollout {idx} metrics:")
        #         print(traj.metrics)

        user_text += "Rollouts:\n\n" + "\n\n".join(serialized_rollouts)

        judge_prompt = dedent(
            f"""
            All of the rollouts below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgement of the agent's goal.

            Grading standards:
            {self.rubric}
            
            To aid in downstream debugging, you should also identify and label any issues you see in the rollouts. This will allow us to track the rates of specific issues and look for patterns. Here are the issues that have already been identified. If while reviewing the rollouts you see a new issue, you may return it and it will be added to the list.

            Existing issues:
            {json.dumps([issue.model_dump() for issue in self.all_issues], indent=2)}
            """
        )

        messages = [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": user_text},
        ]

        response = await acompletion(
            model=self.judge_model,
            messages=messages,
            response_format=JudgeGroupResponse,
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
        parsed = JudgeGroupResponse.model_validate_json(content)
        assert len(parsed.scores) == len(rollouts)

        # Merge any newly discovered issues into our running list, avoiding duplicates.
        if parsed.new_issues:
            existing_labels = {fm.label for fm in self.all_issues}
            for fm in parsed.new_issues:
                if fm.label not in existing_labels:
                    self.all_issues.append(fm)
                    existing_labels.add(fm.label)

        for traj, score in zip(rollouts, parsed.scores):
            traj.metrics["group_judge_score"] = score.score
            traj.reward = (
                score.score
                if traj.metrics.get("failed_format_validation", 0) == 0
                else 0
            )
            traj.log(f"Judge group explanation: {score.explanation}")
            # Record whether each predefined issue was detected in this rollout.
            # We add a metric for every known issue (including any newly discovered
            # ones) so downstream analysis can easily aggregate issue rates even
            # when the issue did not occur.
            for issue in self.all_issues:
                metric_key = f"issues/{issue.severity}/{issue.label}"
                traj.metrics[metric_key] = issue.label in score.issues

            # ------------------------------------------------------------------
            # Propagate judge-group information back onto the rollout's Weave
            # Call so that it shows up inside the trace for easy inspection.
            # We look for the call id that `rollout` stored under
            # `traj.metadata['weave_call_id']` and, if present, attach the
            # score/explanation/issue labels via `add_feedback` (legal even
            # after the Call has finished).
            # ------------------------------------------------------------------

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
                    project="email_agent",
                    config=ProjectPolicyConfig(
                        litellm_model_name=litellm_name,
                    ),
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

            # Judge the group of rollouts.
            judge = GroupJudge(project="email_agent", judge_model="openai/o3")
            judged_rollouts = await judge.judge(rollouts, debug=True)

            print("\nJudge-group rewards:")
            for m, t in zip(models, judged_rollouts):
                print(f"  {m.name:10s}: {t.reward:.3f}")

    asyncio.run(main())
