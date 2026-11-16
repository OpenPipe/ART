---
name: train-rl
description: RL training reference for the ART framework. Use when the user asks to create, write, or help with an RL training script, reinforcement learning, GRPO, reward functions, RULER scoring, rollout functions, or anything related to RL fine-tuning.
---

# ART RL Workflow

Use this skill when the user wants an RL script or help adapting an existing ART agent for RL.

Keep the process simple:
1. Inspect the existing environment or agent first.
2. Ask one question at a time until the blocking decisions are resolved.
3. Generate a runnable script that reuses the real rollout/tool logic instead of approximating it.

This skill is an interactive wizard. Do not write the script immediately.

Rules:
- Ask one question at a time.
- Wait for the user's answer before asking the next question.
- Use the repo to make recommendations, but still ask the user to confirm every required choice.
- Do not skip required questions just because the code suggests a likely answer.
- Do not generate the final script until all required questions below have been answered.

## Blocking Decisions

You must resolve these before writing the final script:
1. Is the environment replayable?
2. Is this multi-turn RL or single-turn training on static examples?
3. Is reward programmatic, RULER, or custom?
4. Is the backend `ServerlessBackend` or `LocalBackend`?

## Required Questions

You must collect answers for all of these, one at a time, before generating code:
1. Replayability confirmation
2. What behavior the agent should optimize
3. Where training scenarios come from
4. If multi-turn: how turns work and when the episode ends
5. Reward choice
6. If RULER: judge model
7. Training split or source
8. Validation split or source
9. Base model
10. Project name
11. Run name
12. Backend
13. Hyperparameters: use defaults or customize
14. Iteration mode: fixed-dataset epoch loop or manual/open-ended loop

If the repo already makes an answer likely, present that as a recommendation and ask the user to confirm or correct it. That still counts as a question and still requires a user response.

## 1. Replayability

Inspect the repo before asking.

- Recommend `multi-turn RL` when episodes can be recreated and rolled out repeatedly from the same initial state.
- Recommend `single-turn static training` when the task depends on live humans, mutable production systems, or other unreproducible state.

If replayability is clear, say so and ask for confirmation. Example:

`This looks replayable because each episode starts from fixed local state and the tools only read from it, so I recommend multi-turn RL. Please confirm there is no hidden live dependency.`

If it is not clear, ask whether the task has a replayable environment or only logged/static scenarios.

## 2. Task Shape

Ask only for the details needed to implement the rollout, but do not skip the required task questions.

Always gather:
- What the agent is supposed to accomplish.
- Where training scenarios come from.

For multi-turn tasks, also gather:
- What the observation is on each turn.
- What actions or tools the agent can take.
- When the episode ends.

When adapting an existing agent:
- Preserve its prompt and tool behavior by default.
- Reuse its real tool execution path and message schema.
- Prefer storing structured terminal outputs such as `final_answer` directly on the trajectory when useful.
- If the existing environment already has a natural typed terminal object or evaluation artifact, keep it in the trajectory or structured logs instead of reducing everything to free-form text.

## 3. Reward Choice

Start from RULER as the default.

Use this rule:
- Choose `programmatic reward` only when correctness is robustly checkable with code.
- Choose `RULER` for open-text answers or tool-use behavior where exact matching is brittle.
- Choose `custom` only when the task genuinely mixes multiple reward sources.

Explain RULER briefly once:
- `RULER` is an LLM judge that compares trajectories within a group and scores which ones are better.

If the user chooses programmatic reward:
- Put the score in `trajectory.reward`.
- Keep extra signals in `trajectory.metrics`.
- Do not invent weak heuristic rewards.
- If the environment already exposes robust auxiliary signals such as correctness, source overlap, pass/fail, completion rate, or tool-error rate, keep logging them even when they are not the main reward.

If the user chooses RULER:
- Prefer `ruler_score_group(...)` with the default rubric.
- Recommend `OPENAI_API_KEY` validation at startup.
- Recommend `openai/gpt-5.4` as the default judge model.
- Ask which judge model to use before generating the script.

## 4. Data Splits and Validation

For fixed datasets:
- Prefer a held-out validation split.
- Prefer capped periodic validation during training.
- Do not run a full held-out pass at step `0` unless the user asks for it.

For validation:
- Log validation groups at a concrete training step.
- Make sure the metric used for checkpoint cleanup is actually logged.
- With the default `await model.delete_checkpoints()`, validation must produce `val/reward`.

## 5. Base Parameters

Ask for, explicitly and separately:
- Base model
- Project name
- Run name
- Backend

Do not present a single "recommended starting point" model by default. Offer all allowed base models:
- `Qwen/Qwen3.8-27B`
- `Qwen/Qwen3.6-35B-A3B`
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-4-26B-A4B-it`
- `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B`

Environment requirements:
- `ServerlessBackend`: require `WANDB_API_KEY`
- `RULER`: require `OPENAI_API_KEY`

## 6. Hyperparameters

Ask whether to use these starting defaults or customize them:
- Learning rate: `1e-5`
- Rollouts per group: `4`
- Groups per step: `2`

Iteration defaults:
- For fixed datasets, prefer `iterate_dataset(..., initial_step=await model.get_step())`.
- For non-fixed/open-ended generation, use a manual step loop.

## Implementation Guardrails

These are the main ART-specific rules that matter in practice:

- Reuse the real environment/agent entrypoints when they already exist.
- Prefer building `Trajectory.messages_and_choices` directly for multi-turn tool use.
- Use `backend.train(model, trajectory_groups, ...)` plus `await model.log(...)`.
- Call `await backend.close()` before exit.
- Catch recoverable inference or tool errors after the trajectory has started and return a partial trajectory with numeric or bool metrics.
- Pass `art.TrajectoryGroup(...)` awaitables directly into `art.gather_trajectory_groups(...)`. Do not await them early.
- If using RULER rescoring, prefer `after_each=lambda group: ruler_score_group(...)`.
- Preserve `group.exceptions` if you rebuild groups after rollout.
- Default `max_exceptions` to scale with the active batch size, typically `args.rollouts_per_group * len(batch.items)` for training and the analogous validation batch size. Do not hard-code a small fixed value unless the user explicitly wants that.
- For fixed datasets, make resume behavior explicit with `initial_step=await model.get_step()`.
- Delete old checkpoints by default after successful training and validation logging unless the user wants to keep all of them.

## Minimal Training Pattern

Use this as the default pattern for fixed datasets with RULER:

```python
from art.rewards import ruler_score_group
from art.utils.iterate_dataset import iterate_dataset


async def rollout(model: TrainableModel, scenario: Scenario) -> art.Trajectory:
    ...


for batch in iterate_dataset(
    train_scenarios,
    groups_per_step=args.groups_per_step,
    num_epochs=args.num_epochs,
    initial_step=await model.get_step(),
):
    train_groups = await art.gather_trajectory_groups(
        [
            art.TrajectoryGroup(
                (rollout(model, scenario) for _ in range(args.rollouts_per_group)),
                metadata={"scenario_id": scenario.id},
            )
            for scenario in batch.items
        ],
        after_each=lambda group: ruler_score_group(
            group,
            judge_model=args.judge_model,
        ),
        max_exceptions=args.rollouts_per_group * len(batch.items),
    )

    train_result = await backend.train(
        model,
        train_groups,
        learning_rate=args.learning_rate,
    )

    await model.log(
        train_groups,
        metrics=train_result.metrics,
        step=train_result.step,
        split="train",
    )

    if should_validate(train_result.step):
        val_groups = await art.gather_trajectory_groups(
            [
                art.TrajectoryGroup(
                    (rollout(model, scenario) for _ in range(args.rollouts_per_group)),
                    metadata={"scenario_id": scenario.id},
                )
                for scenario in validation_scenarios
            ],
            after_each=lambda group: ruler_score_group(
                group,
                judge_model=args.judge_model,
            ),
            max_exceptions=args.rollouts_per_group * len(validation_scenarios),
        )
        await model.log(
            val_groups,
            metrics={"reward": ...},
            step=train_result.step,
            split="val",
        )
        await model.delete_checkpoints()
```

## Final Script Requirements

Every generated script should:
- Validate required environment variables early.
- Follow the repo's existing env-loading convention.
- Use the selected backend.
- Print the final inference model name and a short usage example.
- Close the backend cleanly.

If you fail to find enough information from the repo, say what is missing and ask the next single blocking question. Do not fabricate environment behavior, reward logic, or dataset structure.
