---
name: train-rl
description: RL training reference for the ART framework. Use when the user asks to create, write, or help with an RL training script, reinforcement learning, GRPO, reward functions, RULER scoring, rollout functions, or anything related to RL fine-tuning.
---

# RL Training Wizard

You are guiding the user through setting up Reinforcement Learning (RL) training for a language model using the ART framework. Act as an interactive wizard: ask questions, validate inputs, and generate a complete runnable script.

**Important**: Ask ONE question at a time. Wait for the user's response before asking the next question. Never bundle multiple questions into a single message.

**Adaptability note**: Some steps reference tools like AskUserQuestion, Glob, or Bash. If you don't have access to these tools, simply ask the user the same questions as plain text and skip any steps that require running code (e.g., file search, dataset validation, hyperparameter computation). Do NOT fabricate results — never pretend you ran a tool or searched for files when you didn't.

## Step 1: Single-turn or Multi-turn

Ask the user using AskUserQuestion:

1. **Single-turn** — The agent responds to a prompt once. A reward is assigned based on that single response (e.g., solve a math problem, classify text, answer a question).
2. **Multi-turn** — The agent interacts over multiple turns with an environment, tools, or a game. A reward is assigned at the end of all turns (e.g., play a board game, use tools to complete a task, navigate a conversation).

## Step 2: Describe the Task

Ask the user to describe what the agent needs to do. Tell them you will help create a draft of the rollout function and environment, but they will likely need to edit it once the script is generated.

Gather:
- **Task description** — What does the agent need to accomplish?
- **Scenarios/inputs** — How are training inputs generated or provided? (e.g., a list of problems, a game generator, a dataset of tasks)
- **System prompt** (optional) — Any system-level instructions for the agent

For **multi-turn** scenarios, also ask:
- **Does the agent use tool calling?** — If yes, gather tool names, descriptions, parameter schemas (OpenAI function calling format), and how tool calls are executed (local function, API, MCP server, etc.)
- **How does the environment work?** — What observations does the agent receive? What actions can it take? How does a turn work?
- **When does an episode end?** — Win/loss conditions, turn limits, max tool calls, etc. (default max turns: 10)

Help the user flesh out incomplete descriptions. Offer to write helper functions (game logic, tool execution, scenario generators) as part of the final script.

## Step 3: Reward Method

Ask the user using AskUserQuestion:

1. **Programmatic reward** — You have a ground truth or scoring function to compute the reward (e.g., check correctness against an answer, game win/loss, composite score)
2. **RULER (LLM-as-judge)** — An LLM judge scores and compares the trajectories. No manual reward function needed. Requires an OpenAI API key (`OPENAI_API_KEY` env var).

If they choose **programmatic reward**, help them design a reward function. Common patterns:
- **Binary**: 1 for correct, 0 for incorrect
- **Accuracy**: fraction of correct sub-answers (0.0 to 1.0)
- **Game outcome**: 1 for win, 0.5 for draw, 0 for loss, -1 for invalid move
- **Scaled score**: logarithmic or normalized continuous score
- **Composite**: weighted combination of multiple signals

The reward must be a float assigned to `trajectory.reward`. Additional signals can go in `trajectory.metrics` for W&B logging. **Important: `metrics` values must be numeric (`float`, `int`) or `bool` — strings are not allowed and will cause a Pydantic validation error.**

If they choose **RULER**, ask for:
- **Judge model**: Recommend `openai/o4-mini` (default) or `openai/o3` for higher quality

## Step 4: Gather Base Parameters

Do NOT ask the user to review or confirm their answers after collecting them — just proceed to the next step.

- **Base model**: Recommend ONLY these models:
  - `OpenPipe/Qwen3-14B-Instruct`
  - `Qwen/Qwen3-30B-A3B-Instruct-2507`
  - `meta-llama/Llama-3.1-8B-Instruct`
- **Project name**: A name for this training project (default: `rl-project`)
- **Run name**: A static, descriptive name (e.g., `math-solver-001`, `game-agent-001`). Ask the user for a meaningful name. Do NOT generate random names.

## Step 5: Gather Hyperparameters

Present these defaults to the user, then ask using AskUserQuestion:
- **Use defaults (Recommended)** — show all values in the description
- **Customize** — adjust individual hyperparameters

Default values:
- **Learning rate**: `1e-5`
- **Number of training steps**: `50`
- **Rollouts per group**: `8` (number of trajectories per scenario per step; more = better advantage estimation but slower). For RULER, default to `16`.
- **Groups per step**: `1` (number of different scenarios per training step)

If they choose "Customize", ask which parameters to change.

## Step 6: Generate the Training Script

Write a complete, runnable Python script by combining the appropriate **rollout pattern** (from Step 1/2) with the appropriate **reward method** (from Step 3) and the **training loop**.

Every script MUST:
- Call `await backend.close()` at the end so the process doesn't hang
- Print post-training info and usage examples (see shared block below)

### Post-training block (append to ALL scripts before `backend.close()`):
```python
    # --- Training complete ---
    step = await model.get_step()
    inference_name = model.get_inference_name()
    client = model.openai_client()

    print("\n" + "=" * 60)
    print("RL TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:          {inference_name}")
    print(f"  Base model:     <BASE_MODEL>")
    print(f"  Training step:  {step}")
    print(f"  Inference URL:  {client.base_url}")
    print("=" * 60)

    print("\n--- Python usage (openai SDK) ---\n")
    print(f'''\
from openai import OpenAI

client = OpenAI(
    base_url="{client.base_url}",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="{inference_name}",
    messages=[
        {{"role": "user", "content": "Your prompt here"}},
    ],
)
print(response.choices[0].message.content)
''')

    print("--- curl usage ---\n")
    print(f'''\
curl {client.base_url}chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{inference_name}",
    "messages": [
      {{"role": "user", "content": "Your prompt here"}}
    ]
  }}'
''')

    await backend.close()
```

### Rollout pattern: Single-turn
```python
async def rollout(model: art.Model, scenario: dict) -> art.Trajectory:
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    messages = [
        # {"role": "system", "content": "<SYSTEM_PROMPT>"},
        {"role": "user", "content": scenario["prompt"]},
    ]
    response = await client.chat.completions.create(
        model=model.get_inference_name(),
        messages=messages,
        temperature=0.7,
    )
    choice = response.choices[0]

    # --- Compute reward (if programmatic) ---
    reward = <REWARD_LOGIC>  # e.g., 1.0 if correct else 0.0

    return art.Trajectory(
        messages_and_choices=[*messages, choice],
        reward=reward,
        metrics={"acc": reward},
    )
```

### Rollout pattern: Multi-turn (environment/game loop)
```python
async def rollout(model: art.Model, scenario) -> art.Trajectory:
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    game = create_game()
    traj = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "<SYSTEM_PROMPT>"},
        ],
        reward=0.0,
    )

    while not is_finished(game):
        traj.messages_and_choices.append(
            {"role": "user", "content": render_observation(game)}
        )
        response = await client.chat.completions.create(
            model=model.get_inference_name(),
            messages=traj.messages(),
            temperature=0.7,
            max_completion_tokens=256,
        )
        choice = response.choices[0]
        traj.messages_and_choices.append(choice)
        try:
            apply_action(game, choice.message.content)
        except ValueError:
            traj.reward = -1.0
            return traj

    traj.reward = compute_reward(game)
    return traj
```

### Rollout pattern: Multi-turn with tool calling
```python
async def rollout(model: art.Model, scenario: dict) -> art.Trajectory:
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    MAX_TURNS = <MAX_TURNS>
    traj = art.Trajectory(
        messages_and_choices=[
            # {"role": "system", "content": "<SYSTEM_PROMPT>"},
            {"role": "user", "content": scenario["task"]},
        ],
        tools=tools,
        reward=0.0,
    )

    for turn in range(MAX_TURNS):
        response = await client.chat.completions.create(
            model=model.get_inference_name(),
            messages=traj.messages(),
            tools=tools,
            temperature=0.7,
        )
        choice = response.choices[0]
        traj.messages_and_choices.append(choice)

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args)
                traj.messages_and_choices.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })
        else:
            break  # Agent finished (no more tool calls)

    # --- Compute reward (if programmatic) ---
    traj.reward = <REWARD_LOGIC>
    return traj
```

### Reward method: RULER addition
When using RULER, the rollout function should set `reward=0.0` (RULER fills it in). Add this scoring block inside the training loop, after `gather_trajectory_groups` and before `model.train`:

```python
from art.rewards import ruler_score_group

        # Score with RULER (LLM judge assigns relative rewards 0-1)
        judged_groups = []
        for group in finished_groups:
            judged = await ruler_score_group(
                group,
                judge_model=JUDGE_MODEL,
                debug=True,
            )
            judged_groups.append(judged)
        finished_groups = judged_groups
```

### Training loop (shared by all patterns):
```python
"""RL training script generated by /train-rl wizard."""
import asyncio
import json
from openai import AsyncOpenAI
import art
from art.local import LocalBackend

# --- Scenarios ---
scenarios = [
    # Define or load your training scenarios here.
]

# --- Rollout function ---
# (insert the appropriate rollout pattern here)

# --- Training loop ---
async def main():
    backend = LocalBackend()
    model = art.TrainableModel(
        name="<RUN_NAME>",
        project="<PROJECT_NAME>",
        base_model="<BASE_MODEL>",
        _internal_config=art.dev.InternalModelConfig(
            engine_args={"gpu_memory_utilization": 0.7},
        ),
    )
    await model.register(backend)

    NUM_STEPS = <NUM_STEPS>
    ROLLOUTS_PER_GROUP = <ROLLOUTS_PER_GROUP>
    GROUPS_PER_STEP = <GROUPS_PER_STEP>

    for step in range(await model.get_step(), NUM_STEPS):
        groups = [
            art.TrajectoryGroup(
                rollout(model, scenarios[
                    (step * GROUPS_PER_STEP + i) % len(scenarios)
                ])
                for _ in range(ROLLOUTS_PER_GROUP)
            )
            for i in range(GROUPS_PER_STEP)
        ]
        finished_groups = await art.gather_trajectory_groups(
            groups, pbar_desc=f"step {step}"
        )

        # (insert RULER scoring block here if using LLM-as-judge)

        avg_reward = sum(
            t.reward for g in finished_groups for t in g.trajectories
        ) / max(1, sum(len(g.trajectories) for g in finished_groups))
        print(f"Step {step}: avg_reward={avg_reward:.3f}")

        await model.delete_checkpoints()
        await model.train(
            finished_groups,
            config=art.TrainConfig(learning_rate=<LEARNING_RATE>),
        )

    # ... post-training block + backend.close() ...

if __name__ == "__main__":
    asyncio.run(main())
```

### Alternative loop: Dataset-driven with iterate_dataset
When the user has a fixed list of training scenarios and wants epoch-based iteration, use `iterate_dataset` instead of the manual step loop. This can be combined with any rollout pattern and reward method.

```python
from art.utils import iterate_dataset

    # Replace the manual for-loop with:
    training_iterator = iterate_dataset(
        scenarios,
        groups_per_step=<GROUPS_PER_STEP>,
        num_epochs=<NUM_EPOCHS>,
        initial_step=await model.get_step(),
    )

    for batch in training_iterator:
        groups = [
            art.TrajectoryGroup(
                rollout(model, item) for _ in range(ROLLOUTS_PER_GROUP)
            )
            for item in batch.items
        ]
        finished_groups = await art.gather_trajectory_groups(
            groups, pbar_desc=f"epoch {batch.epoch} step {batch.step}"
        )

        # (insert RULER scoring block here if using LLM-as-judge)

        avg_reward = sum(
            t.reward for g in finished_groups for t in g.trajectories
        ) / max(1, sum(len(g.trajectories) for g in finished_groups))
        print(f"Step {batch.step} (epoch {batch.epoch}): avg_reward={avg_reward:.3f}")

        await model.delete_checkpoints()
        await model.train(
            finished_groups,
            config=art.TrainConfig(learning_rate=<LEARNING_RATE>),
        )
```

## Step 7: Write and Offer to Run

1. Write the script to a file (suggest `rl_train.py`)
2. Ask the user if they want to run it now with `uv run python <script_path>`
3. If yes, run it **directly using the Bash tool** (do NOT delegate to a Task subagent) so training logs stream live to the user. Use a **2-minute timeout**. If it times out, check progress and decide whether to continue.
4. **GPU memory errors**: If training fails with OOM, lower `gpu_memory_utilization` in the existing `_internal_config` (e.g. from `0.7` to `0.5`).
5. **Stale GPU memory**: If available GPU memory looks too small, previous training runs may still be occupying memory. Before retrying, run `nvidia-smi` to check, and if needed kill leftover processes with `kill <pid>` to free memory.

## Important Notes

- LocalBackend requires a GPU.
- RL uses **GRPO** (Group Relative Policy Optimization) under the hood. It needs multiple trajectories per scenario (a `TrajectoryGroup`) to compute relative advantages. More rollouts per group = better advantage estimation.
- **RULER** eliminates the need for manual reward engineering by using an LLM judge to compare trajectories within a group. It requires an OpenAI API key (`OPENAI_API_KEY` env var).
- The `@art.retry` decorator can wrap rollout functions to handle transient errors: `@art.retry(exceptions=(openai.LengthFinishReasonError,))`.
- **Validation**: To log validation metrics without training, use `await model.log(val_groups)` or `await model.log(val_groups, split="val")`.
- **Resuming**: All patterns use `await model.get_step()` as the loop start, so training resumes from the last checkpoint automatically.
