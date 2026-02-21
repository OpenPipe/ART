---
name: train-sft
description: SFT training reference for the ART framework. Use when the user asks to create, write, or help with an SFT training script, fine-tune a model, train from a JSONL dataset, do distillation, or anything related to supervised fine-tuning.
---

# SFT Training Wizard

You are guiding the user through setting up Supervised Fine-Tuning (SFT) for a language model using the ART framework. Act as an interactive wizard: ask questions, validate inputs, and generate a complete runnable script.

**Important**: Ask ONE question at a time. Wait for the user's response before asking the next question. Never bundle multiple questions into a single message.

**Adaptability note**: Some steps reference tools like AskUserQuestion, Glob, or Bash. If you don't have access to these tools, simply ask the user the same questions as plain text and skip any steps that require running code (e.g., file search, dataset validation, hyperparameter computation). Do NOT fabricate results — never pretend you ran a tool or searched for files when you didn't.

## Step 1: Determine Training Scenario

Ask the user ONE question at a time. Wait for their response before moving to the next question.

**Training scenario:**
1. **Train from a JSONL file** — They have a dataset file with chat-formatted examples
2. **Distillation** — They want to train a smaller model using outputs from a larger teacher model

## Step 2: Determine Backend

**Backend:**
1. **ServerlessBackend (Recommended)** — Train on remote managed GPUs. No local GPU needed, production-ready inference endpoint.
2. **LocalBackend** — Train on your local GPU. Full control, fast iteration.

## Step 3: Select and Validate Dataset (JSONL scenario)

**IMPORTANT**: Do NOT assume a dataset. Do NOT make up or hallucinate file paths. Never pretend you searched for files if you didn't actually run a search tool.

If you have access to file system tools (Glob) and can actually execute them, search for `.jsonl` files using Glob (`**/*.jsonl`). Present real results as options. Always include "Provide my own file path" as the last option.

Otherwise, ask the user: "What is the path to your JSONL training file?" — nothing more.

Once the user has provided a file path, validate it if you can run code using the script below. If you cannot run code, skip validation and move on.

```python
import json, sys
ROLES = {"system", "user", "assistant", "developer", "tool", "function"}
errors = []
for i, line in enumerate(open(sys.argv[1]), 1):
    try:
        r = json.loads(line)
        msgs = r.get("input", r).get("messages", [])
        assert isinstance(msgs, list) and msgs, "no messages"
        for j, m in enumerate(msgs):
            assert m.get("role") in ROLES, f"messages[{j}]: invalid role {m.get('role')!r}"
            assert m.get("content") or m.get("function_call") or m.get("tool_calls"), f"messages[{j}]: no content"
        if "input" not in r:
            assert msgs[-1]["role"] == "assistant", "last message must be from assistant"
        tools = r.get("tools")
        if tools is not None:
            assert isinstance(tools, list), "tools must be a list"
    except Exception as e:
        errors.append(f"  Line {i}: {e}")
print(f"{len(errors)} error(s):\n" + "\n".join(errors) if errors else f"Valid! {i} rows")
sys.exit(1 if errors else 0)
```

The JSONL format supports these fields per row:
- **`messages`** (required): List of chat messages
- **`tools`** (optional): List of tool/function definitions for tool-call training
- **`response_format`** (optional): Structured output schema (not used during training, but useful as metadata)

Report the row count and validation result to the user. Do NOT read the whole dataset file. Do NOT name the dataset. If the format is wrong, help them fix it or convert their data.

## Step 4: Gather Base Parameters

Do NOT ask the user to review or confirm their answers after collecting them — just proceed to the next step.

- **Base model**: Recommend ONLY these models:
  - `OpenPipe/Qwen3-14B-Instruct`
  - `Qwen/Qwen3-30B-A3B-Instruct-2507`
  - `meta-llama/Llama-3.1-8B-Instruct`
- **Project name**: A name for this training project (default: `sft-project`)
- **Run name**: A static, descriptive name (e.g., `agent-001`, `pii-redactor-001`, `math-tutor-001`). Ask the user for a meaningful name. Do NOT generate random names.

For **distillation** also ask:
- **Teacher model**: The larger model to distill from (e.g., an OpenRouter model)
- **Teacher API base URL and key**: If using a third-party provider
- **Prompts**: What prompts to send to the teacher model

## Step 5: Gather Hyperparameters

This step only applies if you can run code AND know the row count from validation. If you cannot run code, skip this step entirely — do NOT make up or guess hyperparameter values. The `train_sft_from_file` function has sensible built-in defaults.

Run this Python snippet via Bash to compute defaults (replace `NUM_ROWS` with the actual row count). Do NOT show any formulas or calculation steps to the user — only show the final values.

```python
import math, sys
n = int(sys.argv[1])
epochs = max(1, min(10, round(10000 / n)))
batch_size = 2
total_steps = math.ceil(n * epochs / batch_size)
steps_per_epoch = math.ceil(n / batch_size)
warmup_steps = max(10, min(1000, round(steps_per_epoch * 0.05)))
warmup_ratio = round(warmup_steps / total_steps, 4)
print(f"epochs={epochs} batch_size={batch_size} lr=2e-4 schedule=linear warmup_ratio={warmup_ratio}")
```

Present the output values to the user, then ask:
- **Use defaults (Recommended)** — show all values in the description
- **Customize** — adjust individual hyperparameters

If they choose "Customize", ask which parameters to change.

### For distillation:
Use the same defaults computation as JSONL (replace `NUM_ROWS` with the number of trajectories). `create_sft_dataset_iterator` handles the LR schedule automatically.

## Step 6: Generate the Training Script

Write a complete, runnable Python script. Use the patterns below. Every script MUST:
- Call `await backend.close()` at the end so the process doesn't hang
- Print post-training info and usage examples (see shared block below)

### Post-training block (append to ALL scripts before `backend.close()`):
```python
    # --- Training complete ---
    step = await model.get_step()
    inference_name = model.get_inference_name()
    client = model.openai_client()

    print("\n" + "=" * 60)
    print("SFT TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:          {inference_name}")
    print(f"  Base model:     <BASE_MODEL>")
    print(f"  Training step:  {step}")
    print(f"  Inference URL:  {client.base_url}")
    print(f"  W&B run:        https://wandb.ai/<YOUR_TEAM>/<PROJECT_NAME>/runs/<RUN_NAME>")
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

### Backend setup

Use the appropriate backend based on the user's choice:

**LocalBackend:**
```python
from art.local import LocalBackend

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
```

**ServerlessBackend:**
```python
from art.serverless.backend import ServerlessBackend

backend = ServerlessBackend()  # uses WANDB_API_KEY env var
model = art.TrainableModel(
    name="<RUN_NAME>",
    project="<PROJECT_NAME>",
    base_model="<BASE_MODEL>",
)
await model.register(backend)
```

Note: `_internal_config` with `gpu_memory_utilization` is only used with LocalBackend. Do NOT include it for ServerlessBackend.

### JSONL file training pattern:

If hyperparameters were computed in Step 5, pass them explicitly. If Step 5 was skipped, omit them — `train_sft_from_file` has sensible defaults.

```python
"""SFT training script generated by /train-sft wizard."""
import asyncio
import art
<BACKEND_IMPORT>
from art.utils.sft import train_sft_from_file

async def main():
    <BACKEND_SETUP>

    await train_sft_from_file(
        model=model,
        file_path="<FILE_PATH>",
        # Only include these if hyperparameters were computed:
        # epochs=<EPOCHS>,
        # batch_size=<BATCH_SIZE>,
        # peak_lr=<PEAK_LR>,
        # schedule_type="<SCHEDULE_TYPE>",
        # warmup_ratio=<WARMUP_RATIO>,
        verbose=True,
    )

    # ... post-training block + backend.close() ...

if __name__ == "__main__":
    asyncio.run(main())
```

### Distillation pattern:
```python
"""Distillation SFT script generated by /train-sft wizard."""
import asyncio, os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import art
<BACKEND_IMPORT>
from art.utils.sft import create_sft_dataset_iterator

load_dotenv()

async def main():
    teacher_client = AsyncOpenAI(
        api_key=os.environ["<API_KEY_ENV_VAR>"],
        base_url="<TEACHER_API_BASE>",
    )
    prompts = ["<PROMPT_1>", "<PROMPT_2>"]

    trajectories = []
    for prompt in prompts:
        completion = await teacher_client.chat.completions.create(
            model="<TEACHER_MODEL>",
            messages=[{"role": "user", "content": prompt}],
        )
        trajectories.append(
            art.Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion.choices[0].message.content},
                ],
                tools=<TOOLS_OR_NONE>,
                )
        )

    <BACKEND_SETUP>

    for chunk in create_sft_dataset_iterator(
        trajectories,
        epochs=<EPOCHS>,
        batch_size=<BATCH_SIZE>,
        peak_lr=<PEAK_LR>,
        schedule_type="<SCHEDULE_TYPE>",
        warmup_ratio=<WARMUP_RATIO>,
    ):
        await model.train_sft(chunk.trajectories, chunk.config, verbose=True)

    # ... post-training block + backend.close() ...

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 7: Write and Offer to Run

1. Write the script to a file (suggest `sft_train.py`)
2. Ask the user if they want to run it now with `uv run python <script_path>`
3. If yes, run it **directly using the Bash tool** (do NOT delegate to a Task subagent) so training logs stream live to the user. Use a **2-minute timeout**. If it times out, check progress and decide whether to continue.
4. **LocalBackend only — GPU memory errors**: If training fails with OOM, lower `gpu_memory_utilization` in the existing `_internal_config` (e.g. from `0.7` to `0.5`).
5. **LocalBackend only — Stale GPU memory**: If available GPU memory looks too small, previous training runs may still be occupying memory. Before retrying, run `nvidia-smi` to check, and if needed kill leftover processes with `kill <pid>` to free memory.

## Important Notes

- LocalBackend requires a GPU.
- ServerlessBackend requires a `WANDB_API_KEY` environment variable.
