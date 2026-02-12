import asyncio
from datetime import datetime
import os
from typing import Any, Dict, Iterable, List

from datasets import Dataset
from dotenv import load_dotenv
import openai
from openai.types.chat import ChatCompletionMessageParam
from openpipe import AsyncOpenPipe
from transformers.models.auto.tokenization_auto import AutoTokenizer
from utils import cache, prompt_for_title, pull_data, score_title

import art
from art.local import LocalBackend
from art.utils import iterate_dataset, limit_concurrency

load_dotenv()

MODEL_NAME = "001"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_COMPLETION_LENGTH = 100
MAX_PROMPT_LENGTH = 8192 - MAX_COMPLETION_LENGTH
LEARNING_RATE = 1.2e-5
GROUPS_PER_STEP = 1
EVAL_STEPS = 50
VAL_SET_SIZE = 100
TRAINING_DATASET_SIZE = 5000
PROJECT = "hn_title_generation"
NUM_EPOCHS = 1
NUM_GENERATIONS = 6


# --- Data Loading ---
def filter_on_length(data: Dataset, max_length: int, tokenizer_name: str) -> Dataset:
    """Filters dataset based on tokenized prompt length."""
    print(
        f"Filtering dataset for max prompt length: {max_length} using tokenizer: {tokenizer_name}"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def check_length(x):
        # Ensure 'prompt' is a list of dicts
        if not isinstance(x.get("prompt"), list):
            print(f"Warning: Skipping row with invalid prompt format: {x}")
            return False
        try:
            tokenized_len = len(
                tokenizer.apply_chat_template(
                    x["prompt"], tokenize=True, add_generation_prompt=True
                )
            )
            return tokenized_len <= max_length
        except Exception as e:
            print(
                f"Warning: Error tokenizing prompt, skipping row. Error: {e}, Prompt: {x['prompt']}"
            )
            return False

    len_before = len(data)
    data = data.filter(check_length)
    len_after = len(data)
    print(f"Samples before length filtering: {len_before}, samples after: {len_after}")
    if len_after == 0 and len_before > 0:
        print(
            "Warning: All samples were filtered out. Check MAX_PROMPT_LENGTH and tokenizer."
        )
    elif len_after < len_before * 0.5:
        print(
            f"Warning: More than 50% of samples filtered out ({len_before - len_after} samples)."
        )
    return data


@cache.cache()
async def load_data(
    split: str = "train",
    max_items: int = 10,
    min_score: int = 20,
    max_length: int = 8192,
    tokenizer_name: str = BASE_MODEL,
) -> Dataset:
    """Loads, preprocesses, and filters the dataset."""
    print(
        f"Loading data for split: {split}, max_items: {max_items}, tokenizer: {tokenizer_name}"
    )
    data = pull_data(split=split, max_items=max_items, min_score=min_score)
    if not data:
        raise ValueError(f"No data loaded for split {split}. Check pull_data function.")

    # Ensure 'scraped_body' exists and is text
    def check_scraped_body(x):
        body = x.get("scraped_body")
        return isinstance(body, str) and len(body.strip()) > 0

    data = data.filter(check_scraped_body)
    if not data:
        raise ValueError(
            f"No data remaining after filtering for valid 'scraped_body' in split {split}."
        )

    data = data.map(
        lambda x: {
            "prompt": prompt_for_title(
                x["scraped_body"]
            ),  # Creates the list of messages
            "row": x,  # Keep original row data
        }
    )
    return filter_on_length(data, max_length, tokenizer_name)


# --- Rollout Function ---
@limit_concurrency(10)
async def call_score_title(row_with_title: Dict[str, Any]) -> float:
    """Async wrapper for scoring."""
    return await score_title(row_with_title, "rm")


async def check_title_matches_body(
    client: openai.AsyncOpenAI, body: str, title: str
) -> int:
    system_prompt = "You are a moderator for Hacker News. You are given the body of an article, as well as a proposed title. You are to determine whether the title makes any claims that are not substantiated by the article body. If there are any unsubstantiated claims, you should return False. Otherwise, you should return True. Only return False or True, no other text."
    user_prompt = f"<article>{body}</article>\n<proposed_title>{title}</proposed_title>"
    messages: Iterable[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = await client.chat.completions.create(
            model=BASE_MODEL,  # use the base model, not the policy model being trained for this.
            messages=messages,
            max_tokens=5,  # Should only need 1 token for True/False
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if content:
            # Be robust to variations like "True.", " False", etc.
            content_cleaned = content.strip().lower()
            if content_cleaned.startswith("true"):
                return 1
            elif content_cleaned.startswith("false"):
                return 0
            else:
                print(
                    f"Warning: Unexpected validation response: '{content}'. Defaulting to mismatch (0)."
                )
                return 0
        else:
            print("Warning: Empty validation response. Defaulting to mismatch (0).")
            return 0
    except Exception as e:
        print(
            f"Error during title validation API call: {e}. Defaulting to mismatch (0)."
        )
        return 0


async def rollout(
    client: openai.AsyncOpenAI,
    op_client: AsyncOpenPipe,
    model_name: str,
    prompt: Iterable[ChatCompletionMessageParam],
    row: Dict[str, Any],
    global_step: int,
    epoch: int,
) -> art.Trajectory:
    """Generates a title, validates it, scores it, and returns a trajectory."""
    metrics = {}
    requested_at = datetime.now()
    # 1. Generate Title
    chat_completion = await client.chat.completions.create(
        messages=prompt,
        model=model_name,
        max_tokens=MAX_COMPLETION_LENGTH,
        temperature=1,
        logprobs=True,
    )
    received_at = datetime.now()
    choice = chat_completion.choices[0]
    generated_title = choice.message.content
    if not generated_title:
        print("Warning: Empty title generated.")
        generated_title = ""  # Assign empty string if None or empty

    metrics["length"] = len(generated_title)

    # 2. Validate Title against Body using the LLM itself
    title_matches = await check_title_matches_body(
        client, row["scraped_body"], generated_title
    )
    metrics["matches"] = title_matches

    # 3. Score Title using external RM
    row_with_title = {**row, "title": generated_title}
    rm_score = await asyncio.wait_for(call_score_title(row_with_title), timeout=30)
    metrics["rm"] = rm_score

    # 4. Calculate Final Reward
    # If the title doesn't match the body (makes unsubstantiated claims), reward is 0
    final_reward = 0.0 if title_matches == 0 else rm_score

    # Ensure messages_and_choices includes the actual response choice
    messages_and_choices = [*prompt, choice]

    await op_client.report(
        requested_at=requested_at.timestamp(),
        received_at=received_at.timestamp(),
        req_payload={
            "model": model_name,
            "messages": prompt,
            "metadata": {
                "type": "art_rollout",
                "split": row["split"],
                "step": global_step,
                "epoch": epoch,
                "dataset_id": row["id"],
                **metrics,
            },
        },
        resp_payload=chat_completion,
        status_code=200,
    )

    trajectory = art.Trajectory(
        messages_and_choices=messages_and_choices,
        reward=final_reward,
        metrics=metrics,
    )

    return trajectory


# --- Main Training Loop ---
async def main():
    # Initialize ART Backend and Model
    backend = LocalBackend()
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT,
        base_model=BASE_MODEL,
        _internal_config=art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(
                gpu_memory_utilization=0.75,
            ),
            peft_args=art.dev.PeftArgs(
                lora_alpha=8,
            ),
            trainer_args=art.dev.TrainerArgs(
                max_grad_norm=0.1,
            ),
        ),
    )
    await model.register(backend)
    op_client = AsyncOpenPipe(api_key=os.getenv("OPENPIPE_API_KEY"))

    # Load Data
    print("Loading training data...")
    train_dataset = await load_data(
        split="train",
        max_items=TRAINING_DATASET_SIZE,
        max_length=MAX_PROMPT_LENGTH,
        tokenizer_name=BASE_MODEL,
    )
    print("Loading validation data...")
    val_dataset = await load_data(
        split="val",
        max_items=VAL_SET_SIZE,
        max_length=MAX_PROMPT_LENGTH,
        tokenizer_name=BASE_MODEL,
    )

    if not train_dataset or not val_dataset:
        raise ValueError("Failed to load datasets. Exiting.")

    val_data_list: List[Dict[str, Any]] = list(val_dataset)  # type: ignore
    train_data_list: List[Dict[str, Any]] = list(train_dataset)  # type: ignore

    print(f"Training data size: {len(train_data_list)}")
    print(f"Validation data size: {len(val_data_list)}")

    # Get OpenAI Client for the ART Model
    openai_client = model.openai_client()

    # Training Loop
    start_step = await model.get_step()
    print(f"Starting training from global step {start_step}")

    data_iterator = iterate_dataset(
        dataset=train_data_list,
        groups_per_step=GROUPS_PER_STEP,
        num_epochs=NUM_EPOCHS,
        initial_step=start_step,
        use_tqdm=True,
    )

    for batch in data_iterator:
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(
                        openai_client,
                        op_client,
                        MODEL_NAME,
                        bi["prompt"],
                        bi["row"],
                        batch.step,
                        batch.epoch,
                    )
                    for _ in range(NUM_GENERATIONS)
                )
                for bi in batch.items
            )
        )

        valid_train_groups = []
        for group in train_groups:
            valid_group = [traj for traj in group if isinstance(traj, art.Trajectory)]
            if len(valid_group) > 1:
                valid_train_groups.append(valid_group)

        if not valid_train_groups:
            print(
                f"Warning: No valid trajectories generated for step {batch.step}. Skipping tune step."
            )
            continue

        result = await backend.train(
            model, valid_train_groups, learning_rate=LEARNING_RATE
        )
        await model.log(
            valid_train_groups, metrics=result.metrics, step=result.step, split="train"
        )

        if batch.step > 0 and batch.step % EVAL_STEPS == 0:
            print(f"\n--- Evaluating at Step {batch.step} ---")

            print(f"Running validation rollouts on {len(val_data_list)} samples...")
            val_trajectories = await art.gather_trajectories(
                (
                    rollout(
                        openai_client,
                        op_client,
                        MODEL_NAME,
                        item["prompt"],
                        item["row"],
                        batch.step,
                        batch.epoch,
                    )
                    for item in val_data_list
                ),
                pbar_desc="val",
            )

            await model.log(val_trajectories)
            await model.delete_checkpoints()

    print("Training finished.")


if __name__ == "__main__":
    asyncio.run(main())
