import asyncio
import os
from typing import Tuple

from datasets import Dataset
from dotenv import load_dotenv
import numpy as np
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_callback import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from utils import cache, prompt_for_title, pull_data, score_title
from vllm import SamplingParams
import wandb

from art.utils import limit_concurrency

load_dotenv()

# --- Hyperparameters (Inlined from model11.yaml and script defaults) ---
RUN_NAME = "reference_implementation_grpo"
LORA_RANK = 8
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
GPU_MEMORY_UTILIZATION = 0.6  # Using the second value from the yaml
LEARNING_RATE = 5e-6
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
LR_SCHEDULER_TYPE = "constant"  # Changed from cosine for simplicity, matching yaml
OPTIM = "paged_adamw_8bit"
LOGGING_STEPS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 6
GRADIENT_ACCUMULATION_STEPS = 1
NUM_GENERATIONS = 6  # Used by GRPOTrainer for generating responses during training
NUM_EPOCHS = 1
MAX_COMPLETION_LENGTH = 100
MAX_PROMPT_LENGTH = 8192 - MAX_COMPLETION_LENGTH
SAVE_STEPS = 250
MAX_GRAD_NORM = 0.1
OUTPUT_DIR = "outputs_reference"
TRAINING_DATASET_SIZE = 5000
BETA = 0.0  # GRPO Beta parameter from yaml
VAL_SET_SIZE = 100  # Number of validation samples
EVAL_STEPS = 50  # Evaluate every N training steps


def filter_on_length(
    data: Dataset, max_length: int, tokenizer: PreTrainedTokenizer
) -> Dataset:
    """Filters dataset entries based on tokenized prompt length."""

    def check_length(x):
        return (
            len(
                tokenizer.apply_chat_template(
                    x["prompt"], tokenize=True, add_generation_prompt=True
                )
            )
            <= max_length
        )

    len_before = len(data)
    data = data.filter(check_length)
    print(
        f"Filtered dataset: {len_before} -> {len(data)} samples based on max_length {max_length}"
    )
    return data


@cache.cache()  # Keep caching for efficiency
async def load_title_data(
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_items: int = 10,
    min_score: int = 20,
    max_length: int = 8192,
) -> Dataset:
    """Loads and prepares data specifically for the title generation task."""
    print(
        f"Loading data for split '{split}' (max_items={max_items}, min_score={min_score})..."
    )
    data = pull_data(split=split, max_items=max_items, min_score=min_score)
    data = data.map(
        lambda x: {
            "prompt": prompt_for_title(x["scraped_body"]),
            "row": x,  # Keep original row data for reward calculation
        }
    )
    print(f"Loaded {len(data)} raw samples for split '{split}'.")
    return filter_on_length(data, max_length, tokenizer)


# --- Reward Function ---


@limit_concurrency(10)  # Limit concurrency for external calls
async def score_title_async(row_data: dict) -> float:
    return await score_title(row_data)


# 1. Load Model and Tokenizer
print(f"Loading base model: {BASE_MODEL}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
)
print("Model and tokenizer loaded.")

# 2. Add LoRA Adapters
print(f"Adding LoRA adapters (rank={LORA_RANK}) to modules: {TARGET_MODULES}")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",  # type: ignore
    random_state=3407,
)
print("LoRA adapters added.")


async def calculate_rewards(
    prompts,
    completions,
    rows,
) -> list[Tuple[float, dict[str, float]]]:
    responses = [completion[0]["content"] for completion in completions]
    updated_rows = [{**r, "title": response} for r, response in zip(rows, responses)]

    @limit_concurrency(10)
    async def score_title_async(r):
        return await score_title(r, "rm")

    def check_if_titles_match_bodies(bodies, titles):
        inputs = []
        for body, title in zip(bodies, titles):
            inputs.append(
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are a moderator for Hacker News. You are given the body of an article, as well as a proposed title. You are to determine whether the title makes any claims that are not substantiated by the article body. If there are any unsubstantiated claims, you should return False. Otherwise, you should return True. Only return False or True, no other text.",
                        },
                        {
                            "role": "user",
                            "content": f"<article>{body}</article>\n<proposed_title>{title}</proposed_title>",
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        outputs = model.fast_generate(  # type: ignore
            inputs,
            sampling_params=SamplingParams(max_tokens=2),
        )
        outputs = [o.outputs[0].text for o in outputs]
        for output in outputs:
            if output not in ["True", "False"]:
                print(
                    f"Warning: Invalid output from check_if_title_matches_scraped_body: {output}"
                )
        return [1 if output.lower()[0] == "t" else 0 for output in outputs]

    # Kick this off first so the RM can be scoring while we're doing the matching check locally
    rm_task_coros = [score_title_async(r) for r in updated_rows]

    # Run the matching check locally
    matching_scores = check_if_titles_match_bodies(
        [r["scraped_body"] for r in updated_rows],
        [r["title"] for r in updated_rows],
    )

    rm_scores = await asyncio.gather(*rm_task_coros)

    rewards = []
    for r, rm_score, title_matches in zip(updated_rows, rm_scores, matching_scores):
        if title_matches == 0:
            score = 0
        else:
            score = rm_score
        rewards.append(
            (
                score,
                {
                    "length": len(r["title"]),
                    "matches": title_matches,
                    "rm": rm_score,
                },
            )
        )
    return rewards


def reward_func(
    prompts: list[list[dict]], completions: list[list[dict]], **kwargs
) -> list[float]:
    """
    Adapter function to fit the async reward logic into GRPOTrainer's expected sync signature.
    It extracts only the scalar reward value.
    """
    if "row" not in kwargs:
        raise ValueError("Missing 'row' data in kwargs, cannot calculate rewards.")

    # Run the async reward calculation
    rewards_and_metrics = asyncio.run(
        calculate_rewards(prompts, completions, rows=kwargs["row"])
    )
    # Return only the scalar reward scores
    return [r[0] for r in rewards_and_metrics]


# --- Validation Callback ---


class ValidationCallback(TrainerCallback):
    def __init__(
        self,
        val_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_completion_length: int,
        eval_steps: int,
    ):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.max_completion_length = max_completion_length
        self.eval_steps = eval_steps
        # Store raw prompts as strings for fast_generate
        self.val_prompts_raw = [
            self.tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True
            )
            for p in self.val_dataset["prompt"]
        ]
        print(f"ValidationCallback initialized with {len(self.val_dataset)} samples.")
        # Removed self.val_prompts_tokenized

    def on_step_end(
        self, args, state, control, model: FastLanguageModel, **kwargs
    ):  # Added type hint for model
        if state.global_step % self.eval_steps != 0:
            return control

        # You might think that `model.fast_generate` on a LoRA model would run generation on the actual LoRA model, but
        # you would be wrong, it runs generation on the base model! But if we save the LoRA weights and then load them,
        # we can get the correct behavior.
        model.save_lora("val_model_lora")  # type: ignore

        outputs = model.fast_generate(  # type: ignore
            self.val_prompts_raw,
            sampling_params=SamplingParams(max_tokens=self.max_completion_length),
            use_tqdm=False,
            lora_request=model.load_lora("val_model_lora"),  # type: ignore
        )
        outputs = [o.outputs[0].text for o in outputs]

        print("Validation generation complete. Calculating rewards...")
        # Prepare completions in the format expected by the reward function
        completions_for_reward = [
            [{"role": "assistant", "content": o}] for o in outputs
        ]

        # Calculate rewards asynchronously
        scores_and_metrics = asyncio.run(
            calculate_rewards(
                self.val_dataset["prompt"],  # Original prompts (list of dicts)
                completions_for_reward,
                self.val_dataset["row"],  # Original row data
            )
        )
        scores = [s[0] for s in scores_and_metrics]

        # Log aggregate scores and metrics to W&B
        if scores:  # Check if scores list is not empty
            wandb.log(
                {
                    "val/reward/mean": np.mean(scores),
                    "val/reward/p5": np.percentile(scores, 5),
                    "val/reward/median": np.percentile(scores, 50),
                    "val/reward/p95": np.percentile(scores, 95),
                    "val/reward/std_dev": np.std(scores),
                },
                step=state.global_step,
            )

            # Aggregate and log metrics
            all_metrics: dict[str, list[float]] = {}
            for _, metrics_dict in scores_and_metrics:
                for k, v in metrics_dict.items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)

            for k, v_list in all_metrics.items():
                if v_list:  # Check if list for metric is not empty
                    wandb.log(
                        {
                            f"val/metrics/{k}/mean": np.mean(v_list),
                            f"val/metrics/{k}/p5": np.percentile(v_list, 5),
                            f"val/metrics/{k}/median": np.percentile(v_list, 50),
                            f"val/metrics/{k}/p95": np.percentile(v_list, 95),
                            f"val/metrics/{k}/std_dev": np.std(v_list),
                        },
                        step=state.global_step,
                    )
        else:
            print("Warning: No validation scores generated.")

        # *** Removed table logging logic ***

        print(f"Validation complete for step {state.global_step}.")
        # Ensure model is back in train mode
        return control


print("Starting GRPO Reference Implementation Script")
wandb.init(project="hn_title_generation", name=RUN_NAME)

# 3. Load Datasets (Train and Validation)
print("Loading and preparing datasets...")
# We need the tokenizer to filter data correctly
train_dataset = asyncio.run(
    load_title_data(
        tokenizer=tokenizer,
        split="train",
        max_items=TRAINING_DATASET_SIZE,
        max_length=MAX_PROMPT_LENGTH,
    )
)
print(f"Training dataset loaded with {len(train_dataset)} samples.")

# Load validation dataset
val_dataset = asyncio.run(
    load_title_data(
        tokenizer=tokenizer,
        split="val",  # Use 'val' split
        max_items=VAL_SET_SIZE,  # Use validation size hyperparameter
        max_length=MAX_PROMPT_LENGTH,  # Use same max length
    )
)
print(f"Validation dataset loaded with {len(val_dataset)} samples.")

# 4. Configure GRPO Trainer
print("Configuring GRPOTrainer...")
training_args = GRPOConfig(
    # --- VLLM specific ---
    use_vllm=True,  # Use VLLM for generation during training
    # --- Training Hyperparameters ---
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    optim=OPTIM,
    adam_beta1=ADAM_BETA1,
    adam_beta2=ADAM_BETA2,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,  # Keep last 2 checkpoints
    report_to="wandb",
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    gradient_checkpointing=True,  # Already enabled via get_peft_model 'unsloth' setting
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Recommended for Unsloth
    # --- GRPO Specific ---
    beta=BETA,
    num_generations=NUM_GENERATIONS,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    remove_unused_columns=False,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=reward_func,
    # No eval_dataset needed here, handled by callback
)
print("GRPOTrainer configured.")

# 5. Add Validation Callback
print("Adding validation callback...")
validation_callback = ValidationCallback(
    val_dataset=val_dataset,
    tokenizer=tokenizer,
    max_completion_length=MAX_COMPLETION_LENGTH,
    eval_steps=EVAL_STEPS,
)
trainer.add_callback(validation_callback)
print("Validation callback added.")

# 6. Train
print("Starting training...")
trainer.train()
print("Training finished.")


# Ensure necessary directories exist before starting
os.makedirs(OUTPUT_DIR, exist_ok=True)
