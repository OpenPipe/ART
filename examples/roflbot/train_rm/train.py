import logging
import os
import torch
import json
import argparse
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from peft.tuners.lora import LoraConfig
from peft import get_peft_model, PeftModel
from transformers.tokenization_utils import PreTrainedTokenizer
import wandb
from training_helpers import (
    compute_metrics,
    run_final_inference_and_report_metrics,
    MandT,
    get_dataset,
)
import s3fs
from pydantic import BaseModel


class RunConfig(BaseModel):
    run_name: str
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_length: int = 4096
    train_size: int = 100000
    val_size: int = 500
    base_model: str = "Qwen/Qwen2.5-0.5B"
    accelerator: str = "H100-SXM:1"


def train(config: RunConfig):
    from liger_kernel.transformers import _apply_liger_kernel_to_instance

    output_dir = f"./models/{config.run_name}"

    print(f"Initializing WandB run for {config.run_name}")
    wandb.init(project="roflbot_rm", name=config.run_name)
    wandb.config.update(config.model_dump())

    logging.info("Loading tokenizer and model...")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        truncation=True,
        padding=True,
        max_length=config.max_length,
    )

    base_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=1,  # Regression task
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    _apply_liger_kernel_to_instance(model=base_model)
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    logging.info("Configuring LoRA...")
    model: PeftModel = get_peft_model(
        base_model,
        LoraConfig(
            task_type="SEQ_CLS",
            r=8,
            lora_alpha=16,
            lora_dropout=0,
        ),
    )  # type: ignore
    del base_model

    logging.info("Preparing datasets...")
    # Request the desired total size
    requested_total_size = config.train_size + config.val_size
    train_jokes = get_dataset(
        "train",
        tokenizer,
        config.max_length,
        requested_total_size,  # Ask for the sum initially
    )

    total_jokes_loaded = len(train_jokes)
    print(f"Loaded {total_jokes_loaded} jokes from the 'train' split.")

    # Determine actual validation and training sizes based on loaded data
    # Ensure at least 1 training example if possible, unless dataset is truly tiny
    actual_val_size = min(config.val_size, max(0, total_jokes_loaded - 1))
    actual_train_size = total_jokes_loaded - actual_val_size

    if actual_train_size < config.train_size or actual_val_size < config.val_size:
        logging.warning(
            f"Requested train_size={config.train_size}, val_size={config.val_size}. "
            f"Dataset only has {total_jokes_loaded} examples. "
            f"Using actual_train_size={actual_train_size}, actual_val_size={actual_val_size}."
        )

    test_jokes = get_dataset("test", tokenizer, config.max_length)
    print(f"Test jokes: {len(test_jokes)}")

    # Split train_jokes into train and validation sets using actual sizes
    val_jokes = train_jokes.select(range(actual_val_size))
    train_jokes = train_jokes.select(range(actual_val_size, total_jokes_loaded))

    print(f"Train jokes after split: {len(train_jokes)}")
    print(f"Val jokes after split: {len(val_jokes)}")
    print(f"Test jokes: {len(test_jokes)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=0,
        eval_strategy="steps",
        eval_steps=0.05,
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        report_to="wandb",
        no_cuda=False,
        bf16=True,
        warmup_ratio=0.1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_jokes,
        eval_dataset=val_jokes,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    logging.info("Starting model training...")
    trainer.train()

    logging.info("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logging.info("Running final inference and reporting metrics...")
    metrics = run_final_inference_and_report_metrics(
        MandT(model, tokenizer), test_jokes, output_dir
    )

    s3 = s3fs.S3FileSystem()

    logging.info(
        f"Uploading model to S3 at path s3://{os.getenv('REMOTE_BUCKET')}/roflbot_rm/models/{config.run_name}"
    )
    s3.put(
        output_dir,
        f"s3://{os.getenv('REMOTE_BUCKET')}/roflbot_rm/models/{config.run_name}",
        recursive=True,
        maxdepth=1,
    )

    logging.info("Model training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_json", help="JSON string serialization of the RunConfig"
    )
    args = parser.parse_args()

    config_dict = json.loads(args.config_json)
    config = RunConfig(**config_dict)
    train(config)
