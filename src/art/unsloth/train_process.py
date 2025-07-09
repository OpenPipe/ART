#!/usr/bin/env python3
"""
Unsloth training process that runs separately from vLLM inference.
This process handles loading the Unsloth model, running training steps,
and saving checkpoints.
"""

import argparse
import asyncio
import json
import os
import sys
import torch
from pathlib import Path
from safetensors.torch import save_file
import unsloth  # type: ignore
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import peft
from trl import GRPOConfig, GRPOTrainer
from typing import Any, cast

from art.local.pack import packed_tensors_from_dir
from art.utils.output_dirs import get_step_checkpoint_dir
from art.utils.get_model_step import get_step_from_dir


class UnslothTrainer:
    def __init__(self, base_model: str, checkpoint_dir: str, output_dir: str, config: dict[str, Any]):
        self.base_model = base_model
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.config = config
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        # Initialize Unsloth model
        init_args = self.config.get("init_args", {})
        # Override model_name to use checkpoint_dir if it exists
        if os.path.isdir(self.checkpoint_dir) and self.checkpoint_dir != self.base_model:
            init_args["model_name"] = self.checkpoint_dir
        else:
            init_args["model_name"] = self.base_model
            
        self.model, self.tokenizer = cast(
            tuple[Any, PreTrainedTokenizerBase],
            unsloth.FastLanguageModel.from_pretrained(**init_args),
        )
        
        # Initialize PEFT model
        self.peft_model = cast(
            peft.peft_model.PeftModelForCausalLM,
            unsloth.FastLanguageModel.get_peft_model(
                self.model, **self.config.get("peft_args", {})
            ),
        )
        
        # Initialize trainer with dummy dataset
        data = {"prompt": ""}
        self.trainer = GRPOTrainer(
            model=self.peft_model,  # type: ignore
            reward_funcs=[],
            args=GRPOConfig(**self.config.get("trainer_args", {})),
            train_dataset=Dataset.from_list([data for _ in range(10_000_000)]),
            processing_class=self.tokenizer,
        )
        
    def train_batch(self, batch_info: dict[str, Any]) -> None:
        """Train on a single batch and output metrics."""
        # Load packed tensors
        packed_tensors = packed_tensors_from_dir(**batch_info["disk_packed_tensors"])
        config = batch_info["config"]
        _config = batch_info["_config"]
        
        # Get number of samples
        num_samples = packed_tensors["tokens"].shape[0]
        
        # Train on each sample
        for offset in range(num_samples):
            # Prepare inputs
            inputs = {
                k: v[offset : offset + 1]
                for k, v in packed_tensors.items()
                if isinstance(v, torch.Tensor)
            }
            
            # Override _prepare_inputs to return our inputs
            self.trainer._prepare_inputs = lambda *_, **__: inputs
            
            # Run one training step
            self.trainer.optimizer.zero_grad()
            
            # Get model outputs
            outputs = self.trainer.model(**inputs)
            
            # Calculate loss
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            if config.get("lr", 1e-4) > 0:
                # Scale learning rate
                for param_group in self.trainer.optimizer.param_groups:
                    param_group['lr'] = config.get("lr", 1e-4)
                
                self.trainer.optimizer.step()
            
            # Output metrics
            metrics = {
                "loss": loss.item(),
                "num_gradient_steps": num_samples,
            }
            
            # Add any additional metrics from config
            if "reward" in inputs:
                metrics["reward"] = inputs["reward"].mean().item()
                
            # Output as JSON for the parent process to parse
            print(json.dumps(metrics), flush=True)
        
        # Save checkpoint after batch
        self._save_checkpoint()
        
    def _save_checkpoint(self) -> None:
        """Save the LoRA checkpoint and export weights for vLLM."""
        # Determine next step
        next_step = get_step_from_dir(self.output_dir) + 1
        checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save the LoRA adapter
        self.trainer.save_model(checkpoint_dir)
        
        # Export weights for vLLM to load
        weights_path = "/dev/shm/weights.safetensors"
        
        # Get LoRA weights
        lora_state_dict = {}
        for name, param in self.peft_model.named_parameters():
            if param.requires_grad:  # Only LoRA parameters
                lora_state_dict[name] = param.detach().cpu()
        
        # Save weights
        save_file(lora_state_dict, weights_path)
        
        print(f"Saved checkpoint to {checkpoint_dir}", file=sys.stderr, flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    # Parse config
    config = json.loads(args.config)
    
    # Initialize trainer
    trainer = UnslothTrainer(
        base_model=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        config=config,
    )
    
    # Read batches from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        try:
            batch_info = json.loads(line)
            trainer.train_batch(batch_info)
        except Exception as e:
            print(f"Error processing batch: {e}", file=sys.stderr, flush=True)
            raise


if __name__ == "__main__":
    asyncio.run(main())