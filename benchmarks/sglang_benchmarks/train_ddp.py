#!/usr/bin/env python3
"""
DDP training script for Unsloth — launched via torchrun.

This script replicates the single-GPU training logic from
UnslothTrainingWorker.train_on_packed_tensors() but distributes
sequences across multiple GPUs using PyTorch DDP.

Usage (called by UnslothSGLangService._train_step_ddp):
    CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 \
        --master_port=29500 train_ddp.py --config /path/to/config.json

The config JSON contains all parameters needed for training.
Results (metrics + checkpoint path) are written to a JSON file
that the parent process reads after the subprocess exits.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_ddp")


def _patch_vllm_for_unsloth_import() -> None:
    """Make Unsloth importable even when vLLM's C extension is broken."""
    import types

    try:
        import vllm._C  # noqa: F401
        return
    except (ImportError, OSError, AttributeError):
        pass

    logger.info("vLLM C extension broken — mocking for Unsloth import")
    sys.modules["vllm._C"] = types.ModuleType("vllm._C")

    class _StubModule(types.ModuleType):
        def __getattr__(self, name: str):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            def _noop(*args, **kwargs):
                return None
            return _noop

    if "unsloth_zoo.vllm_utils" not in sys.modules:
        sys.modules["unsloth_zoo.vllm_utils"] = _StubModule("unsloth_zoo.vllm_utils")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    import torch

    # Extract config BEFORE any heavy imports
    base_model = cfg["base_model"]
    output_dir = cfg["output_dir"]
    lora_rank = cfg.get("lora_rank", 1)
    lora_alpha = cfg.get("lora_alpha", 32)
    max_seq_length = cfg.get("max_seq_length", 8192)
    learning_rate = cfg.get("learning_rate", 5e-6)
    moe_backend = cfg.get("moe_backend", "auto")
    load_in_4bit = cfg.get("load_in_4bit", False)
    last_checkpoint = cfg.get("last_checkpoint")
    packed_tensors_dir = cfg["packed_tensors_dir"]
    num_sequences = cfg["num_sequences"]
    sequence_length = cfg["sequence_length"]
    lr = cfg.get("lr") or learning_rate
    step_number = cfg["step_number"]
    results_file = cfg["results_file"]

    # Narrow CUDA_VISIBLE_DEVICES so each rank sees exactly ONE GPU.
    # Parent sets e.g. CUDA_VISIBLE_DEVICES=1,3. We pick the one GPU
    # that belongs to this local_rank, so Unsloth/transformers can only
    # load onto cuda:0 (the sole visible device).
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    parent_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if parent_gpus and parent_gpus[0]:
        os.environ["CUDA_VISIBLE_DEVICES"] = parent_gpus[local_rank]
    torch.cuda.set_device(0)  # only one GPU visible now

    _dist_env_keys = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                      "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK",
                      "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCHELASTIC_RUN_ID"]
    _saved_dist_env = {k: os.environ.pop(k) for k in _dist_env_keys if k in os.environ}

    # Patch vLLM before importing Unsloth
    if moe_backend != "auto":
        os.environ["UNSLOTH_MOE_BACKEND"] = moe_backend
    _patch_vllm_for_unsloth_import()

    from unsloth import FastLanguageModel

    # Restore distributed env vars and NOW initialize process group
    os.environ.update(_saved_dist_env)

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    # Fix unsloth_zoo bug: distributed_function() uses `dist` without
    # importing torch.distributed. Inject it into the module globals.
    import unsloth_zoo.utils
    unsloth_zoo.utils.dist = dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        logger.info(f"DDP training: world_size={world_size}, config={base_model}")
        logger.info(f"Loading model: {base_model} (rank 0)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Resume LoRA weights from previous step
    if last_checkpoint:
        adapter_file = os.path.join(last_checkpoint, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            from safetensors.torch import load_file
            lora_state = load_file(adapter_file)
            model.load_state_dict(lora_state, strict=False)
            if rank == 0:
                logger.info(f"Resumed {len(lora_state)} LoRA tensors from {last_checkpoint}")

    FastLanguageModel.for_training(model)

    # Wrap trainable parameters with DDP
    # DDP needs to wrap the model so gradients are synchronized across ranks.
    # We only wrap after for_training() so Unsloth's patches are applied first.
    device = torch.device("cuda:0")  # only one GPU visible per rank
    model = model.to(device)

    # Find trainable params and create optimizer BEFORE wrapping with DDP
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)

    optimizer = torch.optim.AdamW(
        trainable, lr=lr, betas=(0.9, 0.99), weight_decay=0.1,
    )

    # Wrap with DDP — only LoRA params have requires_grad=True.
    # static_graph=True is required because Unsloth uses gradient
    # checkpointing which causes reentrant backward passes.
    model = DDP(model, device_ids=[0], static_graph=True)

    if rank == 0:
        logger.info(f"DDP ready — {n_params:,} trainable params, {world_size} GPUs")

    # Barrier to ensure all ranks are ready
    dist.barrier()

    # Load packed tensors and split across ranks
    from art.preprocessing.pack import packed_tensors_from_dir
    from art.loss import loss_fn, shift_tensor

    packed = packed_tensors_from_dir(
        dir=packed_tensors_dir,
        num_sequences=num_sequences,
        sequence_length=sequence_length,
    )

    # Split sequences across ranks: rank i gets sequences [start:end].
    # CRITICAL: all ranks MUST call forward+backward the same number of
    # times, otherwise DDP deadlocks. We use ceil-division so every rank
    # loops `iters_per_rank` times; ranks with fewer real sequences do a
    # zero-loss forward on their last real sequence for the extra iters.
    seqs_per_rank = num_sequences // world_size
    remainder = num_sequences % world_size
    if rank < remainder:
        start_idx = rank * (seqs_per_rank + 1)
        end_idx = start_idx + seqs_per_rank + 1
    else:
        start_idx = rank * seqs_per_rank + remainder
        end_idx = start_idx + seqs_per_rank

    my_num_sequences = end_idx - start_idx
    iters_per_rank = -(-num_sequences // world_size)  # ceil division

    if rank == 0:
        logger.info(
            f"Sequence split: {num_sequences} total, "
            f"{my_num_sequences} real seqs for rank 0 [{start_idx}:{end_idx}], "
            f"{iters_per_rank} iters/rank (padded)"
        )

    # Training loop — same as UnslothTrainingWorker.train_on_packed_tensors
    model.train()
    optimizer.zero_grad()
    t0 = time.perf_counter()

    total_loss = 0.0
    n_seqs = 0
    completion_tokens = 0

    for local_iter in range(iters_per_rank):
        real_idx = start_idx + local_iter
        is_padding = real_idx >= end_idx or my_num_sequences == 0
        # For padding iters, re-use the last real sequence (or seq 0)
        data_idx = min(real_idx, max(end_idx - 1, 0)) if not is_padding else 0

        inputs = {
            key: value[data_idx:data_idx + 1].to(device)
            for key, value in packed.items()
            if isinstance(value, torch.Tensor)
        }

        tokens = inputs["tokens"]
        if not is_padding:
            completion_tokens += int(inputs["assistant_mask"].sum().item())

        attn_mask = (inputs["group_ids"] != -1).long()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=tokens,
                position_ids=inputs["input_pos"],
                attention_mask=attn_mask,
            )
            logits = outputs.logits

            labels = shift_tensor(tokens, 0)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            new_logprobs = log_probs.gather(
                dim=-1, index=labels.unsqueeze(-1),
            ).squeeze(-1)

            experimental_config = {"on_policy_correction": True}
            loss_result = loss_fn(
                inputs, new_logprobs, ref_logprobs=None, entropies=None,
                experimental_config=experimental_config,
            )

            if is_padding:
                loss = loss_result.mean_policy_loss * 0.0
            else:
                loss = loss_result.mean_policy_loss / max(my_num_sequences, 1)

        loss.backward()

        if not is_padding:
            total_loss += loss_result.mean_policy_loss.item()
            n_seqs += 1

    # Gradient clipping and optimizer step (DDP syncs gradients in backward)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm=0.1,
    )
    optimizer.step()
    optimizer.zero_grad()

    elapsed = time.perf_counter() - t0

    # Aggregate metrics across ranks
    loss_tensor = torch.tensor([total_loss], device=device)
    tokens_tensor = torch.tensor([completion_tokens], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)

    total_loss_all = loss_tensor.item()
    total_tokens_all = int(tokens_tensor.item())
    avg_loss = total_loss_all / max(num_sequences, 1)

    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    if rank == 0:
        logger.info(
            f"DDP trained: loss={avg_loss:.4f}  {total_tokens_all / elapsed:.0f} tok/s  "
            f"VRAM={gpu_mem_gb:.1f}GB/GPU  {elapsed:.2f}s  ({world_size} GPUs)"
        )

    # Rank 0 saves LoRA checkpoint and results
    if rank == 0:
        ckpt = os.path.join(output_dir, "checkpoints", f"{step_number:04d}")
        os.makedirs(ckpt, exist_ok=True)

        # Unwrap DDP to get the base model for saving
        unwrapped = model.module
        unwrapped.save_pretrained(ckpt)

        adapter = os.path.join(ckpt, "adapter_model.safetensors")
        if os.path.exists(adapter):
            mb = os.path.getsize(adapter) / 1e6
            logger.info(f"LoRA saved: {ckpt} ({mb:.1f} MB)")

        # Write results for parent process
        results = {
            "loss": avg_loss,
            "training_time_s": elapsed,
            "tokens_per_sec": total_tokens_all / elapsed,
            "gpu_memory_gb": gpu_mem_gb,
            "total_tokens": total_tokens_all,
            "batch_size": num_sequences,
            "seq_len": sequence_length,
            "ddp_world_size": world_size,
            "checkpoint": ckpt,
        }
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results written to {results_file}")

    # Wait for rank 0 to finish saving before cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
