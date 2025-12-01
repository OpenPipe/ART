import argparse
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer as NeMoAutoTokenizer
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingRandomBatchSampler,
)
from nemo_aligner.models.nlp.gpt.gpt_sft_model import GPTSFTModel
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.utils.train_script_utils import (
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_peft,
    init_using_ptl,
    resolve_and_create_trainer,
)
from nemo_aligner.data.nlp.builders import build_train_valid_test_datasets
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.utils import logging
import os
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeMo-Aligner LoRA SFT")
    p.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--dataset", required=True, help="Path to JSONL with 'prompt'/'completion' or instruction format")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=10)
    return p.parse_args()


def ensure_dirs(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def build_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    # Minimal trainer/model/data cfg compatible with MegatronTrainerBuilder and GPTSFTModel
    trainer_cfg: Dict[str, Any] = {
        "accelerator": "gpu",
        "devices": args.gpus,
        "num_nodes": args.nodes,
        "precision": "bf16-mixed",
        "max_steps": args.steps,
        "limit_val_batches": 0,
        "enable_progress_bar": False,
        "log_every_n_steps": 1,
    }

    # Small default architecture for smoke tests; for large models the weights/config must be aligned explicitly
    model_cfg: Dict[str, Any] = {
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        # Use FSDP to avoid Apex requirement in default DDP strategy
        "fsdp": True,
        "fsdp_sharding_strategy": "full",
        "fsdp_cpu_offload": False,
        "fsdp_use_orig_params": False,
        "encoder_seq_length": args.seq_length,
        "max_position_embeddings": args.seq_length,
        "precision": 16,
        "megatron_amp_O2": False,
        "gradient_as_bucket_view": True,
        # Tiny backbone to validate the end-to-end loop; override per target model later
        "num_layers": 2,
        "hidden_size": 512,
        "ffn_hidden_size": 2048,
        "num_attention_heads": 8,
        "activation": "gelu",
        "vocab_size": 32000,
        # Tokenizer configured to HF model id
        "tokenizer": {
            "library": "huggingface",
            "type": "AutoTokenizer",
            "pretrained_model_name": args.model,
            "use_fast": True,
            "trust_remote_code": True,
        },
        # Optimizer (simple Adam for smoke test)
        "optim": {
            "name": "adam",
            "lr": 1e-4,
        },
        # PEFT LoRA
        "peft": {
            "peft_scheme": "lora",
            "lora_tuning": {
                "adapter_dim": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": ["attention", "mlp"],
            },
        },
    }

    # Data config used by builders/GPTSFTDataset
    data_cfg: Dict[str, Any] = {
        "data_impl": "json",
        "splits_string": "9999,1,0",
        "train_valid_test_num_samples": [9999, 1, 0],
        "skip_warmup": True,
        "train_ds": {
            "global_batch_size": args.global_batch_size,
            "micro_batch_size": args.micro_batch_size,
        },
        "validation_ds": {
            "global_batch_size": args.global_batch_size,
            "micro_batch_size": args.micro_batch_size,
        },
        # SFT prompt template using 'prompt' and 'completion' keys from our sample jsonl
        "prompt_template": "{prompt}{completion}",
        "label_key": "completion",
        "max_seq_length": args.seq_length,
        "seed": 42,
        "pad_to_max_length": False,
        "truncation_field": "prompt",
        "truncation_method": "right",
    }

    # Supervised trainer loop config
    sup_cfg: Dict[str, Any] = {
        "max_epochs": 1,
        "max_steps": args.steps,
        "val_check_interval": args.steps,  # skip val for smoke
        "limit_val_batches": 0,
        "save_interval": args.steps,
        "gradient_clip_val": 1.0,
    }

    exp_manager: Dict[str, Any] = {
        "exp_dir": args.output,
        "name": "sft",
        "create_checkpoint_callback": True,
        "checkpoint_callback_params": {"save_top_k": 1, "every_n_train_steps": args.steps},
    }

    return {"trainer": trainer_cfg, "model": model_cfg, "data": data_cfg, "sup": sup_cfg, "exp_manager": exp_manager}


def main() -> None:
    args = parse_args()
    ensure_dirs(args.output)

    cfg_dict = build_cfg(args)
    cfg = OmegaConf.create(cfg_dict)

    # Build trainer first
    # Monkeypatch NeMo to bypass Apex hard requirement in strategy checks
    try:
        import nemo.collections.nlp.parts.nlp_overrides as _no
        _no.HAVE_APEX = True  # rely on Torch fallbacks already present
    except Exception:
        pass
    trainer = resolve_and_create_trainer(cfg, pop_trainer_key="enable_progress_bar")

    # Instantiate model with model cfg only
    # Ensure NVIDIA container version env is set to avoid optimization check crash
    if os.environ.get('NVIDIA_PYTORCH_VERSION') is None:
        os.environ['NVIDIA_PYTORCH_VERSION'] = '24.10'
    model = GPTSFTModel(DictConfig(cfg.model), trainer=trainer)

    # Initialize distributed and PTL internals
    init_distributed(trainer, model, use_te=False)

    # Build tokenizer directly for dataset
    tokenizer = NeMoAutoTokenizer(pretrained_model_name=args.model, use_fast=True, trust_remote_code=True)

    # Build datasets from JSONL
    train_ds, val_ds, _ = build_train_valid_test_datasets(
        cls=GPTSFTDataset,
        cfg=cfg,
        data_prefix=OmegaConf.create({"train": args.dataset, "validation": args.dataset, "test": args.dataset}),
        data_impl=cfg.data.data_impl,
        splits_string=cfg.data.splits_string,
        train_valid_test_num_samples=cfg.data.train_valid_test_num_samples,
        seq_length=cfg.model.encoder_seq_length,
        seed=cfg.data.seed,
        tokenizer=tokenizer,
    )

    # Build batch samplers
    from nemo_aligner.utils import parallel_state

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    train_sampler = MegatronPretrainingRandomBatchSampler(
        total_samples=len(train_ds),
        consumed_samples=0,
        micro_batch_size=cfg.model.micro_batch_size,
        global_batch_size=cfg.model.global_batch_size,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
        drop_last=True,
        seed=cfg.data.seed,
    )
    val_sampler = MegatronPretrainingRandomBatchSampler(
        total_samples=len(val_ds),
        consumed_samples=0,
        micro_batch_size=cfg.model.micro_batch_size,
        global_batch_size=cfg.model.global_batch_size,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
        drop_last=True,
        seed=cfg.data.seed,
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=train_ds.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=val_ds.collate_fn)

    # Initialize per PTL and PEFT
    init_using_ptl(trainer, model, train_loader, train_ds)
    init_peft(model, DictConfig(cfg.model))

    # Optimizer and scheduler from model
    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(model)

    # Simple logger wrapper and checkpointing
    ckpt_cb = add_custom_checkpoint_callback(trainer, model)
    class _NoopLogger:
        def log_metrics(self, *args, **kwargs):
            return
        def finalize(self):
            return
    logger = _NoopLogger()

    # Minimal run timer stub
    class _RunTimer:
        def start_time(self):
            return
        def is_finished(self):
            return False

    sup = SupervisedTrainer(
        cfg=DictConfig(cfg.sup),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=val_loader,
        logger=logger,
        ckpt_callback=ckpt_cb,
        run_timer=_RunTimer(),
        run_init_validation=False,
    )

    logging.info("Starting supervised fine-tuning (smoke run)...")
    sup.fit()
    logging.info("Finished.")


if __name__ == "__main__":
    main()


