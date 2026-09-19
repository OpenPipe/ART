"""Create a native checkpoint fixture for the remote public API canary."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    load_dotenv(".env")
    for axis in ("TENSOR_MODEL", "CONTEXT", "DATA", "PIPELINE_MODEL"):
        os.environ[f"ART_MEGATRON_{axis}_PARALLEL_SIZE"] = "1"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        from trainer_rank_support import load_random_checkpoints

        from art.megatron.train import build_training_runtime
        from art.trainer_rank import TrainerRank, validate_checkpoint

        torch.manual_seed(90217)
        runtime = build_training_runtime(model_identifier=args.model, print_env=False)
        rank = TrainerRank(runtime)
        (checkpoint,) = load_random_checkpoints(
            runtime, rank, 1, base_model=args.model, lora_rank=2
        )
        rank.save_checkpoint(str(args.output), checkpoint)
        assert validate_checkpoint(args.output) is not None
        print(f"native checkpoint saved: {args.output}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
