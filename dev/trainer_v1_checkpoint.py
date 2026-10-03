"""Create a native checkpoint fixture for the remote public API canary."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from trainer_v1_support import build_rank, load_checkpoint, nccl_group

from art.trainer_rank import validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    load_dotenv(".env")
    for axis in ("TENSOR_MODEL", "CONTEXT", "DATA", "PIPELINE_MODEL"):
        os.environ[f"ART_MEGATRON_{axis}_PARALLEL_SIZE"] = "1"
    with nccl_group():
        rank = build_rank(args.model, eval_mode=False)
        checkpoint = load_checkpoint(rank, args.model)
        rank.save_checkpoint(str(args.output), checkpoint)
        assert validate_checkpoint(args.output) is not None
        print(f"native checkpoint saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
