"""Real SFT loss/gradient parity across packed rows and independent examples.

Run with pytest on one GPU, or torchrun this file for CP/DP/PP qualification.
ART_SFT_TEST_MODEL optionally selects a pretrained checkpoint instead of the tiny
random Llama fixture; ART_SFT_TEST_SUPPORT_KEY supplies its ART handler key.
"""

from collections import Counter
import json
import os
from pathlib import Path
import tempfile

import pytest
import torch


def _examples(last_only: bool) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(123)
    prefix = torch.randint(1, 128, (128,), generator=generator)
    examples = []
    for index, length in enumerate((193, 217, 241, 205)):
        tokens = torch.cat(
            (prefix, torch.randint(1, 128, (length - 128,), generator=generator))
        )
        if index == 3:
            tokens[:64] = tokens[:64].flip(0)
        labels = tokens.clone()
        labels[:176] = -100
        if not last_only:
            labels[128:144] = tokens[128:144]
        examples.append(
            {
                "input_ids": tokens[None],
                "labels": labels[None],
                "attention_mask": torch.ones_like(tokens)[None],
            }
        )
    return examples


def test_sft_packing_preserves_training_targets():
    from art.megatron.prefix_tree import parse_prefix_tree
    from art.preprocessing.sft import pack_sft_batch

    for last_only in (False, True):
        examples = _examples(last_only)
        expected = Counter(
            (tuple(example["input_ids"][0, :index].tolist()), int(label))
            for example in examples
            for index, label in enumerate(example["labels"][0])
            if index > 0 and label != -100
        )
        for capacity in (384, 1024):
            actual = Counter()
            for row in pack_sft_batch(examples, seq_len=capacity):
                (tree,) = parse_prefix_tree(
                    group_ids=row["group_ids"], parent_ids=row["parent_ids"]
                )
                segments = {segment.group_id: segment for segment in tree.segments}
                tokens, labels = row["input_ids"][0].tolist(), row["labels"][0].tolist()
                for segment in tree.segments:
                    prefix = [
                        token
                        for ancestor in segment.ancestors
                        for token in tokens[
                            segments[ancestor].start : segments[ancestor].end
                        ]
                    ]
                    for position in range(
                        segment.start, min(segment.end, len(tokens) - 1)
                    ):
                        if labels[position + 1] != -100:
                            actual[
                                (
                                    tuple(
                                        prefix + tokens[segment.start : position + 1]
                                    ),
                                    labels[position + 1],
                                )
                            ] += 1
            assert actual == expected
        with pytest.raises(RuntimeError, match="exceeds sequence length"):
            pack_sft_batch(examples, seq_len=64)


def _run(runtime, inputs):
    from art.megatron.train import run_megatron_sft_step
    from art.megatron.training.microbatches import (
        _zero_contribution_sft_inputs,
        build_micro_sample_indices,
        select_sft_micro_inputs,
        sft_global_microbatch_count,
    )

    indices = build_micro_sample_indices(
        step_index=0,
        num_sequences=len(inputs),
        global_grad_accumulation_sequences=sft_global_microbatch_count(
            len(inputs), provider=runtime.provider
        ),
    )

    gradients = {}

    def capture_gradients():
        for index, model in enumerate(runtime.model):
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    gradients[f"{index}.{name}"] = (
                        parameter.main_grad.detach().float().cpu().clone()
                    )

    result = run_megatron_sft_step(
        model_chunks=runtime.model,
        provider=runtime.provider,
        model_support_handler=runtime.model_support_handler,
        optimizer=runtime.optimizer,
        learning_rate=0.0,
        inputs=select_sft_micro_inputs(
            inputs, indices, _zero_contribution_sft_inputs(inputs[0])
        ),
        step_index=0,
        sample_index=indices,
        before_optimizer_step=capture_gradients,
    )
    return float(result.reduced_loss), gradients


def _mape(candidate, reference, name):
    assert torch.isfinite(candidate).all() and torch.isfinite(reference).all()
    assert candidate.abs().sum() > 0 and reference.abs().sum() > 0, name
    return float((candidate - reference).abs().mean() / reference.abs().mean() * 100)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires Megatron CUDA runtime"
)
def test_sft_packing_loss_and_gradients():
    from transformers import LlamaConfig

    from art.megatron.train import build_training_runtime
    from art.preprocessing.sft import pack_sft_batch

    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    os.environ.setdefault("ART_MEGATRON_LORA_RANK", "8")
    os.environ.setdefault(
        "ART_MEGATRON_LORA_TARGET_MODULES", '["q_proj","k_proj","v_proj","o_proj"]'
    )
    if "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group("nccl")
    else:
        os.environ.update(
            RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1"
        )
        torch.distributed.init_process_group(
            "nccl", store=torch.distributed.HashStore(), rank=0, world_size=1
        )
    try:
        with tempfile.TemporaryDirectory(prefix="art-sft-packing-") as directory:
            checkpoint = os.environ.get("ART_SFT_TEST_MODEL")
            if checkpoint is None:
                LlamaConfig(
                    architectures=["LlamaForCausalLM"],
                    hidden_size=256,
                    intermediate_size=512,
                    num_hidden_layers=max(
                        2,
                        int(
                            os.environ.get(
                                "ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE", "1"
                            )
                        )
                        * int(
                            os.environ.get(
                                "ART_MEGATRON_VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE", "1"
                            )
                        ),
                    ),
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    vocab_size=256,
                    max_position_embeddings=2048,
                ).save_pretrained(directory)
            torch.manual_seed(3407)
            runtime = build_training_runtime(
                model_identifier=checkpoint or directory,
                model_initialization="pretrained" if checkpoint else "random",
                model_support_key=os.environ.get(
                    "ART_SFT_TEST_SUPPORT_KEY", "llama3_dense"
                ),
                print_env=False,
            )
            for model in runtime.model:
                model.train()
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        with torch.no_grad():
                            parameter.normal_(std=0.02)
            assert runtime.optimizer is not None
            runtime.optimizer.reload_model_params()
            rows = []
            for last_only in (False, True):
                examples = _examples(last_only)
                expected_loss, expected_gradients = _run(runtime, examples)
                for capacity in (384, 1024):
                    packed = list(pack_sft_batch(examples, seq_len=capacity))
                    loss, gradients = _run(runtime, packed)
                    loss_error = abs(loss - expected_loss) / abs(expected_loss) * 100
                    assert expected_loss > 0 and loss > 0 and loss_error <= 3.0
                    errors = {
                        name: _mape(gradients[name], reference, name)
                        for name, reference in expected_gradients.items()
                    }
                    assert max(errors.values()) <= 5.0, errors
                    rows.append(
                        dict(
                            last_only=last_only,
                            capacity=capacity,
                            packed_rows=len(packed),
                            loss_mape=loss_error,
                            max_gradient_mape=max(errors.values()),
                        )
                    )
            print(json.dumps(rows), flush=True)
            if output := os.environ.get("ART_SFT_TEST_REPORT"):
                Path(output).with_suffix(
                    f".rank{torch.distributed.get_rank()}.json"
                ).write_text(json.dumps(rows, indent=2))
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test_sft_packing_loss_and_gradients()
