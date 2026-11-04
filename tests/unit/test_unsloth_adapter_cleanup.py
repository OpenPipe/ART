"""KL reference adapters must not leak into saved or reloaded LoRA checkpoints."""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
import pytest
from safetensors.torch import load_file, save_file
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from art.unsloth.train import UnslothTrainContext, _use_adapter

ADAPTER_FILES = ["README.md", "adapter_config.json", "adapter_model.safetensors"]


def tiny_lora_model(seed: int):
    torch.manual_seed(seed)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    config = LoraConfig(
        r=2, target_modules=["q_proj", "v_proj"], init_lora_weights=False
    )
    return get_peft_model(base, config)


def make_context(peft_model) -> UnslothTrainContext:
    trainer = SimpleNamespace(
        model=peft_model,
        accelerator=SimpleNamespace(unwrap_model=lambda model, **_: model),
        optimizer=None,
        save_model=peft_model.save_pretrained,
    )
    return UnslothTrainContext(
        model=peft_model,
        tokenizer=None,  # ty: ignore[invalid-argument-type]
        peft_model=peft_model,
        trainer=trainer,  # ty: ignore[invalid-argument-type]
        inputs_queue=asyncio.Queue(),
        results_queue=asyncio.Queue(),
    )


def default_weights(peft_model) -> dict[str, torch.Tensor]:
    return {
        key: value.clone()
        for key, value in get_peft_model_state_dict(peft_model).items()
    }


@pytest.fixture
def adapter_dirs(tmp_path: Path) -> tuple[str, str]:
    """One adapter used as a KL reference and one from another job."""
    reference = tmp_path / "other-model" / "step0"
    tiny_lora_model(seed=1).save_pretrained(reference)
    other_job = tmp_path / "other-job"
    tiny_lora_model(seed=2).save_pretrained(other_job)
    return str(reference), str(other_job)


def test_save_after_kl_reference_holds_only_trained_adapter(
    adapter_dirs, tmp_path: Path
) -> None:
    reference, _ = adapter_dirs
    peft_model = tiny_lora_model(seed=0)
    ctx = make_context(peft_model)
    trained = default_weights(peft_model)
    with _use_adapter(ctx.trainer, reference):
        pass
    assert len(peft_model.peft_config) == 2

    ctx.save_lora_adapter(str(tmp_path / "checkpoint"))

    assert sorted(os.listdir(tmp_path / "checkpoint")) == ADAPTER_FILES
    assert list(peft_model.peft_config) == ["default"]
    saved = load_file(tmp_path / "checkpoint" / "adapter_model.safetensors")
    assert saved.keys() == trained.keys()
    assert all(torch.equal(saved[key], trained[key]) for key in trained)

    # The next KL step loads the reference again.
    with _use_adapter(ctx.trainer, reference):
        assert peft_model.active_adapter != "default"
    assert peft_model.active_adapter == "default"


def test_load_replaces_weights_and_drops_reference(adapter_dirs) -> None:
    reference, other_job = adapter_dirs
    peft_model = tiny_lora_model(seed=0)
    ctx = make_context(peft_model)
    with _use_adapter(ctx.trainer, reference):
        pass

    asyncio.run(ctx.load_lora_adapter(other_job))

    assert list(peft_model.peft_config) == ["default"]
    expected = load_file(os.path.join(other_job, "adapter_model.safetensors"))
    loaded = default_weights(peft_model)
    assert all(torch.equal(loaded[key], expected[key]) for key in expected)


def test_load_refuses_adapter_missing_weights(adapter_dirs, tmp_path: Path) -> None:
    _, other_job = adapter_dirs
    weights = load_file(os.path.join(other_job, "adapter_model.safetensors"))
    partial = tmp_path / "partial"
    partial.mkdir()
    save_file(
        {key: value for key, value in weights.items() if "v_proj" not in key},
        partial / "adapter_model.safetensors",
    )
    peft_model = tiny_lora_model(seed=0)
    ctx = make_context(peft_model)

    with pytest.raises(RuntimeError, match="missing 2 LoRA weights"):
        asyncio.run(ctx.load_lora_adapter(str(partial)))
