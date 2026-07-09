from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn

from art import types
from art.megatron import train as megatron_train
from art.megatron.training import microbatches as megatron_microbatches
from art.preprocessing.pack import PackedTensors


def _packed_inputs(seq_len: int = 4) -> PackedTensors:
    return cast(
        PackedTensors,
        {
            "tokens": torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            "input_pos": torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            "assistant_mask": torch.ones((1, seq_len), dtype=torch.bool),
            "group_ids": torch.zeros((1, seq_len), dtype=torch.long),
            "parent_ids": torch.zeros((1, seq_len), dtype=torch.long),
        },
    )


def test_precompute_reference_logprobs_preserves_sample_steps(monkeypatch) -> None:
    calls: list[tuple[int, int, int]] = []

    def fake_select_indexed_inputs(
        packed_tensors: dict[str, torch.Tensor], sample_index: int
    ) -> dict[str, torch.Tensor]:
        del packed_tensors
        return {"sample_index": torch.tensor(sample_index)}

    def fake_calculate_megatron_logprobs(
        *,
        model_chunks: Any,
        provider: Any,
        model_support_handler: Any,
        inputs: dict[str, torch.Tensor],
        moe_routing_replay_controller: Any,
        step_index: int,
        sample_index: int,
        global_grad_accumulation_sequences: int,
    ) -> torch.Tensor:
        del (
            model_chunks,
            provider,
            model_support_handler,
            inputs,
            moe_routing_replay_controller,
        )
        calls.append((sample_index, step_index, global_grad_accumulation_sequences))
        return torch.tensor([[float(sample_index)]])

    monkeypatch.setattr(
        megatron_train, "select_indexed_inputs", fake_select_indexed_inputs
    )
    monkeypatch.setattr(
        megatron_train,
        "_calculate_megatron_logprobs",
        fake_calculate_megatron_logprobs,
    )
    runtime = SimpleNamespace(
        rank=0,
        model=[],
        provider=object(),
        model_support_handler=object(),
        moe_routing_replay_controller=object(),
    )

    result = megatron_train._precompute_reference_logprobs(
        runtime=cast(megatron_train.TrainingRuntime, runtime),
        packed_tensors=_packed_inputs(),
        sample_step_indices={3: 1, 0: 0},
        global_grad_accumulation_sequences=4,
    )

    assert calls == [(0, 0, 4), (3, 1, 4)]
    assert sorted(result) == [0, 3]


def test_prepare_kl_reference_logprobs_requires_reference_path() -> None:
    runtime = SimpleNamespace(rank=0)
    job = SimpleNamespace(
        config=types.TrainConfig(kl_penalty_coef=0.25),
        experimental_config={},
        lora_path="/tmp/current",
    )

    try:
        megatron_train._prepare_kl_reference_logprobs(
            runtime=cast(megatron_train.TrainingRuntime, runtime),
            job=cast(megatron_train.MegatronTrainingJob, job),
            adapter_model={},
            packed_tensors=_packed_inputs(),
            num_sequences=1,
            num_steps=1,
            global_grad_accumulation_sequences=1,
        )
    except RuntimeError as exc:
        assert "kl_ref_adapter_path" in str(exc)
    else:
        raise AssertionError("Expected missing reference path to raise")


class _ReplayController:
    def __init__(self) -> None:
        self.events: list[tuple[str, int, int | None, int | None]] = []

    def set_step(
        self,
        *,
        step_index: int,
        sample_index: int,
        global_grad_accumulation_sequences: int | None = None,
    ) -> None:
        self.events.append(
            ("set_step", step_index, sample_index, global_grad_accumulation_sequences)
        )

    def begin_micro(self, sample_index: int, micro_order: int) -> None:
        self.events.append(("begin_micro", micro_order, sample_index, None))

    def finalize_step(self) -> None:
        self.events.append(("finalize_step", 0, None, None))


class _Chunk(nn.Module):
    def __init__(self, controller: _ReplayController) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.controller = controller
        self.training_modes_seen: list[bool] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        packed_seq_params: Any | None = None,
    ) -> torch.Tensor:
        del input_ids, position_ids, attention_mask, packed_seq_params
        self.training_modes_seen.append(self.training)
        assert self.controller.events == [
            ("set_step", 2, 5, 8),
            ("begin_micro", 0, 5, None),
        ]
        return torch.full(labels.shape, 0.25, dtype=torch.float32, device=labels.device)


class _Handler:
    def get_forward_kwargs(self, _chunk: nn.Module, *, attention_bias: Any) -> dict:
        del attention_bias
        return {}


def test_calculate_megatron_logprobs_replays_routes(monkeypatch) -> None:
    controller = _ReplayController()
    chunk = _Chunk(controller)
    monkeypatch.setattr(
        megatron_microbatches,
        "create_prefix_tree_state",
        lambda **kwargs: (kwargs["group_ids"], kwargs["parent_ids"]),
    )
    monkeypatch.setattr(
        megatron_train,
        "_infer_parallel_topology",
        lambda _model_chunks: megatron_train.ParallelTopology(),
    )

    logprobs = megatron_train._calculate_megatron_logprobs(
        model_chunks=cast(megatron_train.ModelChunks, [chunk]),
        provider=object(),
        model_support_handler=_Handler(),
        inputs=_packed_inputs(),
        moe_routing_replay_controller=cast(
            megatron_train.MoeRoutingReplayController, controller
        ),
        step_index=2,
        sample_index=5,
        global_grad_accumulation_sequences=8,
    )

    assert controller.events == [
        ("set_step", 2, 5, 8),
        ("begin_micro", 0, 5, None),
        ("finalize_step", 0, None, None),
    ]
    assert chunk.training_modes_seen == [False]
    assert chunk.training is True
    assert torch.equal(logprobs, torch.full((1, 4), -0.25))
