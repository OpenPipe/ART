from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
import gc
from importlib.util import find_spec
import inspect
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.prefix_tree_packing import prefix_tree_pack
from art.trainer_rank import (
    AdamParams,
    AdapterSelection,
    ForwardInput,
    ForwardOutput,
    MaterializedCheckpoint,
    TopK,
    TrainerRank,
    TrainerRankMemoryError,
    TrainerRankSlotStateError,
    Unset,
)
from art.trainer_rank._checkpoint import (
    CheckpointManifest,
    LocalOptimizerState,
    OptimizerConfig,
    PreparedCheckpoint,
    _file_digest,
    _FinalizedSave,
    _manifest_digest,
    _merge_component,
    _PreparedSave,
    _slot_snapshot,
    _validate_save_state,
    abort_checkpoint_save,
    finish_checkpoint_save,
    materialize_checkpoint,
    materialize_lora,
    prepare_checkpoint,
    prepare_checkpoint_save,
    snapshot_prepared_checkpoint,
    validate_checkpoint,
)
from art.trainer_rank._impl import (
    _anchor_disconnected_outputs,
    _CheckpointSlot,
    _MemoryCheck,
    _MemoryProfile,
    _validate_top_k,
)

if TYPE_CHECKING:
    from art.megatron.lora import LoRASlotRef
    from art.megatron.train import TrainingRuntime


class _Model:
    vocab_size = 8


def test_public_types_have_canonical_module_paths() -> None:
    import art.trainer_rank

    assert {
        "AdapterSelection",
        "Unset",
    } <= set(art.trainer_rank.__all__)
    for public_type in (
        AdamParams,
        ForwardInput,
        ForwardOutput,
        TopK,
        TrainerRank,
        TrainerRankMemoryError,
        TrainerRankSlotStateError,
    ):
        assert public_type.__module__ == "art.trainer_rank"


class _FakeLoRASite(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.A_T = torch.nn.Parameter(torch.zeros(4, 2, device=device, dtype=dtype))
        self.B_T = torch.nn.Parameter(torch.zeros(2, 5, device=device, dtype=dtype))

    def _expected_weight_keys(self, suffix: str) -> list[str]:
        return [f"{self.prefix}.{suffix}.weight"]


class _NativeOptimizer:
    config = None
    param_groups: list[dict[str, object]] = []

    def __init__(self) -> None:
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self) -> tuple[bool, float, int | None]:
        self.step_calls += 1
        raise AssertionError("TrainerRank must not step the native optimizer")

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1


@dataclass(frozen=True)
class _SlotRef:
    name: str | None


def _runtime(
    model: torch.nn.Module | None = None,
    *,
    optimizer: object | None = None,
) -> "TrainingRuntime":
    # Deliberately lightweight structural fake; importing/constructing the real
    # Megatron runtime would make these CPU-only unit tests require Megatron.
    return SimpleNamespace(
        model=[model or torch.nn.Linear(1, 1)],
        optimizer=optimizer,
        provider=SimpleNamespace(
            hidden_size=4,
            num_layers=1,
            kv_channels=2,
            art_flex_sliding_windows=(16,),
        ),
        model_support_handler=SimpleNamespace(
            build_gdn_execution_spec=True,
            canonicalize_loaded_lora_state=lambda state, _model: state,
            from_vllm_lora_tensors=lambda state, **_kwargs: state,
            to_vllm_lora_tensors=lambda state, **kwargs: (
                state,
                kwargs["adapter_config"],
            ),
            zero_internal_padding_grads=lambda _model: None,
            zero_internal_padding_params=lambda _model: None,
        ),
        rank=0,
        world_size=1,
    )  # type: ignore


def _slot_ref(name: str | None) -> "LoRASlotRef":
    return _SlotRef(name)  # type: ignore


def _target_request(token: int) -> ForwardInput[torch.Tensor, None, None, None]:
    tokens = torch.tensor([token, token + 1], dtype=torch.long)
    return ForwardInput(input_tokens=tokens, target_tokens=tokens)


def _indexed_outputs(plan: object, **_kwargs: object) -> list[ForwardOutput]:
    return [
        ForwardOutput(torch.tensor([index], dtype=torch.float32), None, None, None)
        for index in range(int(getattr(plan, "request_count")))
    ]


def _empty_outputs(plan: object, **_kwargs: object) -> list[ForwardOutput]:
    return [ForwardOutput(None, None, None, None)] * int(getattr(plan, "request_count"))


def _stub_forward(mp, rank, out=_empty_outputs, dp=(0, 1), profiled=False) -> None:
    mp.setattr(rank, "_dp_rank_and_size", lambda: dp)

    def run(*args, **kwargs):
        return out(*args, **kwargs), None

    mp.setattr(rank, "_run_flat_plan_with_memory_tracking", run)
    if profiled:
        mp.setattr(rank, "_all_ranks_have_memory_profile", lambda **_: True)


def _output_values(outputs: object) -> list[int]:
    if isinstance(outputs, ForwardOutput):
        target_logprobs = outputs.target_logprobs
        assert isinstance(target_logprobs, torch.Tensor)
        return [int(target_logprobs.item())]
    values: list[int] = []
    assert isinstance(outputs, Iterable)
    for item in outputs:
        values.extend(_output_values(item))
    return values


def _output_shape(outputs: object) -> object:
    if isinstance(outputs, ForwardOutput):
        return "output"
    assert isinstance(outputs, Iterable)
    return [_output_shape(item) for item in outputs]


def _trainer_with_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    value: torch.Tensor,
) -> tuple[TrainerRank, torch.nn.Parameter]:
    trainer = TrainerRank(_runtime())
    param = torch.nn.Parameter(value.clone())
    trainer._checkpoint_slots.setdefault("student", _CheckpointSlot()).params = (param,)
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(item.grad.float() for item in params),
    )
    return trainer, param


def _tracked_targets(
    trainer: TrainerRank, ref: "LoRASlotRef", *scales: float
) -> list[torch.Tensor]:
    tracked = trainer._track_slot_graph_outputs(
        ref,
        [
            ForwardOutput(torch.ones(1, requires_grad=True) * scale, None, None, None)
            for scale in scales
        ],
    )
    targets: list[torch.Tensor] = []
    for output in tracked:
        target = output.target_logprobs
        assert isinstance(target, torch.Tensor)
        targets.append(target)
    return targets


def test_forward_input_validation() -> None:
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        ForwardInput(input_tokens=torch.tensor([1]), top_k=0)
    assert "lora" not in ForwardInput.__dataclass_fields__
    with pytest.raises(ValueError, match="top_k=9 exceeds vocabulary size 8"):
        _validate_top_k(9, _Model())


@pytest.mark.parametrize(("checkpoint", "expected"), ((Unset, Unset), (None, None)))
def test_forward_input_distinguishes_unset_and_base_checkpoint(
    checkpoint: AdapterSelection, expected: AdapterSelection
) -> None:
    request = ForwardInput(input_tokens=torch.tensor([1]), checkpoint=checkpoint)

    assert request.checkpoint is expected


def test_dp_rank_forward_rejects_unloaded_explicit_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    _stub_forward(monkeypatch, trainer)
    request = ForwardInput(
        input_tokens=torch.tensor([1]),
        target_tokens=torch.tensor([1]),
        checkpoint="typo",
    )

    with pytest.raises(TrainerRankSlotStateError, match="unloaded.*'typo'"):
        trainer.dp_rank_forward([request])


@pytest.mark.parametrize("checkpoint", (None, "student"))
def test_dp_rank_forward_accepts_base_or_loaded_explicit_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str | None,
) -> None:
    trainer = TrainerRank(_runtime())
    trainer._checkpoint_slots.setdefault("student", _CheckpointSlot()).params = ()
    _stub_forward(monkeypatch, trainer)
    request = ForwardInput(
        input_tokens=torch.tensor([1]),
        target_tokens=torch.tensor([1]),
        checkpoint=checkpoint,
    )

    output = trainer.dp_rank_forward([request])

    assert isinstance(output[0], ForwardOutput)


def test_forward_method_checkpoint_is_request_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    trainer._checkpoint_slots["method"] = _CheckpointSlot()
    trainer._checkpoint_slots["request"] = _CheckpointSlot()
    monkeypatch.setattr(trainer, "_slot_ref", _slot_ref)
    seen: list[str | None] = []

    def execute(plan: object, **_kwargs: object) -> list[ForwardOutput]:
        seen.extend(group.slot_ref.name for group in cast(Any, plan).groups)
        return _empty_outputs(plan)

    _stub_forward(monkeypatch, trainer, execute)
    inputs = [
        ForwardInput(input_tokens=torch.tensor([1]), target_tokens=torch.tensor([1])),
        ForwardInput(
            input_tokens=torch.tensor([2]),
            target_tokens=torch.tensor([2]),
            checkpoint="request",
        ),
        ForwardInput(
            input_tokens=torch.tensor([3]),
            target_tokens=torch.tensor([3]),
            checkpoint=None,
        ),
    ]

    trainer.dp_rank_forward(inputs, checkpoint="method")

    assert seen == ["method", "request", None]


def test_forward_method_checkpoint_rejects_unloaded_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    _stub_forward(monkeypatch, trainer)

    with pytest.raises(TrainerRankSlotStateError, match="unloaded.*'typo'"):
        trainer.dp_rank_forward([_target_request(1)], checkpoint="typo")


@pytest.mark.parametrize(
    ("ambient_grad", "no_grad", "expected"),
    (
        (True, None, True),
        (False, None, False),
        (True, True, False),
        (False, False, True),
    ),
)
def test_dp_rank_forward_grad_mode(
    monkeypatch: pytest.MonkeyPatch,
    ambient_grad: bool,
    no_grad: bool | None,
    expected: bool,
) -> None:
    trainer = TrainerRank(_runtime())
    seen: list[bool] = []

    def execute(plan: object, **_kwargs: object) -> list[ForwardOutput]:
        seen.append(torch.is_grad_enabled())
        return _empty_outputs(plan)

    _stub_forward(monkeypatch, trainer, execute)
    with torch.set_grad_enabled(ambient_grad):
        trainer.dp_rank_forward([_target_request(1)], no_grad=no_grad)

    assert seen == [expected]


@pytest.mark.parametrize("api", ("dp_rank_forward", "forward_micro_batches"))
def test_forward_input_overrides_grad_mode_by_group(
    monkeypatch: pytest.MonkeyPatch,
    api: str,
) -> None:
    trainer = TrainerRank(_runtime())
    seen: list[bool] = []

    def execute(plan: object, **_kwargs: object) -> list[ForwardOutput]:
        seen.extend(group.grad_enabled for group in cast(Any, plan).groups)
        return _empty_outputs(plan)

    _stub_forward(monkeypatch, trainer, execute, profiled=True)
    inputs = [
        ForwardInput(
            input_tokens=torch.tensor([token, token + 1]),
            target_tokens=torch.tensor([token, token + 1]),
            no_grad=no_grad,
        )
        for token, no_grad in ((1, True), (2, False))
    ]
    if api == "dp_rank_forward":
        trainer.dp_rank_forward(inputs)
    else:
        list(trainer.forward_micro_batches(inputs))

    assert seen == [False, True]


def test_forward_groups_execute_in_their_selected_grad_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    seen: list[bool] = []
    groups = tuple(
        SimpleNamespace(
            slot_ref=_slot_ref(checkpoint),
            grad_enabled=enabled,
            packed=None,
            items=(None,),
            request_indices=(index,),
        )
        for index, (checkpoint, enabled) in enumerate(
            (("teacher", False), ("student", True))
        )
    )
    plan = SimpleNamespace(
        request_count=2,
        output_metadata=(("teacher", True), ("student", False)),
        groups=groups,
    )

    monkeypatch.setattr(trainer, "_validate_hybridep_topology", lambda: None)
    monkeypatch.setattr(trainer, "_topology", lambda: object())
    monkeypatch.setattr(trainer, "_configure_hybridep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer, "_prepare_packed_forward", lambda _packed: None)

    class UseLoRASlot:
        def __enter__(self) -> None:
            pass

        def __exit__(self, *_args: object) -> None:
            pass

    lora = ModuleType("art.megatron.lora")
    cast(Any, lora).use_lora_slot = lambda _slot: UseLoRASlot()
    monkeypatch.setitem(sys.modules, "art.megatron.lora", lora)

    def forward(_items: object, _prepared: object) -> list[ForwardOutput]:
        seen.append(torch.is_grad_enabled())
        return [ForwardOutput(None, None, None, None)]

    monkeypatch.setattr(trainer, "_forward_packed", forward)
    outputs = cast(Any, trainer)._execute_flat_plan(plan)

    assert seen == [False, True]
    assert [(output.checkpoint, output.no_grad) for output in outputs] == [
        ("teacher", True),
        ("student", False),
    ]


def test_forward_micro_batches_keeps_grad_mode_across_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    seen: list[bool] = []

    def execute(plan: object, **_kwargs: object) -> list[ForwardOutput]:
        seen.append(torch.is_grad_enabled())
        return _empty_outputs(plan)

    _stub_forward(monkeypatch, trainer, execute, profiled=True)
    batches = trainer.forward_micro_batches(
        [_target_request(index) for index in range(3)], no_grad=True
    )
    assert torch.is_grad_enabled()

    list(batches)

    assert seen and not any(seen)
    assert torch.is_grad_enabled()


def test_forward_micro_batches_uses_method_checkpoint_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    trainer._checkpoint_slots["teacher"] = _CheckpointSlot()
    monkeypatch.setattr(trainer, "_slot_ref", _slot_ref)
    seen: list[str | None] = []

    def execute(plan: object, **_kwargs: object) -> list[ForwardOutput]:
        seen.extend(group.slot_ref.name for group in cast(Any, plan).groups)
        return _empty_outputs(plan)

    _stub_forward(monkeypatch, trainer, execute, profiled=True)

    list(trainer.forward_micro_batches([_target_request(1)], checkpoint="teacher"))

    assert seen == ["teacher"]


def test_forward_input_preserves_public_runtime_shape() -> None:
    fields = tuple(ForwardInput.__dataclass_fields__)
    assert tuple(inspect.signature(ForwardInput).parameters) == fields
    assert ForwardInput.__match_args__ == fields


@pytest.mark.parametrize(
    "knob",
    (
        "shared_prefix_max_depth",
        "head_chunk_tokens",
        "memory_safety_factor",
        "memory_reserve_fraction",
    ),
)
def test_trainer_rank_rejects_removed_planner_knobs(knob: str) -> None:
    with pytest.raises(TypeError):
        TrainerRank(_runtime(), **{knob: 1})


@pytest.mark.skipif(find_spec("megatron") is None, reason="requires Megatron")
def test_cp1_packed_forward_uses_model_attention_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron.context_parallel.types import ParallelTopology

    runtime = _runtime()
    trainer = TrainerRank(runtime)
    trainer.device = torch.device("meta")
    batch = prefix_tree_pack(
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 4])), max_depth=1
    )
    captured: dict[str, object] = {}
    state = object()

    def create_state(**kwargs: object) -> object:
        captured.update(kwargs)
        return state

    monkeypatch.setattr(trainer, "_topology", lambda: ParallelTopology())
    monkeypatch.setattr(trainer, "_configure_hybridep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "art.megatron.prefix_tree_state.create_prefix_tree_state", create_state
    )
    monkeypatch.setattr(
        "art.megatron.training.microbatches._gdn_planner_config_for_provider",
        lambda provider, handler: "planner-config",
    )

    prepared = trainer._prepare_packed_forward(batch)

    assert prepared.attention_state is state
    assert captured["model_support_handler"] is runtime.model_support_handler
    assert captured["sliding_windows"] == (16,)
    assert captured["gdn_planner_config"] == "planner-config"
    assert captured["target_device"] == torch.device("meta")
    assert prepared.tokens.device == torch.device("meta")
    assert prepared.position_ids.device == torch.device("meta")
    assert batch.tokens.device == torch.device("cpu")
    assert batch.position_ids.device == torch.device("cpu")
    torch.testing.assert_close(
        cast(torch.Tensor, captured["input_pos"]), batch.position_ids
    )


@pytest.mark.parametrize("dp", [1, 4])
def test_hybridep_validates_topology_for_empty_forward(
    monkeypatch: pytest.MonkeyPatch,
    dp: int,
) -> None:
    runtime = _runtime()
    runtime.provider.expert_model_parallel_size = 4
    trainer = TrainerRank(runtime)
    monkeypatch.setattr(trainer, "_topology", lambda: SimpleNamespace(dp=dp, cp=4))

    if dp > 1:
        with pytest.raises(NotImplementedError, match="DP=1"):
            trainer.dp_rank_forward([])
    else:
        assert trainer.dp_rank_forward([]) == []


@pytest.mark.skipif(find_spec("megatron") is None, reason="requires Megatron")
def test_hybridep_uses_maximum_cp_model_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron.context_parallel.types import ParallelTopology

    trainer = TrainerRank(_runtime())
    batch = prefix_tree_pack((torch.arange(9),), max_depth=0)
    short_batch = prefix_tree_pack((torch.arange(5),), max_depth=0)
    topology = ParallelTopology(cp=4)
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "megatron.core.parallel_state.get_expert_model_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(
        "art.megatron.train._ensure_hybridep_capacity",
        lambda runtime, **kwargs: calls.update(capacity=kwargs),
    )
    monkeypatch.setattr(
        "art.megatron.context_parallel.runtime.context_parallel_rank_model_token_counts",
        lambda **kwargs: (
            8,
            int(cast(torch.Tensor, kwargs["group_ids"]).numel()) + 4,
            9,
            11,
        ),
    )
    monkeypatch.setattr(
        "art.megatron.training.microbatches._context_parallel_config_for_provider",
        lambda *_: "cp-config",
    )
    monkeypatch.setattr(
        "art.megatron.training.microbatches._gdn_planner_config_for_provider",
        lambda *_: "gdn-config",
    )

    buffer = SimpleNamespace(
        configurer=SimpleNamespace(
            buffer_config=SimpleNamespace(max_num_of_tokens_per_rank=1024)
        )
    )
    monkeypatch.setattr(
        "megatron.core.transformer.moe.fused_a2a._hybrid_ep_buffer", buffer
    )

    configured = trainer._configure_hybridep((batch, short_batch), topology=topology)

    assert calls == {
        "capacity": {
            "packed_sequence_length": 9,
            "context_parallel_size": 4,
        },
    }
    assert configured == ((13, 11), 13)


@pytest.mark.skipif(find_spec("megatron") is None, reason="requires Megatron")
def test_hybridep_rejects_buffer_growth_with_live_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron.context_parallel.types import ParallelTopology

    trainer = TrainerRank(_runtime())
    trainer._hybridep_graph_tracking = True
    output = trainer._track_slot_graph_outputs(
        None,
        [ForwardOutput(torch.ones(1, requires_grad=True), None, None, None)],
    )[0]
    assert output.target_logprobs is not None
    buffer = SimpleNamespace(
        configurer=SimpleNamespace(
            buffer_config=SimpleNamespace(max_num_of_tokens_per_rank=4)
        )
    )
    trainer._hybridep_buffer_id = id(buffer)
    trainer._hybridep_rows_high_water = 4
    monkeypatch.setattr(
        "megatron.core.parallel_state.get_expert_model_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(
        "megatron.core.transformer.moe.fused_a2a._hybrid_ep_buffer", buffer
    )

    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._configure_hybridep(
            (prefix_tree_pack((torch.arange(9),), max_depth=0),),
            topology=ParallelTopology(),
        )


def test_trainer_rank_checkpoint_stack_errors() -> None:
    trainer = TrainerRank(_runtime())

    with pytest.raises(RuntimeError, match="No pushed checkpoint"):
        trainer.pop_checkpoint()
    trainer._slot_stack.append(object())  # type: ignore
    with pytest.raises(RuntimeError, match="Cannot load a checkpoint"):
        trainer.load_checkpoint("teacher")


async def test_checkpoint_prefetch_and_sync_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    fetched: list[str] = []

    def prepare(path: str) -> object:
        fetched.append(path)
        return object()

    def install(trainer: TrainerRank, _source: object, path: str) -> None:
        trainer._checkpoint_slots.setdefault(path, _CheckpointSlot()).params = ()
        trainer._checkpoint_slots[path].revision = 0

    monkeypatch.setattr("art.trainer_rank._checkpoint.prepare_checkpoint", prepare)
    monkeypatch.setattr("art.trainer_rank._checkpoint.load_checkpoint", install)

    assert trainer.load_checkpoint("student") is None
    assert fetched == [trainer._checkpoint_source_key("student")]
    assert trainer._default_slot_ref == trainer._slot_ref("student")

    task = trainer.prefetch_checkpoints("teacher", "reference")
    assert isinstance(task, asyncio.Task)
    await task
    assert fetched[-2:] == [
        trainer._checkpoint_source_key("teacher"),
        trainer._checkpoint_source_key("reference"),
    ]

    with trainer.push_checkpoint("student"):
        assert trainer._slot_stack == [trainer._slot_ref("student")]
        with trainer.push_checkpoint("missing"):
            assert trainer._slot_stack == [
                trainer._slot_ref("student"),
                trainer._slot_ref("missing"),
            ]
        assert trainer._slot_stack == [trainer._slot_ref("student")]
    assert trainer._slot_stack == []
    assert fetched[-1] == trainer._checkpoint_source_key("missing")


def test_checkpoint_sync_context_and_body_error_preservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    trainer._checkpoint_slots["student"] = _CheckpointSlot()
    with trainer.push_checkpoint("student"):
        assert trainer._slot_stack == [trainer._slot_ref("student")]
    assert trainer._slot_stack == []

    pushed = trainer.push_checkpoint("student")
    monkeypatch.setattr(
        trainer,
        "pop_checkpoint",
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )
    with pytest.raises(ExceptionGroup) as captured:
        with pushed:
            raise ValueError("body failed")
    assert {type(error) for error in captured.value.exceptions} == {
        ValueError,
        RuntimeError,
    }


def test_pushed_checkpoint_cannot_be_reused() -> None:
    trainer = TrainerRank(_runtime())
    trainer._checkpoint_slots["student"] = _CheckpointSlot()
    pushed = trainer.push_checkpoint("student")
    assert not isinstance(pushed, asyncio.Future)
    assert not hasattr(pushed, "__await__")
    assert not hasattr(pushed, "__aenter__")

    with pushed:
        pass
    with pytest.raises(RuntimeError, match="cannot be entered twice"):
        with pushed:
            pass


def test_snapshot_disposal_is_not_public() -> None:
    assert not hasattr(TrainerRank, "discard_snapshot_checkpoint")


@pytest.mark.parametrize(
    "consumer",
    (
        "module",
        "parameter",
        "buffer",
        "forward",
        "forward_micro_batches",
        "optim_step",
        "save",
        "export_lora",
        "snapshot",
    ),
)
def test_explicit_consumers_activate_prefetched_checkpoint(
    consumer: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer = TrainerRank(_runtime())
    installed: list[str] = []

    def install(target: TrainerRank, _source: object, name: str) -> None:
        installed.append(name)
        target._checkpoint_slots[name] = _CheckpointSlot()

    trainer._register_checkpoint_prefetch(
        "student", "student", lambda: cast(PreparedCheckpoint, object())
    )
    monkeypatch.setattr("art.trainer_rank._checkpoint.load_checkpoint", install)

    if consumer == "module":
        trainer.module("head", lambda: torch.nn.Linear(1, 1), checkpoint="student")
    elif consumer == "parameter":
        trainer.parameter("bias", lambda: torch.ones(1), checkpoint="student")
    elif consumer == "buffer":
        trainer.buffer("mean", lambda: torch.zeros(1), checkpoint="student")
    elif consumer == "forward":
        _stub_forward(monkeypatch, trainer)
        trainer.dp_rank_forward([_target_request(1)], checkpoint="student")
    elif consumer == "forward_micro_batches":
        _stub_forward(monkeypatch, trainer, profiled=True)
        list(trainer.forward_micro_batches([_target_request(1)], checkpoint="student"))
    elif consumer == "optim_step":
        with pytest.raises(TrainerRankSlotStateError, match="no gradients"):
            trainer.optim_step(
                checkpoints=["student"], params=AdamParams(learning_rate=1e-3)
            )
    elif consumer == "save":
        monkeypatch.setattr(
            "art.trainer_rank._checkpoint.prepare_checkpoint_save",
            lambda *_args: None,
        )
        trainer.prepare_checkpoint_save(str(tmp_path), "student")
    elif consumer == "export_lora":
        monkeypatch.setattr(
            "art.trainer_rank._lora_export.export_lora", lambda *_args: 1
        )
        assert trainer.export_lora(str(tmp_path), "student") == 1
    else:
        monkeypatch.setattr(
            "art.trainer_rank._checkpoint.snapshot_checkpoint",
            lambda *_args: True,
        )
        assert trainer.snapshot_checkpoint("student", "saved")

    assert installed == ["student"]
    assert trainer._default_slot_ref is None


def test_implicit_optim_step_does_not_activate_prefetched_checkpoints() -> None:
    trainer = TrainerRank(_runtime())
    future = trainer._register_checkpoint_prefetch(
        "student", "student", lambda: cast(PreparedCheckpoint, object())
    )
    future.result()

    with pytest.raises(TrainerRankSlotStateError, match="requires a loaded checkpoint"):
        trainer.optim_step(params=AdamParams(learning_rate=1e-3))

    assert "student" not in trainer._checkpoint_slots
    assert "student" in trainer._checkpoint_prefetch_sources


def test_collective_activation_loads_required_union_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    loaded: list[str] = []
    for name in ("a", "b"):
        trainer._register_checkpoint_prefetch(
            name, name, lambda: cast(PreparedCheckpoint, object())
        ).result()

    def gather(value: object, _group: object = None) -> tuple[object, ...]:
        if isinstance(value, tuple) and value and isinstance(value[0], str):
            return (("b",), ("a",))
        return (value, value)

    def install(target: TrainerRank, _source: object, name: str) -> None:
        loaded.append(name)
        target._checkpoint_slots[name] = _CheckpointSlot()

    monkeypatch.setattr("art.trainer_rank._checkpoint._gather", gather)
    monkeypatch.setattr("art.trainer_rank._checkpoint.load_checkpoint", install)

    trainer._ensure_checkpoint_slots(("b",))

    assert loaded == ["a", "b"]


def test_concurrent_first_consumers_install_prefetched_slot_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    started = threading.Event()
    release = threading.Event()
    installed = 0
    errors: list[BaseException] = []

    def prepare() -> PreparedCheckpoint:
        started.set()
        assert release.wait(5)
        return cast(PreparedCheckpoint, object())

    def install(target: TrainerRank, _source: object, name: str) -> None:
        nonlocal installed
        installed += 1
        target._checkpoint_slots[name] = _CheckpointSlot()

    def consume() -> None:
        try:
            trainer._ensure_checkpoint_slots(("student",))
        except BaseException as error:
            errors.append(error)

    trainer._register_checkpoint_prefetch("student", "student", prepare)
    monkeypatch.setattr("art.trainer_rank._checkpoint.load_checkpoint", install)
    first = threading.Thread(target=consume)
    second = threading.Thread(target=consume)
    first.start()
    assert started.wait(5)
    second.start()
    release.set()
    first.join()
    second.join()

    assert errors == []
    assert installed == 1


async def test_shared_checkpoint_prefetch_survives_waiter_cancellation() -> None:
    trainer = TrainerRank(_runtime())
    started = threading.Event()
    release = threading.Event()
    source = cast(PreparedCheckpoint, object())

    def prepare() -> PreparedCheckpoint:
        started.set()
        release.wait()
        return source

    future = trainer._register_checkpoint_prefetch("student", "student", prepare)
    first = asyncio.create_task(trainer._await_checkpoint_prefetch(future))
    second = asyncio.create_task(trainer._await_checkpoint_prefetch(future))
    await asyncio.to_thread(started.wait)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()

    assert await second is source
    [cached] = trainer._checkpoint_prefetches.values()
    assert cached.result() is source


async def test_shared_checkpoint_prefetch_serves_successful_waiters() -> None:
    trainer = TrainerRank(_runtime())
    started = threading.Event()
    release = threading.Event()
    source = cast(PreparedCheckpoint, object())
    calls = 0

    def prepare() -> PreparedCheckpoint:
        nonlocal calls
        calls += 1
        started.set()
        release.wait()
        return source

    first_future = trainer._register_checkpoint_prefetch("student", "shared", prepare)
    second_future = trainer._register_checkpoint_prefetch("teacher", "shared", prepare)
    first = asyncio.create_task(trainer._await_checkpoint_prefetch(first_future))
    second = asyncio.create_task(trainer._await_checkpoint_prefetch(second_future))
    await asyncio.to_thread(started.wait)
    release.set()

    assert await asyncio.gather(first, second) == [source, source]
    assert calls == 1
    [cached] = trainer._checkpoint_prefetches.values()
    assert cached.result() is source


async def test_materialized_sources_keep_logical_checkpoint_identities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer = TrainerRank(_runtime())
    prepared: list[str] = []
    installed: list[tuple[str, object]] = []

    def prepare(source_path: str) -> object:
        prepared.append(source_path)
        return object()

    def install(trainer: TrainerRank, source: object, logical_path: str) -> None:
        installed.append((logical_path, source))
        trainer._checkpoint_slots.setdefault(
            logical_path, _CheckpointSlot()
        ).params = ()
        trainer._checkpoint_slots[logical_path].revision = 0

    monkeypatch.setattr("art.trainer_rank._checkpoint.prepare_checkpoint", prepare)
    monkeypatch.setattr("art.trainer_rank._checkpoint.load_checkpoint", install)
    root_a = str(tmp_path / "immutable-a")
    root_b = str(tmp_path / "immutable-b")
    logical_a = "wandb-artifact:///entity/project/run:step1"
    logical_b = "wandb-artifact:///entity/project/run-teacher:step1"
    logical_c = "wandb-artifact:///entity/project/run-reference:step1"

    await trainer.prefetch_checkpoints(
        MaterializedCheckpoint(logical_a, root_a),
        MaterializedCheckpoint(logical_b, root_a),
    )
    trainer.load_checkpoint(MaterializedCheckpoint(logical_a, root_a))
    trainer.load_checkpoint(MaterializedCheckpoint(logical_b, root_a))
    assert prepared == [trainer._checkpoint_source_key(root_a)]

    await trainer.prefetch_checkpoints(MaterializedCheckpoint(logical_c, root_b))
    trainer.load_checkpoint(MaterializedCheckpoint(logical_c, root_b))
    assert sorted(prepared) == sorted(
        (trainer._checkpoint_source_key(root_a), trainer._checkpoint_source_key(root_b))
    )
    assert [logical_path for logical_path, _source in installed] == [
        logical_a,
        logical_b,
        logical_c,
    ]
    assert set(trainer._checkpoint_slots) == {
        logical_a,
        logical_b,
        logical_c,
    }
    assert trainer._default_slot_ref == trainer._slot_ref(logical_c)
    for logical_path in (logical_a, logical_b, logical_c):
        request = ForwardInput(input_tokens=torch.tensor([1]), checkpoint=logical_path)
        assert trainer._resolve_slot_ref(request) == trainer._slot_ref(logical_path)

    refreshed_root = str(tmp_path / "immutable-new")
    trainer.load_checkpoint(MaterializedCheckpoint(logical_a, refreshed_root))
    assert installed[-1][0] == logical_a
    assert prepared[-1] == trainer._checkpoint_source_key(refreshed_root)

    prepared_before_push = tuple(prepared)
    with trainer.push_checkpoint(
        MaterializedCheckpoint(logical_a, str(tmp_path / "unused-while-loaded"))
    ):
        assert tuple(prepared) == prepared_before_push
        assert trainer._slot_stack == [trainer._slot_ref(logical_a)]


async def test_prefetch_does_not_silently_ignore_empty_materialized_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    seen: list[str] = []

    def prepare(path: str) -> object:
        seen.append(path)
        return object()

    monkeypatch.setattr("art.trainer_rank._checkpoint.prepare_checkpoint", prepare)
    await trainer.prefetch_checkpoints(MaterializedCheckpoint("logical", ""))
    assert seen == [trainer._checkpoint_source_key("")]


def test_checkpoint_mutations_are_synchronous_and_recover_from_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    installed: list[str] = []

    def prepare(_path: str) -> object:
        return object()

    def install(trainer: TrainerRank, _source: object, path: str) -> None:
        installed.append(path)
        if path == "bad":
            raise RuntimeError("injected load failure")
        trainer._checkpoint_slots.setdefault(path, _CheckpointSlot()).params = ()
        trainer._checkpoint_slots[path].revision = 0

    monkeypatch.setattr("art.trainer_rank._checkpoint.prepare_checkpoint", prepare)
    monkeypatch.setattr("art.trainer_rank._checkpoint.load_checkpoint", install)
    trainer.load_checkpoint("first")
    trainer.load_checkpoint("second")
    assert installed == ["first", "second"]

    with pytest.raises(RuntimeError, match="injected load failure"):
        trainer.load_checkpoint("bad")
    trainer.load_checkpoint("after")
    assert installed[-2:] == ["bad", "after"]


def test_checkpoint_prefetch_failure_propagates_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())

    def prepare(_path: str) -> object:
        raise OSError("rank-local prefetch failed")

    monkeypatch.setattr("art.trainer_rank._checkpoint.prepare_checkpoint", prepare)
    with pytest.raises(OSError, match="rank-local prefetch failed"):
        trainer.load_checkpoint("student")


def test_remote_checkpoint_prefetch_failures_are_coordinated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())

    def coordinated(
        error: BaseException | None, phase: str, _group: object | None = None
    ) -> None:
        assert phase == "prepare checkpoint"
        assert error is None
        raise RuntimeError("a rank failed to prepare checkpoint")

    monkeypatch.setattr(
        "art.trainer_rank._checkpoint.prepare_checkpoint", lambda _path: object()
    )
    monkeypatch.setattr("art.trainer_rank._checkpoint.raise_distributed", coordinated)
    with pytest.raises(RuntimeError, match="a rank failed"):
        trainer.load_checkpoint("student")


def test_trainer_rank_rejects_adapter_keys_without_installed_lora_site() -> None:
    trainer = TrainerRank(_runtime(_FakeLoRASite("base.layer")))
    valid = {
        "base.layer.lora_A.weight": torch.empty(1),
        "base.layer.lora_B.weight": torch.empty(1),
    }
    trainer._prepare_adapter_model("student", valid)

    with pytest.raises(ValueError, match="matching LoRA target modules"):
        trainer._prepare_adapter_model(
            "student",
            {**valid, "base.other.lora_A.weight": torch.empty(1)},
        )


def test_trainer_rank_normalizes_adapter_tensors_to_installed_site() -> None:
    site = _FakeLoRASite("base.layer", dtype=torch.bfloat16)
    trainer = TrainerRank(_runtime(site))
    adapter = {
        "base.layer.lora_A.weight": torch.ones(3, 4, dtype=torch.float32),
        "base.layer.lora_B.weight": torch.ones(5, 3, dtype=torch.float32),
    }

    normalized = trainer._prepare_adapter_model("student", adapter)

    assert all(tensor.device == site.A_T.device for tensor in normalized.values())
    assert all(tensor.dtype == torch.bfloat16 for tensor in normalized.values())


def test_checkpoint_slot_adapter_config_is_validated_and_copied() -> None:
    trainer = TrainerRank(_runtime())
    config = {
        "base_model_name_or_path": "Qwen/Qwen3-8B",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj"],
    }

    retained = trainer._validate_checkpoint_adapter_config("student", config, alpha=16)

    assert retained == config
    config["target_modules"].append("v_proj")  # type: ignore[union-attr]
    assert retained is not None
    assert retained["target_modules"] == ["q_proj"]
    with pytest.raises(ValueError, match="conflicts"):
        trainer._validate_checkpoint_adapter_config("student", config, alpha=32)
    with pytest.raises(ValueError, match="missing"):
        trainer._validate_checkpoint_adapter_config("student", {"r": 8}, alpha=None)


@pytest.mark.parametrize(
    "base_model",
    ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B"),
)
def test_qwen35_checkpoint_adapter_config_captures_attention_dimensions(
    base_model: str,
) -> None:
    runtime = _runtime()
    runtime.provider.num_attention_heads = 16
    runtime.provider.num_query_groups = 4
    runtime.provider.kv_channels = 128
    trainer = TrainerRank(runtime)

    retained = trainer._validate_checkpoint_adapter_config(
        "student",
        {
            "base_model_name_or_path": base_model,
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj"],
        },
        alpha=16,
    )

    assert retained is not None
    assert retained["num_attention_heads"] == 16
    assert retained["num_key_value_heads"] == 4
    assert retained["head_dim"] == 128
    assert retained["hidden_size"] == 4


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("base_model_name_or_path", 1),
        ("r", "8"),
        ("lora_alpha", "16"),
        ("target_modules", 1),
    ),
)
def test_checkpoint_slot_adapter_config_rejects_invalid_field_types(
    field: str,
    value: object,
) -> None:
    trainer = TrainerRank(_runtime())
    config: dict[str, object] = {
        "base_model_name_or_path": "Qwen/Qwen3-8B",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj"],
    }
    config[field] = value

    with pytest.raises(TypeError, match=field):
        trainer._validate_checkpoint_adapter_config("student", config, alpha=None)


def test_checkpoint_slot_adapter_config_rejects_cross_rank_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    monkeypatch.setattr("art.trainer_rank.dist.is_initialized", lambda: True)
    monkeypatch.setattr("art.trainer_rank.dist.get_world_size", lambda: 2)

    checkpoint_group = cast(dist.ProcessGroup, object())
    trainer._checkpoint_process_group = checkpoint_group
    trainer._checkpoint_finalize_process_group = cast(dist.ProcessGroup, object())

    def gather(
        output: list[object], value: object, *, group: object | None = None
    ) -> None:
        assert group is checkpoint_group
        revision = value[1] if isinstance(value, tuple) and len(value) == 2 else None
        output[:] = [value, ({"different": True}, revision)]

    monkeypatch.setattr("art.trainer_rank.dist.all_gather_object", gather)

    with pytest.raises(ValueError, match="differs across ranks"):
        trainer._validate_checkpoint_adapter_config("student", None, alpha=None)


@pytest.mark.skipif(find_spec("megatron") is None, reason="requires Megatron")
def test_slot_load_canonicalizes_only_local_incoming_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[dict[str, torch.Tensor], object]] = []
    loaded_state: dict[str, torch.Tensor] = {}
    runtime = _runtime()
    monkeypatch.setattr(
        runtime.model_support_handler,
        "canonicalize_loaded_lora_state",
        lambda state, model: (
            calls.append((state, model))
            or {key: torch.zeros_like(value) for key, value in state.items()}
        ),
    )
    monkeypatch.setattr(
        runtime.model_support_handler,
        "zero_internal_padding_params",
        lambda _model: pytest.fail(
            "slot load must not mutate unrelated slot parameters"
        ),
    )
    trainer = TrainerRank(runtime)
    monkeypatch.setattr(
        trainer, "_local_lora_adapter_templates", lambda: {"weight": torch.empty(1)}
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    checkpoint_group = cast(dist.ProcessGroup, object())
    trainer._checkpoint_process_group = checkpoint_group
    trainer._checkpoint_finalize_process_group = cast(dist.ProcessGroup, object())

    def gather_expected(
        values: list[set[str] | None],
        local: set[str],
        *,
        group: object | None = None,
    ) -> None:
        assert group is checkpoint_group
        values[:] = [local, {"remote_weight"}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_expected)
    monkeypatch.setattr(trainer, "_guard_slot_can_load", lambda *_: None)

    def load_slot(
        _model: object,
        _ref: object,
        adapter_model: dict[str, torch.Tensor],
        **_kwargs: object,
    ) -> int:
        loaded_state.update(adapter_model)
        return 1

    monkeypatch.setattr(
        "art.megatron.lora.load_lora_slot_into_model",
        load_slot,
    )

    adapter = {"weight": torch.ones(1), "remote_weight": torch.ones(1)}
    trainer._load_checkpoint_slot("student", adapter, alpha=1.0)

    assert calls == [({"weight": adapter["weight"]}, runtime.model)]
    torch.testing.assert_close(loaded_state["weight"], torch.zeros(1))
    assert "remote_weight" not in loaded_state
    torch.testing.assert_close(adapter["weight"], torch.ones(1))


@pytest.mark.skipif(find_spec("megatron") is None, reason="requires Megatron")
def test_checkpoint_slot_snapshot_preserves_loaded_slot_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron import lora as lora_module
    from art.megatron.lora import LoRA, LoRASlotRef
    from art.trainer_rank._checkpoint import _slot_snapshot

    monkeypatch.setattr(lora_module.ps, "get_expert_model_parallel_rank", lambda: 0)
    lora = LoRA("layer", 3, 4, 2, 32, torch.float32, torch.device("cpu"))
    ref = LoRASlotRef("checkpoint", "student")
    assert lora.load_lora_slot(
        ref,
        {
            "layer.lora_A.weight": torch.randn(2, 3),
            "layer.lora_B.weight": torch.randn(4, 2),
        },
        requires_grad=True,
    )

    snapshot = _slot_snapshot(TrainerRank(_runtime(lora)))

    assert snapshot[0][3] == {"slot_0": ref}


@pytest.mark.skipif(find_spec("megatron") is None, reason="requires Megatron")
def test_forward_snapshot_is_independent_and_forward_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron import lora as lora_module
    from art.megatron.lora import LoRA, LoRASlotRef

    monkeypatch.setattr(lora_module.ps, "get_expert_model_parallel_rank", lambda: 0)
    lora = LoRA("layer", 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    trainer = TrainerRank(_runtime(lora))
    source = LoRASlotRef("checkpoint", "student")
    assert lora.load_lora_slot(
        source,
        {
            "layer.lora_A.weight": torch.randn(2, 3),
            "layer.lora_B.weight": torch.randn(4, 2),
        },
        alpha=2,
        requires_grad=True,
    )
    trainer._checkpoint_slots["student"] = _CheckpointSlot(
        tuple(lora.lora_slot_params(source)),
        cast(
            Any,
            {
                "base_model_name_or_path": "test/model",
                "r": 2,
                "lora_alpha": 2,
                "target_modules": ["q_proj"],
            },
        ),
    )

    assert trainer.snapshot_checkpoint("student", "saved")
    assert not trainer.snapshot_checkpoint("student", "saved")
    saved = LoRASlotRef("checkpoint", "saved")
    before = tuple(param.detach().clone() for param in lora.lora_slot_params(saved))
    with torch.no_grad():
        for param in lora.lora_slot_params(source):
            param.add_(1)
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(before, lora.lora_slot_params(saved), strict=True)
    )
    assert all(not param.requires_grad for param in lora.lora_slot_params(saved))
    request = ForwardInput(
        input_tokens=torch.tensor([1]),
        target_tokens=torch.tensor([1]),
        checkpoint="saved",
    )
    assert trainer._resolve_slot_ref(request) == saved
    with pytest.raises(TrainerRankSlotStateError, match="forward-only"):
        trainer.optim_step(params=AdamParams(learning_rate=1e-3), checkpoints=["saved"])
    with pytest.raises(TrainerRankSlotStateError, match="load over forward-only"):
        trainer._guard_slot_can_load(saved)

    trainer._discard_snapshot_checkpoint("saved")
    assert "saved" not in trainer._checkpoint_slots
    assert lora._slot(saved) is None


def test_prepared_snapshot_loads_forward_only_without_replacing_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    source = cast(Any, object())
    calls: list[tuple[object, str, bool]] = []

    def load(
        target: TrainerRank,
        prepared: object,
        destination: str,
        *,
        forward_only: bool = False,
    ) -> None:
        calls.append((prepared, destination, forward_only))
        target._checkpoint_slots[destination] = _CheckpointSlot(snapshot=forward_only)

    monkeypatch.setattr("art.trainer_rank._checkpoint.load_checkpoint", load)
    assert snapshot_prepared_checkpoint(trainer, source, "saved")
    assert calls == [(source, "saved", True)]
    assert not snapshot_prepared_checkpoint(trainer, source, "saved")
    trainer._checkpoint_slots["loaded"] = _CheckpointSlot()
    with pytest.raises(TrainerRankSlotStateError, match="already loaded"):
        snapshot_prepared_checkpoint(trainer, source, "loaded")


def test_checkpoint_export_requires_retained_adapter_config() -> None:
    trainer = TrainerRank(_runtime())
    with pytest.raises(TrainerRankSlotStateError, match="unloaded checkpoint"):
        trainer.export_lora("/unused", "missing")

    trainer._checkpoint_slots.setdefault("student", _CheckpointSlot()).params = ()
    with pytest.raises(TrainerRankSlotStateError, match="adapter_config"):
        trainer.export_lora("/unused", "student")


def test_prepared_lora_export_lifecycle_without_megatron(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.trainer_rank import _lora_export

    snapshot = object()
    captures: list[object] = []
    saved: list[tuple[str, object]] = []

    def capture(*_args: object, **_kwargs: object) -> tuple[object, dict[str, float]]:
        captures.append(snapshot)
        return snapshot, {"d2h": 1.0}

    monkeypatch.setattr(_lora_export, "_capture_lora_publish_inputs", capture)
    monkeypatch.setattr(
        _lora_export,
        "_save_lora_publish_inputs",
        lambda output, value: saved.append((output, value)) or {"serialize": 2.0},
    )
    trainer = TrainerRank(_runtime())
    trainer._checkpoint_slots["student"] = _CheckpointSlot(
        config={
            "base_model_name_or_path": "test",
            "r": 1,
            "lora_alpha": 1,
            "target_modules": [],
        },
        revision=7,
    )

    revision, timings = trainer._prepare_lora_export(
        "publish", "student", owner_id="owner"
    )
    assert revision == 7
    assert timings["d2h"] == 1.0
    assert timings["slot_validation"] >= 0
    with pytest.raises(RuntimeError, match="already prepared"):
        trainer._prepare_lora_export("publish", "student", owner_id="other")
    trainer._abort_lora_export("publish", owner_id="other")
    assert captures == [snapshot]
    assert trainer._finish_lora_export("publish", "/output", owner_id="owner") == {
        "serialize": 2.0
    }
    assert saved == [("/output", snapshot)]
    with pytest.raises(ValueError, match="Unknown prepared LoRA export"):
        trainer._finish_lora_export("publish", "/output", owner_id="owner")

    trainer._prepare_lora_export("aborted", "student", owner_id="owner")
    trainer._abort_lora_export("aborted", owner_id="owner")
    with pytest.raises(ValueError, match="Unknown prepared LoRA export"):
        trainer._finish_lora_export("aborted", "/output", owner_id="owner")

    assert trainer.export_lora("/legacy", "student") == 7
    assert saved[-1] == ("/legacy", snapshot)
    assert not trainer._prepared_lora_exports


def test_checkpoint_save_rejects_accumulated_gradients() -> None:
    trainer = TrainerRank(_runtime())
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.ones_like(parameter)
    trainer._checkpoint_slots.setdefault("student", _CheckpointSlot()).params = (
        parameter,
    )
    trainer._checkpoint_slots["student"].config = {
        "base_model_name_or_path": "test",
        "r": 1,
        "lora_alpha": 1,
        "target_modules": [],
    }

    with pytest.raises(TrainerRankSlotStateError, match="accumulated gradients"):
        _validate_save_state(trainer, "student")


def _canonical_checkpoint(
    root: Path, *, with_optimizer: bool = True
) -> CheckpointManifest:
    from safetensors.torch import save_file

    root.mkdir()
    config = {
        "base_model_name_or_path": "test/model",
        "r": 1,
        "lora_alpha": 1,
        "target_modules": ["q_proj"],
        "art_lora_format": "art-trainer-rank-v1",
    }
    (root / "adapter_config.json").write_text(json.dumps(config))
    key = "layer.q_proj.lora_A.weight"
    save_file({key: torch.ones(1, 2)}, root / "adapter_model.safetensors")
    files = []
    if with_optimizer:
        (root / "optimizer").mkdir()
        for component in ("master", "exp_avg", "exp_avg_sq"):
            relative = f"optimizer/{component}.safetensors"
            save_file({key: torch.ones(1, 2)}, root / relative)
            files.append(relative)
    manifest: CheckpointManifest = {
        "format_version": 1,
        "base_model_name_or_path": "test/model",
        "optimizer": (
            OptimizerConfig(
                learning_rate=1e-3,
                beta1=0.9,
                beta2=0.99,
                eps=1e-8,
                weight_decay=0.1,
            )
            if with_optimizer
            else None
        ),
        "parameters": {key: files} if with_optimizer else {},
        "steps": {key: 3.0} if with_optimizer else {},
        "files": {},
        "digest": "",
    }
    payloads = {"adapter_config.json", "adapter_model.safetensors", *files}
    manifest["files"] = {
        relative: _file_digest(root / relative) for relative in payloads
    }
    manifest["digest"] = _manifest_digest(manifest)
    (root / "checkpoint.json").write_text(json.dumps(manifest))
    return manifest


def test_weights_only_checkpoint_validation_is_canonical(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    manifest = _canonical_checkpoint(root, with_optimizer=False)

    assert validate_checkpoint(root) == manifest
    with pytest.raises(RuntimeError, match="does not contain optimizer state"):
        validate_checkpoint(root, require_optimizer=True)


def test_materialize_checkpoint_preserves_validated_state(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = _canonical_checkpoint(source)
    output = tmp_path / "output"

    assert materialize_checkpoint(source, output) == manifest["digest"]
    assert validate_checkpoint(output) == manifest


def test_weights_only_load_replaces_stale_optimizer_and_recreates_it_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from art.trainer_rank import _checkpoint as checkpoint_module

    root = tmp_path / "checkpoint"
    _canonical_checkpoint(root, with_optimizer=False)
    source = prepare_checkpoint(str(root))
    assert source.manifest is not None and source.manifest["optimizer"] is None

    trainer = TrainerRank(_runtime())
    config = cast(Any, source.config)
    old = torch.nn.Parameter(torch.zeros(1, 2))
    trainer._checkpoint_slots["student"] = _CheckpointSlot((old,), config)
    stale = trainer._dynamic_optimizer("student", AdamParams(learning_rate=1e-3))
    replacement = torch.nn.Parameter(torch.ones(1, 2))
    key = source.keys[0]
    monkeypatch.setattr(
        trainer, "_local_lora_adapter_templates", lambda: {key: replacement}
    )
    monkeypatch.setattr(trainer, "_load_checkpoint_slot", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        trainer,
        "_validate_checkpoint_consistency",
        lambda *_args: (replacement,),
    )
    monkeypatch.setattr(trainer, "_validate_loaded_checkpoint_config", lambda *_: None)
    monkeypatch.setattr(checkpoint_module, "_slot_snapshot", lambda _trainer: ())
    monkeypatch.setattr(checkpoint_module, "_commit_slot", lambda *_args: None)
    monkeypatch.setattr(
        checkpoint_module,
        "_optimizer_state",
        lambda *_args: pytest.fail("weights-only load read optimizer state"),
    )

    checkpoint_module.load_checkpoint(trainer, source, "student")

    slot = trainer._checkpoint_slots["student"]
    assert slot.params == (replacement,)
    assert slot.optimizer is None
    assert stale is not slot.optimizer

    replacement.grad = torch.ones_like(replacement)
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(item.grad.float() for item in params),
    )
    result = trainer.optim_step(
        checkpoints=["student"],
        params=AdamParams(learning_rate=1e-3, weight_decay=0),
    )

    assert result["update_successful"] == 1
    assert slot.optimizer is not None
    assert slot.optimizer is not stale


def test_checkpoint_snapshot_handles_existing_slot_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LoRA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            slot = torch.nn.Linear(1, 1)
            setattr(slot, "ref", object())
            self._slot_keys = {slot.ref: "slot_0"}  # type: ignore[attr-defined]
            self._slot_modules = torch.nn.ModuleDict({"slot_0": slot})

    module = ModuleType("art.megatron.lora")
    setattr(module, "LoRA", LoRA)
    setattr(module, "LoRASlotRef", object)
    monkeypatch.setitem(__import__("sys").modules, "art.megatron.lora", module)
    lora = LoRA()

    snapshot = _slot_snapshot(TrainerRank(_runtime(lora)))

    assert len(snapshot) == 1
    assert snapshot[0][3] == {"slot_0": lora._slot_modules["slot_0"].ref}  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda manifest: manifest["steps"].update({next(iter(manifest["steps"])): 4.0}),
        lambda manifest: manifest["optimizer"].update({"eps": 1e-6}),  # type: ignore[union-attr]
        lambda manifest: manifest["parameters"].update(
            {next(iter(manifest["parameters"])): ("../bad", "x", "y")}
        ),
    ),
)
def test_checkpoint_manifest_semantics_are_authenticated(
    tmp_path: Path, mutate: object
) -> None:
    root = tmp_path / "checkpoint"
    manifest = _canonical_checkpoint(root)
    cast(Any, mutate)(manifest)
    (root / "checkpoint.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="digest mismatch|Unsafe checkpoint"):
        prepare_checkpoint(str(root))


@pytest.mark.parametrize("extra", (True, False))
def test_checkpoint_optimizer_mapping_must_match_adapter(
    tmp_path: Path, extra: bool
) -> None:
    root = tmp_path / "checkpoint"
    manifest = _canonical_checkpoint(root)
    if extra:
        manifest["parameters"]["unexpected"] = next(
            iter(manifest["parameters"].values())
        )
        manifest["steps"]["unexpected"] = 0
    else:
        manifest["parameters"].pop(next(iter(manifest["parameters"])))
    (root / "checkpoint.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="mapping differs"):
        prepare_checkpoint(str(root))


def test_materialize_lora_validates_exact_artifact_without_optimizer_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    manifest = _canonical_checkpoint(source)
    local = tmp_path / "local"
    local.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors", "checkpoint.json"):
        (local / name).write_bytes((source / name).read_bytes())
    entries = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "checkpoint.json",
        *(file for files in manifest["parameters"].values() for file in files),
    }
    monkeypatch.setattr(
        "art.megatron.model_support.lora_disk.normalize_lora_checkpoint_to_vllm",
        lambda _path: None,
    )

    output = tmp_path / "output"
    materialize_lora(
        local,
        output,
        require_optimizer=True,
        artifact_entries=entries,
        expected_digest=manifest["digest"],
    )
    assert {path.name for path in output.iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }

    from safetensors.torch import save_file

    save_file(
        {next(iter(manifest["parameters"])): torch.zeros(1, 2)},
        local / "adapter_model.safetensors",
    )
    with pytest.raises(RuntimeError, match="file digest mismatch"):
        materialize_lora(
            local,
            tmp_path / "corrupt-output",
            require_optimizer=True,
            artifact_entries=entries,
            expected_digest=manifest["digest"],
        )

    manifest["files"]["adapter_model.safetensors"] = _file_digest(
        local / "adapter_model.safetensors"
    )
    (local / "checkpoint.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Checkpoint digest mismatch"):
        materialize_lora(
            local,
            tmp_path / "tampered-manifest-output",
            require_optimizer=True,
            artifact_entries=entries,
            expected_digest=manifest["digest"],
        )
    (local / "checkpoint.json").write_bytes((source / "checkpoint.json").read_bytes())

    with pytest.raises(RuntimeError, match="digest mismatch"):
        materialize_lora(
            local,
            tmp_path / "bad-digest",
            artifact_entries=entries,
            expected_digest="bad",
        )
    with pytest.raises(RuntimeError, match="missing entries"):
        materialize_lora(
            local,
            tmp_path / "missing-entry",
            require_optimizer=True,
            artifact_entries={"adapter_config.json", "adapter_model.safetensors"},
        )


def _save_state_trainer() -> TrainerRank:
    trainer = TrainerRank(_runtime())
    trainer._checkpoint_process_group = None
    return trainer


def _prepared_save(root: Path, sequence: int) -> _PreparedSave:
    snapshot = root / f"snapshot-{sequence}"
    reservation = root / f"reserved-{sequence}"
    snapshot.mkdir()
    reservation.mkdir()
    return _PreparedSave(
        sequence=sequence,
        snapshot=snapshot,
        reservation=reservation,
        destination=root / f"output-{sequence}",
        config={},
        shards=(),
        optimizer=None,
    )


def test_optimizer_shards_are_received_as_float32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_save(tmp_path, 0)
    metadata = [
        SimpleNamespace(
            key="weight",
            owner_rank=1,
            shape=(2, 3),
            dtype_name="bfloat16",
            manifest={"kind": "replicated"},
            block="block",
        )
    ]
    received: list[torch.dtype] = []
    monkeypatch.setattr("art.trainer_rank._checkpoint._rank", lambda: 0)
    monkeypatch.setattr(
        "art.trainer_rank._checkpoint.raise_distributed", lambda *_args: None
    )

    def recv(tensor: torch.Tensor, **_kwargs: object) -> None:
        received.append(tensor.dtype)

    monkeypatch.setattr(dist, "recv", recv)
    monkeypatch.setitem(
        sys.modules,
        "art.megatron.weights.lora_publish",
        SimpleNamespace(
            merge_sharded_adapter_entries=lambda entries: {
                key: values[0][1] for key, values in cast(dict, entries).items()
            }
        ),
    )

    merged = _merge_component(prepared, cast(Any, metadata), "master", None)

    assert received == [torch.float32]
    assert merged["weight"].dtype == torch.float32


def test_checkpoint_fifo_abort_and_failure_do_not_block_later_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _save_state_trainer()
    first = _prepared_save(tmp_path, 0)
    second = _prepared_save(tmp_path, 1)
    third = _prepared_save(tmp_path, 2)
    trainer._prepared_checkpoint_saves = {
        "first": first,
        "second": second,
        "third": third,
    }
    calls: list[int] = []

    def finalize(_trainer: TrainerRank, prepared: _PreparedSave) -> None:
        calls.append(prepared.sequence)
        if prepared.sequence == 1:
            raise RuntimeError("injected finalization failure")

    monkeypatch.setattr("art.trainer_rank._checkpoint._finish", finalize)
    abort_checkpoint_save(trainer, "first")
    with pytest.raises(RuntimeError, match="injected"):
        finish_checkpoint_save(trainer, "second")
    finish_checkpoint_save(trainer, "third")

    assert calls == [1, 2]
    assert trainer._checkpoint_save_next == 3


def test_checkpoint_out_of_order_finalization_fails_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _save_state_trainer()
    first = _prepared_save(tmp_path, 0)
    second = _prepared_save(tmp_path, 1)
    trainer._prepared_checkpoint_saves = {"first": first, "second": second}
    monkeypatch.setattr("art.trainer_rank._checkpoint._finish", lambda *_args: None)

    with pytest.raises(RuntimeError, match="finalized in preparation order"):
        finish_checkpoint_save(trainer, "second")
    finish_checkpoint_save(trainer, "first")
    finish_checkpoint_save(trainer, "second")

    assert trainer._checkpoint_save_next == 2


def test_concurrent_checkpoint_finish_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _save_state_trainer()
    prepared = _prepared_save(tmp_path, 0)
    trainer._prepared_checkpoint_saves = {"save": prepared}
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def finalize(_trainer: TrainerRank, _prepared: _PreparedSave) -> None:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=2)

    monkeypatch.setattr("art.trainer_rank._checkpoint._finish", finalize)
    errors: list[BaseException] = []

    def finish() -> None:
        try:
            finish_checkpoint_save(trainer, "save")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=finish) for _ in range(2)]
    threads[0].start()
    assert entered.wait(timeout=2)
    threads[1].start()
    time.sleep(0.05)
    release.set()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert not errors
    assert calls == 1


@pytest.mark.parametrize("action", ("finish", "abort"))
def test_checkpoint_cleanup_failure_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    from art.trainer_rank import _checkpoint

    trainer = _save_state_trainer()
    prepared = _prepared_save(tmp_path, 0)
    trainer._prepared_checkpoint_saves = {"save": prepared}
    finalizations = 0

    def finalize(_trainer: TrainerRank, _prepared: _PreparedSave) -> None:
        nonlocal finalizations
        finalizations += 1

    original = _checkpoint.shutil.rmtree
    failed = False

    def fail_once(path: Path, ignore_errors: bool = False, **_: object) -> None:
        nonlocal failed
        if Path(path) == prepared.snapshot and not failed:
            failed = True
            raise OSError("injected cleanup failure")
        original(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(_checkpoint, "_finish", finalize)
    monkeypatch.setattr(_checkpoint.shutil, "rmtree", fail_once)
    operation = finish_checkpoint_save if action == "finish" else abort_checkpoint_save
    with pytest.raises(BaseExceptionGroup, match="cleanup failed"):
        operation(trainer, "save")
    operation(trainer, "save")

    assert finalizations == (1 if action == "finish" else 0)
    assert "save" not in trainer._prepared_checkpoint_saves
    assert not prepared.snapshot.exists()
    assert not prepared.reservation.exists()


def test_checkpoint_cleanup_gather_failure_releases_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from art.trainer_rank import _checkpoint

    trainer = _save_state_trainer()
    prepared = _prepared_save(tmp_path, 0)
    trainer._prepared_checkpoint_saves = {"save": prepared}
    original = _checkpoint._gather
    failed = False

    def fail_once(
        value: object, group: dist.ProcessGroup | None = None
    ) -> tuple[object, ...]:
        nonlocal failed
        if isinstance(value, tuple) and len(value) == 2 and not failed:
            failed = True
            raise RuntimeError("injected cleanup gather failure")
        return original(value, group)

    monkeypatch.setattr(_checkpoint, "_finish", lambda *_: None)
    monkeypatch.setattr(_checkpoint, "_gather", fail_once)
    with pytest.raises(RuntimeError, match="cleanup gather"):
        finish_checkpoint_save(trainer, "save")
    assert "save" not in trainer._checkpoint_finalizing_saves
    finish_checkpoint_save(trainer, "save")
    assert "save" not in trainer._prepared_checkpoint_saves


def test_checkpoint_asymmetric_cleanup_gather_can_converge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from art.trainer_rank import _checkpoint

    completed = _save_state_trainer()
    retained = _save_state_trainer()
    retained_root = tmp_path / "retained"
    retained_root.mkdir()
    retained_save = _prepared_save(retained_root, 0)
    completed._finalized_checkpoint_saves["save"] = _FinalizedSave(0, "finish")
    retained._prepared_checkpoint_saves["save"] = retained_save
    retained._checkpoint_save_outcomes["save"] = "finish"

    def mixed(
        value: object, _group: dist.ProcessGroup | None = None
    ) -> tuple[object, ...]:
        if isinstance(value, bool):
            return (True, False)
        return (value, value)

    monkeypatch.setattr(_checkpoint, "_gather", mixed)
    finish_checkpoint_save(completed, "save")
    finish_checkpoint_save(retained, "save")
    assert "save" in completed._finalized_checkpoint_saves
    assert "save" in retained._finalized_checkpoint_saves
    assert "save" not in retained._prepared_checkpoint_saves


def test_checkpoint_cleanup_gather_preserves_finish_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from art.trainer_rank import _checkpoint

    trainer = _save_state_trainer()
    prepared = _prepared_save(tmp_path, 0)
    trainer._prepared_checkpoint_saves = {"save": prepared}
    original = _checkpoint._gather

    def fail_cleanup(
        value: object, group: dist.ProcessGroup | None = None
    ) -> tuple[object, ...]:
        if isinstance(value, tuple) and len(value) == 2:
            raise RuntimeError("cleanup collective failed")
        return original(value, group)

    def fail_finish(*_: object) -> None:
        raise ValueError("snapshot failed")

    monkeypatch.setattr(_checkpoint, "_finish", fail_finish)
    monkeypatch.setattr(
        _checkpoint, "_cleanup_paths", lambda *_: OSError("unlink failed")
    )
    monkeypatch.setattr(_checkpoint, "_gather", fail_cleanup)
    with pytest.raises(BaseExceptionGroup) as raised:
        finish_checkpoint_save(trainer, "save")
    assert any(isinstance(error, ValueError) for error in raised.value.exceptions)
    assert any(isinstance(error, OSError) for error in raised.value.exceptions)
    assert any(isinstance(error, RuntimeError) for error in raised.value.exceptions)


def test_checkpoint_prepare_preserves_foreign_reservation(tmp_path: Path) -> None:
    trainer = _save_state_trainer()
    trainer._checkpoint_slots["student"] = _CheckpointSlot(
        config={
            "base_model_name_or_path": "test",
            "r": 1,
            "lora_alpha": 1,
            "target_modules": [],
        }
    )
    output = tmp_path / "save"
    reservation = tmp_path / ".save.reserved"
    reservation.mkdir()
    marker = reservation / "owner"
    marker.write_text("foreign")

    with pytest.raises(FileExistsError):
        prepare_checkpoint_save(trainer, str(output), "student")

    assert marker.read_text() == "foreign"


def test_checkpoint_prepare_reports_snapshot_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from art.trainer_rank import _checkpoint

    trainer = _save_state_trainer()
    trainer._checkpoint_slots["student"] = _CheckpointSlot(
        config={
            "base_model_name_or_path": "test",
            "r": 1,
            "lora_alpha": 1,
            "target_modules": [],
        }
    )
    original = _checkpoint.shutil.rmtree

    def fail_snapshot(path: Path, ignore_errors: bool = False, **_: object) -> None:
        if ".snapshot-" in Path(path).name:
            raise OSError("injected cleanup failure")
        original(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(
        _checkpoint,
        "_local_state",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
    )
    monkeypatch.setattr(_checkpoint.shutil, "rmtree", fail_snapshot)

    with pytest.raises(BaseExceptionGroup) as captured:
        prepare_checkpoint_save(trainer, str(tmp_path / "save"), "student")
    messages = " ".join(str(error) for error in captured.value.exceptions)
    assert "snapshot failed" in messages
    assert "cleanup" in messages


def _checkpoint_load_failure_worker(
    rank: int, world_size: int, init_method: str, phase: str
) -> None:
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(seconds=15),
    )
    from art.trainer_rank import _checkpoint as checkpoint_module
    from art.trainer_rank import _lora_export as lora_export_module

    originals = (
        checkpoint_module._load_adapter,
        checkpoint_module._optimizer_state,
        checkpoint_module._commit_slot,
        checkpoint_module._slot_snapshot,
        checkpoint_module._restore_slots,
    )
    try:
        trainer = TrainerRank.__new__(TrainerRank)
        trainer.runtime = SimpleNamespace(
            model=[],
            model_identifier=None,
            model_support_spec=None,
            provider=SimpleNamespace(),
        )
        trainer._checkpoint_process_group = None
        trainer._checkpoint_slots = {}
        trainer._slot_stack = []
        trainer._local_lora_adapter_templates = lambda: {}  # type: ignore[method-assign]
        trainer._guard_slot_can_load = lambda _ref: None  # type: ignore[method-assign]
        trainer._load_checkpoint_slot = lambda *_args, **_kwargs: 1  # type: ignore[method-assign]
        trainer._validate_checkpoint_consistency = lambda *_args: ()  # type: ignore[method-assign]
        trainer._validate_loaded_checkpoint_config = lambda *_args: None  # type: ignore[method-assign]
        trainer._restore_canonical_optimizer = lambda *_args: cast(Any, object())  # type: ignore[method-assign]
        setattr(checkpoint_module, "_slot_snapshot", lambda *_args: ())
        setattr(checkpoint_module, "_restore_slots", lambda *_args: None)
        if phase == "export":
            if rank == 1:
                trainer._checkpoint_slots["student"] = _CheckpointSlot(
                    config={
                        "base_model_name_or_path": "test/model",
                        "r": 1,
                        "lora_alpha": 1,
                        "target_modules": [],
                    }
                )
            with pytest.raises((ValueError, RuntimeError), match="Unknown|Another"):
                lora_export_module.export_lora(trainer, "/unused", "student")
            completed = torch.tensor(1)
            dist.all_reduce(completed)
            assert completed.item() == world_size
            return

        optimizer = (
            OptimizerConfig(
                learning_rate=1e-3,
                beta1=0.9,
                beta2=0.99,
                eps=1e-8,
                weight_decay=0.1,
            )
            if phase == "optimizer"
            else None
        )
        manifest: CheckpointManifest | None = (
            {
                "format_version": 1,
                "base_model_name_or_path": "test/model",
                "optimizer": optimizer,
                "parameters": {},
                "steps": {},
                "files": {},
                "digest": "digest",
            }
            if phase != "read"
            else None
        )
        source = PreparedCheckpoint(
            Path("/unused"),
            {
                "base_model_name_or_path": "test/model",
                "r": 1,
                "lora_alpha": 1,
                "target_modules": [],
            },
            (),
            manifest,
            "digest",
        )

        setattr(
            checkpoint_module,
            "_load_adapter",
            (
                lambda *_args: (
                    (_ for _ in ()).throw(RuntimeError("injected snapshot read"))
                    if phase == "read" and rank == 1
                    else {}
                )
            ),
        )
        setattr(
            checkpoint_module,
            "_optimizer_state",
            (
                lambda *_args: (
                    (_ for _ in ()).throw(RuntimeError("injected optimizer read"))
                    if phase == "optimizer" and rank == 1
                    else LocalOptimizerState(
                        (), (), (), (), cast(OptimizerConfig, optimizer)
                    )
                )
            ),
        )
        setattr(
            checkpoint_module,
            "_commit_slot",
            (
                lambda *_args: (
                    (_ for _ in ()).throw(RuntimeError("injected rank-zero commit"))
                    if phase == "commit" and rank == 0
                    else None
                )
            ),
        )

        with pytest.raises(RuntimeError, match="injected|Another rank failed"):
            checkpoint_module.load_checkpoint(trainer, source, "student")
        assert "student" not in trainer._checkpoint_slots
        assert not any(
            name.startswith("__art_loading_") for name in trainer._checkpoint_slots
        )
        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == world_size
    finally:
        for name, value in zip(
            (
                "_load_adapter",
                "_optimizer_state",
                "_commit_slot",
                "_slot_snapshot",
                "_restore_slots",
            ),
            originals,
            strict=True,
        ):
            setattr(checkpoint_module, name, value)
        dist.destroy_process_group()


@pytest.mark.parametrize("phase", ("read", "optimizer", "commit", "export"))
def test_checkpoint_load_failure_is_collective_and_transactional(
    tmp_path: Path, phase: str
) -> None:
    context = mp.spawn(
        _checkpoint_load_failure_worker,
        args=(2, f"file://{tmp_path / f'load-{phase}'}", phase),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if context.join(timeout=1):
            return
    else:
        for process in context.processes:
            process.terminate()
        pytest.fail(f"collective checkpoint {phase} failure test hung")


@pytest.mark.skipif(find_spec("megatron") is None, reason="requires Megatron")
@pytest.mark.parametrize(
    "with_optimizer", (False, True), ids=("weights-only", "optimizer")
)
def test_real_checkpoint_codec_round_trips_with_optional_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_optimizer: bool
) -> None:
    from art.megatron import lora as lora_module
    from art.megatron.lora import LoRA
    from art.trainer_rank import _checkpoint as checkpoint_module

    monkeypatch.setattr(lora_module.ps, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        lora_module.ps,
        "get_data_parallel_rank",
        lambda **_kwargs: 0,
    )
    config = cast(
        Any,
        {
            "base_model_name_or_path": "test/model",
            "r": 2,
            "lora_alpha": 2,
            "target_modules": ["q_proj"],
        },
    )
    adapter = {
        "layer.q_proj.lora_A.weight": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "layer.q_proj.lora_B.weight": torch.tensor(
            [[0.2, 0.1], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        ),
    }
    adam = AdamParams(
        learning_rate=3e-4,
        beta1=0.8,
        beta2=0.95,
        weight_decay=0.1,
        grad_clip_norm=10,
    )

    def make_trainer() -> TrainerRank:
        lora = LoRA("layer.q_proj", 3, 4, 2, 2, torch.float32, torch.device("cpu"))
        trainer = TrainerRank(_runtime(lora))
        loaded = trainer._load_checkpoint_slot("student", adapter, alpha=2)
        params = trainer._validate_checkpoint_consistency(
            "student", loaded, set(adapter)
        )
        trainer._checkpoint_slots["student"] = _CheckpointSlot(params, config)
        monkeypatch.setattr(
            trainer,
            "_reduce_dynamic_grads",
            lambda params, **_kwargs: tuple(item.grad.float() for item in params),
        )
        return trainer

    original = make_trainer()
    if with_optimizer:
        for parameter in original._checkpoint_slots["student"].params:
            parameter.grad = torch.full_like(parameter, 0.25)
        original.optim_step(params=adam)
    output = tmp_path / "exact"
    original.save_checkpoint(str(output), "student")
    original.save_checkpoint(str(output), "student")
    assert not list(tmp_path.glob(".exact.snapshot-*"))
    assert not (tmp_path / ".exact.reserved").exists()
    prepared = prepare_checkpoint(str(output))
    assert prepared.manifest is not None
    assert validate_checkpoint(output) == prepared.manifest
    assert (prepared.manifest["optimizer"] is not None) is with_optimizer

    restored_lora = LoRA("layer.q_proj", 3, 4, 2, 2, torch.float32, torch.device("cpu"))
    restored = TrainerRank(_runtime(restored_lora))
    monkeypatch.setattr(
        restored,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(item.grad.float() for item in params),
    )
    checkpoint_module.load_checkpoint(restored, prepared, "student")
    assert (
        restored._checkpoint_slots["student"].optimizer is not None
    ) is with_optimizer

    for trainer in (original, restored):
        for parameter in trainer._checkpoint_slots["student"].params:
            parameter.grad = torch.full_like(parameter, -0.125)
        trainer.optim_step(params=adam)
    for actual, expected in zip(
        restored._checkpoint_slots["student"].params,
        original._checkpoint_slots["student"].params,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    restored_optimizer = restored._checkpoint_slots["student"].optimizer
    original_optimizer = original._checkpoint_slots["student"].optimizer
    assert restored_optimizer is not None and original_optimizer is not None
    _assert_nested_tensors_equal(
        restored_optimizer.optimizer.state_dict(),
        original_optimizer.optimizer.state_dict(),
    )
    with pytest.raises(FileExistsError, match="different state"):
        original.save_checkpoint(str(output), "student")
    assert not list(tmp_path.glob(".exact.snapshot-*"))
    assert not (tmp_path / ".exact.reserved").exists()


def test_trainer_rank_default_forward_uses_explicit_base_slot() -> None:
    trainer = TrainerRank(_runtime())

    plan = trainer._plan_flat_forward([_target_request(1)])

    assert len(plan.groups) == 1
    slot = plan.groups[0].slot_ref
    assert slot is not None
    assert getattr(slot, "name") is None


def test_optim_step_requires_loaded_checkpoint_slot() -> None:
    optimizer = _NativeOptimizer()
    trainer = TrainerRank(_runtime(optimizer=optimizer))

    with pytest.raises(TrainerRankSlotStateError, match="loaded checkpoint slot"):
        trainer.optim_step(params=AdamParams(learning_rate=1e-3))

    assert optimizer.step_calls == 0


def test_optim_step_rejects_loaded_slots_without_grads() -> None:
    trainer = TrainerRank(_runtime())
    trainer._checkpoint_slots.setdefault("student", _CheckpointSlot()).params = (
        torch.nn.Parameter(torch.ones(2)),
    )

    with pytest.raises(TrainerRankSlotStateError, match="none have gradients"):
        trainer.optim_step(params=AdamParams(learning_rate=1e-3))
    with pytest.raises(TrainerRankSlotStateError, match="no gradients"):
        trainer.optim_step(
            params=AdamParams(learning_rate=1e-3),
            checkpoints=["student"],
        )


def test_optim_step_rejects_explicit_slot_subset_with_missing_grads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    ready = torch.nn.Parameter(torch.ones(2))
    missing = torch.nn.Parameter(torch.ones(2))
    ready.grad = torch.ones_like(ready)
    trainer._checkpoint_slots.setdefault("ready", _CheckpointSlot()).params = (ready,)
    trainer._checkpoint_slots.setdefault("missing", _CheckpointSlot()).params = (
        missing,
    )
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(param.grad.float() for param in params),
    )

    with pytest.raises(TrainerRankSlotStateError, match="missing"):
        trainer.optim_step(
            params=AdamParams(learning_rate=1e-3),
            checkpoints=["ready", "missing"],
        )


def test_optim_step_implicitly_steps_only_slots_with_grads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    ready = torch.nn.Parameter(torch.ones(2))
    untouched = torch.nn.Parameter(torch.ones(2))
    ready.grad = torch.ones_like(ready)
    trainer._checkpoint_slots.setdefault("ready", _CheckpointSlot()).params = (ready,)
    trainer._checkpoint_slots.setdefault("untouched", _CheckpointSlot()).params = (
        untouched,
    )
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(param.grad.float() for param in params),
    )

    before_ready = ready.detach().clone()
    before_untouched = untouched.detach().clone()
    trainer.optim_step(
        params=AdamParams(learning_rate=1e-2, weight_decay=0.0, grad_clip_norm=10.0)
    )

    assert trainer._checkpoint_slots["ready"].optimizer is not None
    assert trainer._checkpoint_slots["untouched"].optimizer is None
    assert not torch.equal(before_ready, ready)
    torch.testing.assert_close(untouched, before_untouched)


def test_optim_step_accepts_per_checkpoint_params_and_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    scales: dict[str, float] = {}
    names: dict[int, str] = {}
    for name in ("policy", "adversary", "unselected"):
        param = torch.nn.Parameter(torch.ones(1))
        param.grad = torch.ones_like(param)
        trainer._checkpoint_slots[name] = _CheckpointSlot(params=(param,))
        names[id(param)] = name

    def reduce_grads(
        params: tuple[torch.nn.Parameter, ...], *, scale_grads: float
    ) -> tuple[torch.Tensor, ...]:
        name = names[id(params[0])]
        scales[name] = scale_grads
        assert all(param.grad is not None for param in params)
        return tuple(
            cast(torch.Tensor, param.grad).float().mul(scale_grads) for param in params
        )

    monkeypatch.setattr(trainer, "_reduce_dynamic_grads", reduce_grads)
    metrics = trainer.optim_step(
        params={
            "policy": AdamParams(
                learning_rate=1e-2, weight_decay=0.0, grad_clip_norm=0.0
            ),
            "adversary": AdamParams(
                learning_rate=2e-2, weight_decay=0.0, grad_clip_norm=0.0
            ),
        },
        scale_grads={"policy": 0.5, "adversary": 0.25},
    )

    assert scales == {"adversary": 0.25, "policy": 0.5}
    assert trainer._checkpoint_slots["unselected"].optimizer is None
    for name, learning_rate in (("policy", 1e-2), ("adversary", 2e-2)):
        optimizer = trainer._checkpoint_slots[name].optimizer
        assert optimizer is not None
        assert optimizer.optimizer.param_groups[0]["lr"] == learning_rate
        assert metrics[f"learning_rate/{name}"] == learning_rate
    assert "learning_rate" not in metrics


@pytest.mark.parametrize("mapped", (False, True))
def test_optim_step_clips_per_checkpoint(
    monkeypatch: pytest.MonkeyPatch, mapped: bool
) -> None:
    trainer = TrainerRank(_runtime())
    stepped: dict[str, torch.Tensor] = {}
    dynamics: dict[str, object] = {}

    class RecordingOptimizer:
        def __init__(self, name: str, master: torch.nn.Parameter) -> None:
            self.name = name
            self.master = master

        def step(self) -> None:
            assert self.master.grad is not None
            stepped[self.name] = self.master.grad.detach().clone()

        def zero_grad(self, *, set_to_none: bool = False) -> None:
            del set_to_none
            self.master.grad = None

    for name, grad in (("policy", 3.0), ("adversary", 4.0)):
        param = torch.nn.Parameter(torch.ones(1))
        param.grad = torch.full_like(param, grad)
        trainer._checkpoint_slots[name] = _CheckpointSlot(params=(param,))
        master = torch.nn.Parameter(param.detach().float().clone())
        dynamics[name] = SimpleNamespace(
            master_params=(master,), optimizer=RecordingOptimizer(name, master)
        )

    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(param.grad.float() for param in params),
    )
    monkeypatch.setattr(
        trainer, "_dynamic_optimizer", lambda name, _params: dynamics[name]
    )
    monkeypatch.setattr(trainer, "_prune_slot_graphs", lambda *_args: None)

    adam = AdamParams(learning_rate=1e-2, grad_clip_norm=1.0)
    params = (
        {
            "policy": adam,
            "adversary": AdamParams(learning_rate=1e-2, grad_clip_norm=2.0),
        }
        if mapped
        else adam
    )
    metrics = trainer.optim_step(
        params=params,
        checkpoints=None if mapped else ["policy", "adversary"],
    )

    torch.testing.assert_close(stepped["policy"], torch.ones(1))
    torch.testing.assert_close(
        stepped["adversary"], torch.full((1,), 2.0 if mapped else 1.0)
    )
    assert metrics["grad_norm"] == pytest.approx(5.0)
    assert metrics["learning_rate"] == 1e-2
    assert metrics["grad_norm/policy"] == pytest.approx(3.0)
    assert metrics["grad_norm/adversary"] == pytest.approx(4.0)


def test_optim_step_checks_all_checkpoint_grads_before_stepping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    for name, grad in (
        ("policy", 1.0),
        ("adversary", float("nan")),
        ("unselected", 1.0),
    ):
        param = torch.nn.Parameter(torch.ones(1))
        param.grad = torch.full_like(param, grad)
        trainer._checkpoint_slots[name] = _CheckpointSlot(params=(param,))
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(param.grad.float() for param in params),
    )
    monkeypatch.setattr(
        trainer,
        "_dynamic_optimizer",
        lambda *_args: pytest.fail("optimizer initialized before finite preflight"),
    )

    metrics = trainer.optim_step(
        params={
            name: AdamParams(learning_rate=1e-2) for name in ("policy", "adversary")
        }
    )

    assert metrics["update_successful"] == 0.0
    assert all(slot.revision == 0 for slot in trainer._checkpoint_slots.values())
    assert all(
        param.grad is None
        for name, slot in trainer._checkpoint_slots.items()
        if name != "unselected"
        for param in slot.params
    )
    assert trainer._checkpoint_slots["unselected"].params[0].grad is not None


def test_optim_step_requires_matching_checkpoint_configuration() -> None:
    trainer = TrainerRank(_runtime())

    with pytest.raises(ValueError, match="same checkpoint names"):
        trainer.optim_step(
            params={"policy": AdamParams(learning_rate=1e-3)},
            scale_grads={"adversary": 1.0},
        )
    with pytest.raises(ValueError, match="same checkpoint names"):
        trainer.optim_step(
            params={"policy": AdamParams(learning_rate=1e-3)},
            checkpoints=["adversary"],
        )


def test_optim_step_mapping_rejects_unready_checkpoints() -> None:
    trainer = TrainerRank(_runtime())
    trainer._checkpoint_slots["student"] = _CheckpointSlot(
        params=(torch.nn.Parameter(torch.ones(1)),)
    )

    with pytest.raises(TrainerRankSlotStateError, match="no gradients"):
        trainer.optim_step(params={"student": AdamParams(learning_rate=1e-3)})
    with pytest.raises(TrainerRankSlotStateError, match="unloaded checkpoint"):
        trainer.optim_step(params={"missing": AdamParams(learning_rate=1e-3)})


@pytest.mark.parametrize("mapped", ("params", "scale_grads"))
def test_optim_step_allows_either_configuration_to_be_mapped(
    monkeypatch: pytest.MonkeyPatch, mapped: str
) -> None:
    trainer = TrainerRank(_runtime())
    param = torch.nn.Parameter(torch.ones(1))
    param.grad = torch.ones_like(param)
    trainer._checkpoint_slots["student"] = _CheckpointSlot(params=(param,))
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(param.grad.float() for param in params),
    )
    adam = AdamParams(learning_rate=1e-3, weight_decay=0.0)

    trainer.optim_step(
        params={"student": adam} if mapped == "params" else adam,
        scale_grads={"student": 0.5} if mapped == "scale_grads" else 0.5,
    )

    assert trainer._checkpoint_slots["student"].revision == 1


def test_optim_step_prepares_all_optimizers_before_first_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    stepped = False
    for name in ("a", "b"):
        param = torch.nn.Parameter(torch.ones(1))
        param.grad = torch.ones_like(param)
        trainer._checkpoint_slots[name] = _CheckpointSlot(params=(param,))
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(param.grad.float() for param in params),
    )

    class RecordingOptimizer:
        def step(self) -> None:
            nonlocal stepped
            stepped = True

        def zero_grad(self, *, set_to_none: bool = False) -> None:
            del set_to_none

    dynamic = SimpleNamespace(
        master_params=(torch.nn.Parameter(torch.ones(1)),),
        optimizer=RecordingOptimizer(),
    )

    def optimizer(name: str, _params: AdamParams) -> object:
        if name == "b":
            raise RuntimeError("invalid optimizer state")
        return dynamic

    monkeypatch.setattr(trainer, "_dynamic_optimizer", optimizer)
    with pytest.raises(RuntimeError, match="invalid optimizer state"):
        trainer.optim_step(
            params={name: AdamParams(learning_rate=1e-3) for name in ("a", "b")}
        )

    assert not stepped
    assert all(slot.revision == 0 for slot in trainer._checkpoint_slots.values())
    assert all(slot.optimizer is None for slot in trainer._checkpoint_slots.values())


def test_optim_step_configuration_must_match_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda _group=None: 2)

    def gather(outputs: list[torch.Tensor], value: torch.Tensor) -> None:
        outputs[0].copy_(value)
        outputs[1].zero_()

    monkeypatch.setattr(dist, "all_gather", gather)

    with pytest.raises(TrainerRankSlotStateError, match="differ across ranks"):
        trainer._guard_optim_step_configuration(
            ["student"], AdamParams(learning_rate=1e-3), "allow"
        )


def test_optim_step_implicitly_ignores_resident_forward_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    student = torch.nn.Parameter(torch.ones(2))
    student.grad = torch.ones_like(student)
    snapshot = torch.nn.Parameter(torch.full((2,), 2.0), requires_grad=False)
    trainer._checkpoint_slots["student"] = _CheckpointSlot(params=(student,))
    trainer._checkpoint_slots["saved"] = _CheckpointSlot(
        params=(snapshot,), snapshot=True
    )
    monkeypatch.setattr(trainer, "_slot_ref", _slot_ref)
    trainer._set_default_slot(_slot_ref("student"))
    _stub_forward(monkeypatch, trainer, profiled=True)
    list(
        trainer.forward_micro_batches(
            [_target_request(1)], checkpoint="saved", no_grad=True
        )
    )
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(param.grad.float() for param in params),
    )

    before_student = student.detach().clone()
    before_snapshot = snapshot.detach().clone()
    trainer.optim_step(
        params=AdamParams(learning_rate=1e-2, weight_decay=0.0, grad_clip_norm=10.0)
    )

    assert not torch.equal(before_student, student)
    torch.testing.assert_close(snapshot, before_snapshot)
    assert trainer._checkpoint_slots["saved"].optimizer is None
    assert trainer._default_slot_ref == _slot_ref("student")
    with pytest.raises(TrainerRankSlotStateError, match="forward-only"):
        trainer.optim_step(params=AdamParams(learning_rate=1e-2), checkpoints=["saved"])
    del trainer._checkpoint_slots["student"]
    with pytest.raises(TrainerRankSlotStateError, match="trainable checkpoint"):
        trainer.optim_step(params=AdamParams(learning_rate=1e-2))


def test_dynamic_optimizer_zeroes_internal_padding_grads_before_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    param = torch.nn.Parameter(torch.tensor([1.0, 0.0]))
    param.grad = torch.ones_like(param)
    runtime = _runtime()

    def zero_padding_grads(_model: object) -> None:
        calls.append("grads")
        assert param.grad is not None
        param.grad[-1] = 0.0

    monkeypatch.setattr(
        runtime.model_support_handler,
        "zero_internal_padding_grads",
        zero_padding_grads,
    )
    monkeypatch.setattr(
        runtime.model_support_handler,
        "zero_internal_padding_params",
        lambda _model: pytest.fail(
            "slot step must not mutate unrelated slot parameters"
        ),
    )
    trainer = TrainerRank(runtime)
    trainer._checkpoint_slots.setdefault("student", _CheckpointSlot()).params = (param,)
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **_kwargs: tuple(item.grad.float() for item in params),
    )

    trainer.optim_step(
        params=AdamParams(
            learning_rate=1e-2,
            weight_decay=0.1,
            grad_clip_norm=10.0,
        )
    )

    assert calls == ["grads"]
    assert param[-1].item() == 0.0


def test_canonical_optimizer_state_reproduces_exact_next_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adam = AdamParams(
        learning_rate=3e-4,
        beta1=0.8,
        beta2=0.95,
        weight_decay=0.1,
        grad_clip_norm=10.0,
    )
    original, original_param = _trainer_with_checkpoint(
        monkeypatch, torch.tensor([0.5, -0.25], dtype=torch.bfloat16)
    )
    original._checkpoint_slots["student"].revision = 0
    original_param.grad = torch.tensor([0.2, -0.4], dtype=torch.bfloat16)
    original.optim_step(params=adam)
    dynamic = original._checkpoint_slots["student"].optimizer
    assert dynamic is not None
    optimizer_state = dynamic.optimizer.state[dynamic.master_params[0]]
    group = dynamic.optimizer.param_groups[0]
    beta1, beta2 = cast(tuple[float, float], group["betas"])
    state = LocalOptimizerState(
        masters=tuple(param.detach().clone() for param in dynamic.master_params),
        exp_avgs=(cast(torch.Tensor, optimizer_state["exp_avg"]).clone(),),
        exp_avg_sqs=(cast(torch.Tensor, optimizer_state["exp_avg_sq"]).clone(),),
        steps=(float(cast(torch.Tensor, optimizer_state["step"]).item()),),
        config=OptimizerConfig(
            learning_rate=float(group["lr"]),
            beta1=beta1,
            beta2=beta2,
            eps=float(group["eps"]),
            weight_decay=float(group["weight_decay"]),
        ),
    )

    restored, restored_param = _trainer_with_checkpoint(
        monkeypatch, original_param.detach()
    )
    restored._checkpoint_slots[
        "student"
    ].optimizer = restored._restore_canonical_optimizer("student", state)
    for param in (original_param, restored_param):
        param.grad = torch.tensor([-0.3, 0.1], dtype=torch.bfloat16)
    original.optim_step(params=adam)
    restored.optim_step(params=adam)

    torch.testing.assert_close(restored_param, original_param, atol=0, rtol=0)
    restored_optimizer = restored._checkpoint_slots["student"].optimizer
    original_optimizer = original._checkpoint_slots["student"].optimizer
    assert restored_optimizer is not None and original_optimizer is not None
    _assert_nested_tensors_equal(
        restored_optimizer.optimizer.state_dict(),
        original_optimizer.optimizer.state_dict(),
    )
    assert original._checkpoint_slots["student"].revision == 2


def test_dynamic_optimizer_keeps_fp32_master_weight_and_moments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, param = _trainer_with_checkpoint(
        monkeypatch, torch.tensor([0.1], dtype=torch.bfloat16)
    )

    for _ in range(100):
        param.grad = torch.ones_like(param)
        trainer.optim_step(
            params=AdamParams(
                learning_rate=1e-5,
                weight_decay=0.0,
                grad_clip_norm=10.0,
            )
        )

    dynamic = trainer._checkpoint_slots["student"].optimizer
    assert dynamic is not None
    assert dynamic.master_params[0].dtype == torch.float32
    assert param.item() < torch.tensor(0.1, dtype=torch.bfloat16).item()
    state = dynamic.optimizer.state[dynamic.master_params[0]]
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


def test_canonical_optimizer_rejects_incompatible_local_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _ = _trainer_with_checkpoint(monkeypatch, torch.ones(2))
    state = LocalOptimizerState(
        masters=(torch.ones(3),),
        exp_avgs=(torch.zeros(3),),
        exp_avg_sqs=(torch.zeros(3),),
        steps=(1.0,),
        config=OptimizerConfig(
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.99,
            eps=1e-8,
            weight_decay=0.0,
        ),
    )

    with pytest.raises(TrainerRankSlotStateError, match="master parameter shape"):
        trainer._restore_canonical_optimizer("student", state)


@pytest.mark.parametrize("operation", ("load", "step"))
def test_trainer_rank_rejects_mutating_slot_with_pending_graph(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    ref = _slot_ref("teacher")
    monkeypatch.setattr(trainer, "_slot_ref", _slot_ref)
    target = _tracked_targets(trainer, ref, 2)[0]
    guard = (
        (lambda: trainer._guard_slot_can_load(ref))
        if operation == "load"
        else (lambda: trainer._guard_checkpoint_can_step("teacher"))
    )

    with pytest.raises(TrainerRankSlotStateError, match="Cannot"):
        guard()

    target.sum().backward()
    guard()


def test_trainer_rank_step_allows_missing_slot_graph_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank.__new__(TrainerRank)
    monkeypatch.setattr(trainer, "_slot_ref", _slot_ref)

    trainer._guard_checkpoint_can_step("student")


def test_trainer_rank_zero_grad_does_not_clear_live_slot_graphs() -> None:
    trainer = TrainerRank(_runtime())
    ref = _slot_ref("teacher")
    output = ForwardOutput(
        None,
        TopK(
            torch.ones(1, requires_grad=True) * 2,
            torch.ones(1, dtype=torch.long),
        ),
        None,
        None,
    )

    tracked = trainer._track_slot_graph_outputs(ref, [output])
    trainer.zero_grad()

    assert tracked[0].top_k is not None
    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)


def test_trainer_rank_graph_tracking_does_not_copy_outputs() -> None:
    trainer = TrainerRank(_runtime())
    ref = _slot_ref("teacher")
    source = torch.ones(4, requires_grad=True) * 2
    output = ForwardOutput(source, None, None, None)

    tracked = trainer._track_slot_graph_outputs(ref, [output])[0]

    assert tracked.target_logprobs is not None
    assert tracked.target_logprobs.data_ptr() == source.data_ptr()


def test_trainer_rank_retained_backward_keeps_slot_graph_guard() -> None:
    trainer = TrainerRank(_runtime())
    ref = _slot_ref("teacher")
    target = _tracked_targets(trainer, ref, 2)[0]

    target.sum().backward(retain_graph=True)
    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)

    target.sum().backward()
    trainer._guard_slot_can_load(ref)


def test_trainer_rank_tracks_each_independent_output_graph() -> None:
    trainer = TrainerRank(_runtime())
    ref = _slot_ref("teacher")
    first, second = _tracked_targets(trainer, ref, 2, 3)

    first.sum().backward()
    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)

    second.sum().backward()
    trainer._guard_slot_can_load(ref)


def test_trainer_rank_tracks_graph_after_output_is_replaced_by_loss() -> None:
    trainer = TrainerRank(_runtime())
    ref = _slot_ref("teacher")
    target = _tracked_targets(trainer, ref, 2)[0]
    loss = target.sum()
    del target
    gc.collect()

    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)

    loss.backward()
    trainer._guard_slot_can_load(ref)


def test_trainer_rank_releases_abandoned_output_graph() -> None:
    trainer = TrainerRank(_runtime())
    ref = _slot_ref("teacher")
    target = _tracked_targets(trainer, ref, 2)[0]
    del target
    gc.collect()

    trainer._guard_slot_can_load(ref)


def _tracked_param_target(
    trainer: TrainerRank,
    ref: "LoRASlotRef",
    value: torch.Tensor,
) -> torch.Tensor:
    [tracked] = trainer._track_slot_graph_outputs(
        ref,
        [ForwardOutput(value, None, None, None)],
    )
    assert tracked.target_logprobs is not None
    return tracked.target_logprobs


def test_optim_step_allows_unused_live_graph_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, param = _trainer_with_checkpoint(monkeypatch, torch.tensor([2.0]))
    ref = trainer._slot_ref("student")
    # This models a trajectory with no sampled tokens: its differentiable forward
    # output is retained but intentionally contributes no loss or gradients.
    unused = _tracked_param_target(trainer, ref, param.square())
    used = _tracked_param_target(trainer, ref, param * 3.0)
    used.sum().backward()
    before = param.detach().clone()

    result = trainer.optim_step(params=AdamParams(learning_rate=1e-2, weight_decay=0.0))

    assert result["update_successful"] == 1.0
    assert not torch.equal(param, before)
    with pytest.raises(RuntimeError, match="modified by an inplace operation|version"):
        unused.sum().backward()


def test_optim_step_can_error_on_unused_live_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, param = _trainer_with_checkpoint(monkeypatch, torch.tensor([2.0]))
    ref = trainer._slot_ref("student")
    unused = _tracked_param_target(trainer, ref, param.square())
    used = _tracked_param_target(trainer, ref, param * 3.0)
    used.sum().backward()
    before = param.detach().clone()

    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer.optim_step(
            params=AdamParams(learning_rate=1e-2, weight_decay=0.0),
            on_live_graphs="error",
        )

    assert unused.grad_fn is not None
    torch.testing.assert_close(param, before)
    assert param.grad is not None


def test_optim_step_rejects_invalid_live_graph_policy() -> None:
    trainer = TrainerRank(_runtime())

    with pytest.raises(ValueError, match="on_live_graphs"):
        cast(Any, trainer).optim_step(
            params=AdamParams(learning_rate=1e-3),
            on_live_graphs="warn",
        )


def _live_graph_error_worker(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(seconds=30),
    )
    retained: torch.Tensor | None = None
    try:
        trainer = TrainerRank(_runtime())
        cast(Any, trainer)._slot_ref = _slot_ref
        param = torch.nn.Parameter(torch.tensor([2.0]))
        trainer._checkpoint_slots.setdefault("student", _CheckpointSlot()).params = (
            param,
        )
        param.grad = torch.ones_like(param)
        if rank == 0:
            retained = _tracked_param_target(
                trainer,
                trainer._slot_ref("student"),
                param.square(),
            )

        with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
            trainer.optim_step(
                params=AdamParams(learning_rate=1e-3),
                on_live_graphs="error",
            )

        completed = torch.tensor(1)
        dist.all_reduce(completed)
        assert completed.item() == world_size
        assert retained is None or retained.grad_fn is not None
    finally:
        dist.destroy_process_group()


def test_optim_step_live_graph_error_is_collective(tmp_path: Path) -> None:
    context = mp.spawn(
        _live_graph_error_worker,
        args=(2, f"file://{tmp_path / 'live-graph'}"),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if context.join(timeout=1):
            return
    for process in context.processes:
        process.terminate()
    pytest.fail("collective live-graph policy test hung")


def test_dp_rank_forward_preserves_nested_shape_for_inactive_requests() -> None:
    trainer = TrainerRank(_runtime())
    trainer._default_slot_ref = _slot_ref("teacher")
    request_a = ForwardInput(input_tokens=torch.tensor([1]))
    request_b = ForwardInput(input_tokens=torch.tensor([2]))

    outputs = trainer.dp_rank_forward([[request_a], [request_b]], no_grad=True)

    assert len(outputs) == 2
    assert len(outputs[0]) == 1
    assert outputs[0][0].target_logprobs is None
    assert outputs[1][0].target_logprobs is None
    assert outputs[0][0].checkpoint == "teacher"
    assert outputs[1][0].checkpoint == "teacher"
    assert outputs[0][0].no_grad
    assert outputs[1][0].no_grad
    assert not hasattr(trainer, "forward")
    assert not hasattr(trainer, "micro_batches")


def test_dp_rank_forward_supports_arbitrary_nested_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    _stub_forward(monkeypatch, trainer, _indexed_outputs)
    nested = [
        [[[[[_target_request(1)]]]]],
        [[[[[_target_request(3), _target_request(5)]]]]],
    ]

    outputs = cast(Any, trainer).dp_rank_forward(nested)

    assert _output_shape(outputs) == [
        [[[[["output"]]]]],
        [[[[["output", "output"]]]]],
    ]
    assert _output_values(outputs) == [0, 1, 2]


def test_forward_micro_batches_uses_deterministic_dp_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    _stub_forward(monkeypatch, trainer, dp=(1, 2))

    batches = list(
        trainer.forward_micro_batches([_target_request(i) for i in range(5)])
    )

    assert [batch.indices for batch in batches] == [(1,), (3,), ()]
    assert [len(batch.outputs) for batch in batches] == [1, 1, 0]


def test_forward_micro_batches_syncs_fit_decision_across_dp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    _stub_forward(monkeypatch, trainer, dp=(1, 2), profiled=True)
    sync_flags: list[bool] = []

    def memory_check(required: int, *, sync_across_dp: bool = False) -> _MemoryCheck:
        sync_flags.append(sync_across_dp)
        return _MemoryCheck(
            estimated_required_bytes=required,
            available_bytes=1 << 30,
            fits=True,
        )

    monkeypatch.setattr(trainer, "_memory_check_required", memory_check)
    next(iter(trainer.forward_micro_batches([_target_request(i) for i in range(6)])))

    assert sync_flags
    assert all(sync_flags)


def test_forward_micro_batches_supports_arbitrary_nested_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    _stub_forward(monkeypatch, trainer, _indexed_outputs, profiled=True)
    expected = [
        [[[[[_target_request(1)]]]]],
        [[[[[_target_request(3), _target_request(5)]]]]],
    ]
    nested = [(child for child in item) for item in expected]

    batches = list(cast(Any, trainer).forward_micro_batches(nested))

    assert batches[0].inputs == expected
    assert _output_shape(batches[0].outputs) == [
        [[[[["output"]]]]],
        [[[[["output", "output"]]]]],
    ]
    assert _output_values(batches[0].outputs) == [0, 1, 2]


def test_forward_micro_batches_ramps_after_first_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())

    def run(plan, **_kwargs):
        trainer._memory_profiles[plan.signature] = _MemoryProfile(
            bytes_per_token=0.0,
            packed_tokens=plan.packed_tokens,
        )
        return [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ]

    _stub_forward(monkeypatch, trainer, run)

    batches = list(
        trainer.forward_micro_batches([_target_request(i) for i in range(8)])
    )

    assert batches[0].stats.global_count == 1
    assert batches[0].stats.cold_start
    assert batches[1].stats.global_count > 1
    assert not batches[1].stats.cold_start


def test_forward_micro_batches_profiles_caller_peak_after_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    _stub_forward(monkeypatch, trainer, profiled=True)
    plan = trainer._plan_flat_forward([_target_request(1)])
    monkeypatch.setattr(
        trainer,
        "_run_flat_plan_with_memory_tracking",
        lambda *_args, **_kwargs: (_empty_outputs(plan), 123),
    )
    profiles: list[tuple[int, int | None]] = []
    monkeypatch.setattr(
        trainer,
        "_update_peak_memory_profile",
        lambda candidate, baseline: profiles.append(
            (candidate.packed_tokens, baseline)
        ),
    )

    batches = trainer.forward_micro_batches([_target_request(1)])
    next(batches)

    assert profiles == []
    with pytest.raises(StopIteration):
        next(batches)
    assert profiles == [(plan.packed_tokens, 123)]


def test_memory_profiles_distinguish_grad_mode() -> None:
    trainer = TrainerRank(_runtime())
    request = _target_request(1)

    grad_signature = trainer._plan_flat_forward([request]).signature
    with torch.no_grad():
        no_grad_signature = trainer._plan_flat_forward([request]).signature

    assert grad_signature.grad_enabled
    assert not no_grad_signature.grad_enabled
    assert grad_signature != no_grad_signature


def test_forward_micro_batches_does_not_overtrust_tiny_memory_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (0, 1))
    inputs = [_target_request(i) for i in range(64)]
    tiny_plan = trainer._plan_flat_forward([inputs[0]])
    trainer._memory_profiles[tiny_plan.signature] = _MemoryProfile(
        bytes_per_token=0.0,
        packed_tokens=tiny_plan.packed_tokens,
    )

    candidate = trainer._select_next_micro_batch(inputs, 0)

    assert candidate.stats_global_count == 8
    assert candidate.plan.packed_tokens == 16


def test_forward_micro_batches_tail_does_not_reset_stable_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    trainer._last_global_micro_batch_size = 64
    _stub_forward(monkeypatch, trainer, profiled=True)
    monkeypatch.setattr(
        trainer,
        "_estimate_required_memory_bytes_from_values",
        lambda **kwargs: kwargs["packed_tokens"],
    )
    monkeypatch.setattr(
        trainer,
        "_memory_check_required",
        lambda required, *, sync_across_dp=False: _MemoryCheck(
            estimated_required_bytes=required,
            available_bytes=128,
            fits=required <= 128,
        ),
    )
    batches = list(
        trainer.forward_micro_batches([_target_request(i) for i in range(130)])
    )

    assert [batch.stats.global_count for batch in batches] == [64, 64, 2]
    assert trainer._last_global_micro_batch_size == 64


def test_forward_micro_batches_raises_when_smallest_batch_will_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    monkeypatch.setattr(trainer, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(
        trainer,
        "_estimate_required_memory_bytes_from_values",
        lambda **_kwargs: 4,
    )
    monkeypatch.setattr(
        trainer,
        "_memory_check_required",
        lambda required, *, sync_across_dp=False: _MemoryCheck(
            estimated_required_bytes=required,
            available_bytes=3,
            fits=False,
        ),
    )
    with pytest.raises(TrainerRankMemoryError, match="smallest DP microbatch"):
        next(iter(trainer.forward_micro_batches([_target_request(1)])))


def test_forward_micro_batches_rejects_mismatched_replicated_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    import art.trainer_rank as trainer_rank

    monkeypatch.setattr(trainer_rank.dist, "is_available", lambda: True)
    monkeypatch.setattr(trainer_rank.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_rank.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(trainer_rank.dist, "all_reduce", lambda *_args, **_kwargs: None)

    def gather(output, value):
        output[:] = [value, value + 1]

    monkeypatch.setattr(trainer_rank.dist, "all_gather_object", gather)

    with pytest.raises(ValueError, match="same top-level input count"):
        list(trainer.forward_micro_batches([_target_request(1)]))

    monkeypatch.setattr(trainer_rank.dist, "is_initialized", lambda: False)
    _stub_forward(monkeypatch, trainer, dp=(1, 2))
    invalid = ForwardInput(
        input_tokens=torch.tensor([1, 2]), target_tokens=torch.tensor([1, 2, 3])
    )
    with pytest.raises(ValueError, match="target_tokens"):
        next(iter(trainer.forward_micro_batches([invalid, _target_request(1)])))


def test_forward_plan_estimates_output_memory_for_request_combo() -> None:
    class FakeGPT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                hidden_size=4,
                num_layers=1,
                padded_vocab_size=10,
            )
            self.decoder = object()

        def _preprocess(self, *args: object, **kwargs: object) -> None:
            return None

    trainer = TrainerRank(_runtime(FakeGPT()))
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    labels = torch.stack((tokens, tokens + 1), dim=-1)

    request = ForwardInput(
        input_tokens=tokens,
        target_tokens=labels,
        top_k=5,
        logits=True,
        hidden_states=True,
    )
    plan = trainer._plan_flat_forward([request])
    estimate = trainer._estimate_flat_forward([request])

    target_bytes = 3 * 2 * 4
    topk_bytes = 3 * 5 * (4 + 8)
    logits_bytes = 3 * 10 * 4
    hidden_bytes = 3 * 4 * 4
    assert estimate is not None and estimate[0] == plan.packed_tokens
    assert plan.output_bytes == target_bytes + topk_bytes + logits_bytes + hidden_bytes


def test_disconnected_outputs_keep_zero_graph_anchor() -> None:
    hidden = torch.randn(2, 3, requires_grad=True)
    disconnected = torch.zeros(4)
    top_k = TopK(logprobs=torch.zeros(4, 2), tokens=torch.ones(4, 2, dtype=torch.long))

    (anchored,), (anchored_top_k,) = _anchor_disconnected_outputs(
        [disconnected],
        [top_k],
        hidden,
    )

    assert anchored is not None
    assert anchored.requires_grad
    assert anchored_top_k is not None
    assert anchored_top_k.logprobs.requires_grad
    torch.testing.assert_close(anchored, disconnected)
    torch.testing.assert_close(anchored_top_k.logprobs, top_k.logprobs)
    (anchored.sum() + anchored_top_k.logprobs.sum()).backward()
    assert hidden.grad is not None
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))


def _assert_nested_tensors_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        actual_dict = cast(dict[Any, object], actual)
        expected_dict = cast(dict[Any, object], expected)
        for key in expected_dict:
            _assert_nested_tensors_equal(actual_dict[key], expected_dict[key])
    elif isinstance(expected, tuple | list):
        assert isinstance(actual, type(expected)) and len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_tensors_equal(actual_item, expected_item)
    else:
        assert actual == expected
