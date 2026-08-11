from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
import gc
from importlib.util import find_spec
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from art.megatron.prefix_tree_packing import prefix_tree_pack
from art.trainer_rank import (
    AdamParams,
    AdapterSelection,
    ForwardInput,
    ForwardOutput,
    TopK,
    TrainerRank,
    TrainerRankMemoryError,
    TrainerRankSlotStateError,
    Unset,
)
from art.trainer_rank._checkpoint import (
    AdamWRecord,
    LocalOptimizerState,
    _validate_save_state,
)
from art.trainer_rank._impl import (
    _anchor_disconnected_outputs,
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
    mp.setattr(rank, "_run_flat_plan_with_memory_tracking", out)
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
    trainer._checkpoint_slot_params_by_name["student"] = (param,)
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
    trainer._checkpoint_slot_params_by_name["student"] = ()
    _stub_forward(monkeypatch, trainer)
    request = ForwardInput(
        input_tokens=torch.tensor([1]),
        target_tokens=torch.tensor([1]),
        checkpoint=checkpoint,
    )

    output = trainer.dp_rank_forward([request])

    assert isinstance(output[0], ForwardOutput)


def test_forward_input_preserves_public_runtime_shape() -> None:
    fields = tuple(ForwardInput.__dataclass_fields__)
    assert tuple(inspect.signature(ForwardInput).parameters) == fields
    assert ForwardInput.__match_args__ == fields


@pytest.mark.parametrize("depth", (0, 2))
def test_trainer_rank_accepts_shared_prefix_depth(depth: int) -> None:
    trainer = TrainerRank(_runtime(), shared_prefix_max_depth=depth)

    assert trainer.shared_prefix_max_depth == depth


@pytest.mark.skipif(find_spec("megatron") is None, reason="requires Megatron")
def test_cp1_packed_forward_uses_model_attention_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron.context_parallel.types import ParallelTopology

    runtime = _runtime()
    trainer = TrainerRank(runtime)
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


async def test_trainer_rank_checkpoint_stack_errors() -> None:
    trainer = TrainerRank(_runtime())

    with pytest.raises(RuntimeError, match="No pushed checkpoint"):
        trainer.pop_checkpoint()
    trainer._slot_stack.append(object())  # type: ignore
    with pytest.raises(RuntimeError, match="Cannot load a checkpoint"):
        await trainer.load_checkpoint("teacher")


async def test_checkpoint_tasks_and_async_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    trainer._checkpoint_sources = {}
    fetched: list[str] = []

    async def prefetch(path: str) -> object:
        fetched.append(path)
        return object()

    def install(trainer: TrainerRank, _source: object, path: str) -> None:
        trainer._checkpoint_slot_params_by_name[path] = ()
        trainer._checkpoint_revisions[path] = 0

    monkeypatch.setattr(trainer, "_prefetch_checkpoint", prefetch)
    monkeypatch.setattr("art.trainer_rank._checkpoint.load_checkpoint", install)

    task = trainer.load_checkpoint("student")
    assert isinstance(task, asyncio.Task)
    await task
    assert fetched == ["student"]
    assert trainer._default_slot_ref == trainer._slot_ref("student")

    task = trainer.prefetch_checkpoints("teacher", "reference")
    assert isinstance(task, asyncio.Task)
    await task
    assert fetched[-2:] == ["teacher", "reference"]

    pushed = trainer.push_checkpoint("student")
    await pushed
    assert trainer._slot_stack == [trainer._slot_ref("student")]
    trainer.pop_checkpoint()
    async with trainer.push_checkpoint("student"):
        assert trainer._slot_stack == [trainer._slot_ref("student")]
        async with trainer.push_checkpoint("missing"):
            assert trainer._slot_stack == [
                trainer._slot_ref("student"),
                trainer._slot_ref("missing"),
            ]
        assert trainer._slot_stack == [trainer._slot_ref("student")]
    assert trainer._slot_stack == []
    assert fetched[-1] == "missing"


async def test_checkpoint_context_cancellation_after_successful_push_cleans_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    trainer._checkpoint_slot_params_by_name["student"] = ()
    parent = asyncio.current_task()
    assert parent is not None
    original_slot_ref = trainer._slot_ref
    cancellation_scheduled = False

    def cancel_parent_after_resolving(path: str | None):
        nonlocal cancellation_scheduled
        ref = original_slot_ref(path)
        if not cancellation_scheduled:
            cancellation_scheduled = True
            asyncio.get_running_loop().call_soon(parent.cancel)
        return ref

    monkeypatch.setattr(trainer, "_slot_ref", cancel_parent_after_resolving)
    pushed = trainer.push_checkpoint("student")
    entered = False

    with pytest.raises(asyncio.CancelledError):
        async with pushed:
            entered = True

    assert cancellation_scheduled
    assert pushed.task.done() and not pushed.task.cancelled()
    assert not entered
    assert trainer._slot_stack == []


async def test_shared_checkpoint_prefetch_survives_waiter_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    started = asyncio.Event()
    release = asyncio.Event()
    source = object()

    async def delayed_to_thread(_function: object, *_args: object) -> object:
        started.set()
        await release.wait()
        return source

    monkeypatch.setattr(asyncio, "to_thread", delayed_to_thread)
    first = asyncio.create_task(trainer._prefetch_checkpoint("student"))
    second = asyncio.create_task(trainer._prefetch_checkpoint("student"))
    await started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()

    assert await second is source
    assert (
        trainer._checkpoint_sources[trainer._checkpoint_source_key("student")] is source
    )
    assert trainer._checkpoint_prefetch_tasks == {}


async def test_shared_checkpoint_prefetch_serves_successful_waiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    started = asyncio.Event()
    release = asyncio.Event()
    source = object()
    calls = 0

    async def delayed_to_thread(_function: object, *_args: object) -> object:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return source

    monkeypatch.setattr(asyncio, "to_thread", delayed_to_thread)
    first = asyncio.create_task(trainer._prefetch_checkpoint("student"))
    second = asyncio.create_task(trainer._prefetch_checkpoint("student"))
    await started.wait()
    await asyncio.sleep(0)
    release.set()

    assert await asyncio.gather(first, second) == [source, source]
    assert calls == 1
    assert trainer._checkpoint_prefetch_tasks == {}


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
        trainer._checkpoint_slot_params_by_name[logical_path] = ()
        trainer._checkpoint_revisions[logical_path] = 0

    monkeypatch.setattr("art.trainer_rank._checkpoint.prepare_checkpoint", prepare)
    monkeypatch.setattr("art.trainer_rank._checkpoint.load_checkpoint", install)
    root_a = str(tmp_path / "immutable-a")
    root_b = str(tmp_path / "immutable-b")
    logical_a = "wandb-artifact:///entity/project/run:step1"
    logical_b = "wandb-artifact:///entity/project/run-teacher:step1"
    logical_c = "wandb-artifact:///entity/project/run-reference:step1"

    await asyncio.gather(
        trainer._load_checkpoint_from_source(logical_a, root_a),
        trainer._load_checkpoint_from_source(logical_b, root_a),
    )
    assert prepared == [trainer._checkpoint_source_key(root_a)]

    await trainer._prefetch_checkpoints_from_sources((logical_c, root_b))
    await trainer._load_checkpoint_from_source(logical_c, root_b)
    assert sorted(prepared) == sorted(
        (trainer._checkpoint_source_key(root_a), trainer._checkpoint_source_key(root_b))
    )
    assert [logical_path for logical_path, _source in installed] == [
        logical_a,
        logical_b,
        logical_c,
    ]
    assert set(trainer._checkpoint_slot_params_by_name) == {
        logical_a,
        logical_b,
        logical_c,
    }
    assert trainer._default_slot_ref == trainer._slot_ref(logical_c)
    for logical_path in (logical_a, logical_b, logical_c):
        request = ForwardInput(input_tokens=torch.tensor([1]), checkpoint=logical_path)
        assert trainer._resolve_slot_ref(request) == trainer._slot_ref(logical_path)

    refreshed_root = str(tmp_path / "immutable-new")
    await trainer._load_checkpoint_from_source(logical_a, refreshed_root)
    assert installed[-1][0] == logical_a
    assert prepared[-1] == trainer._checkpoint_source_key(refreshed_root)

    prepared_before_push = tuple(prepared)
    pushed = trainer._push_checkpoint_from_source(
        logical_a, str(tmp_path / "unused-while-loaded")
    )
    await pushed
    assert tuple(prepared) == prepared_before_push
    assert trainer._slot_stack == [trainer._slot_ref(logical_a)]
    trainer.pop_checkpoint()


async def test_checkpoint_mutations_follow_call_order_and_recover_from_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = TrainerRank(_runtime())
    trainer._checkpoint_sources = {}
    started = {name: asyncio.Event() for name in ("first", "second")}
    ready = {name: asyncio.Event() for name in ("first", "second")}
    installed: list[str] = []

    async def prefetch(path: str) -> object:
        event = started.get(path)
        if event is not None:
            event.set()
            await ready[path].wait()
        return object()

    def install(trainer: TrainerRank, _source: object, path: str) -> None:
        installed.append(path)
        if path == "bad":
            raise RuntimeError("injected load failure")
        trainer._checkpoint_slot_params_by_name[path] = ()
        trainer._checkpoint_revisions[path] = 0

    monkeypatch.setattr(trainer, "_prefetch_checkpoint", prefetch)
    monkeypatch.setattr("art.trainer_rank._checkpoint.load_checkpoint", install)
    first = trainer.load_checkpoint("first")
    second = trainer.load_checkpoint("second")
    await asyncio.gather(*(event.wait() for event in started.values()))
    ready["second"].set()
    await asyncio.sleep(0)
    assert installed == []
    ready["first"].set()
    await asyncio.gather(first, second)
    assert installed == ["first", "second"]

    bad = trainer.load_checkpoint("bad")
    after = trainer.load_checkpoint("after")
    with pytest.raises(RuntimeError, match="injected load failure"):
        await bad
    await after
    assert installed[-2:] == ["bad", "after"]


@pytest.mark.parametrize("local_failure", [False, True])
async def test_checkpoint_prefetch_failures_are_coordinated(
    monkeypatch: pytest.MonkeyPatch, local_failure: bool
) -> None:
    trainer = TrainerRank(_runtime())

    async def prefetch(_path: str) -> object:
        if local_failure:
            raise OSError("rank-local prefetch failed")
        return object()

    def coordinated(error: BaseException | None, phase: str) -> None:
        assert phase == "prepare checkpoint"
        assert isinstance(error, OSError) is local_failure
        raise RuntimeError("a rank failed to prepare checkpoint")

    monkeypatch.setattr(trainer, "_prefetch_checkpoint", prefetch)
    monkeypatch.setattr("art.trainer_rank._checkpoint._raise_distributed", coordinated)
    with pytest.raises(RuntimeError, match="a rank failed"):
        await trainer.load_checkpoint("student")


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

    def gather(output: list[object], value: object) -> None:
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

    def gather_expected(values: list[set[str] | None], local: set[str]) -> None:
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


def test_checkpoint_export_requires_retained_adapter_config() -> None:
    trainer = TrainerRank(_runtime())
    with pytest.raises(ValueError, match="Unknown checkpoint"):
        trainer.export_lora("/unused", "missing")

    trainer._checkpoint_slot_params_by_name["student"] = ()
    with pytest.raises(TrainerRankSlotStateError, match="adapter_config"):
        trainer.export_lora("/unused", "student")


def test_checkpoint_save_rejects_accumulated_gradients() -> None:
    trainer = TrainerRank(_runtime())
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.ones_like(parameter)
    trainer._checkpoint_slot_params_by_name["student"] = (parameter,)
    trainer._checkpoint_slot_adapter_configs["student"] = {
        "base_model_name_or_path": "test",
        "r": 1,
        "lora_alpha": 1,
        "target_modules": [],
    }

    with pytest.raises(TrainerRankSlotStateError, match="accumulated gradients"):
        _validate_save_state(trainer, "student")


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
    trainer._checkpoint_slot_params_by_name["student"] = (
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
    trainer._checkpoint_slot_params_by_name["ready"] = (ready,)
    trainer._checkpoint_slot_params_by_name["missing"] = (missing,)
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
    trainer._checkpoint_slot_params_by_name["ready"] = (ready,)
    trainer._checkpoint_slot_params_by_name["untouched"] = (untouched,)
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

    assert "ready" in trainer._dynamic_optimizers
    assert "untouched" not in trainer._dynamic_optimizers
    assert not torch.equal(before_ready, ready)
    torch.testing.assert_close(untouched, before_untouched)


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
    trainer._checkpoint_slot_params_by_name["student"] = (param,)
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
    original._checkpoint_revisions["student"] = 0
    original_param.grad = torch.tensor([0.2, -0.4], dtype=torch.bfloat16)
    original.optim_step(params=adam)
    dynamic = original._dynamic_optimizers["student"]
    optimizer_state = dynamic.optimizer.state[dynamic.master_params[0]]
    group = dynamic.optimizer.param_groups[0]
    beta1, beta2 = cast(tuple[float, float], group["betas"])
    state = LocalOptimizerState(
        masters=tuple(param.detach().clone() for param in dynamic.master_params),
        exp_avgs=(cast(torch.Tensor, optimizer_state["exp_avg"]).clone(),),
        exp_avg_sqs=(cast(torch.Tensor, optimizer_state["exp_avg_sq"]).clone(),),
        steps=(float(cast(torch.Tensor, optimizer_state["step"]).item()),),
        config=AdamWRecord(
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
    restored._dynamic_optimizers["student"] = restored._restore_canonical_optimizer(
        "student", state
    )
    for param in (original_param, restored_param):
        param.grad = torch.tensor([-0.3, 0.1], dtype=torch.bfloat16)
    original.optim_step(params=adam)
    restored.optim_step(params=adam)

    torch.testing.assert_close(restored_param, original_param, atol=0, rtol=0)
    _assert_nested_tensors_equal(
        restored._dynamic_optimizers["student"].optimizer.state_dict(),
        original._dynamic_optimizers["student"].optimizer.state_dict(),
    )
    assert original._checkpoint_revisions["student"] == 2


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

    dynamic = trainer._dynamic_optimizers["student"]
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
        config=AdamWRecord(
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


def test_dp_rank_forward_preserves_nested_shape_for_inactive_requests() -> None:
    trainer = TrainerRank(_runtime())
    request_a = ForwardInput(input_tokens=torch.tensor([1]))
    request_b = ForwardInput(input_tokens=torch.tensor([2]))

    outputs = trainer.dp_rank_forward([[request_a], [request_b]])

    assert len(outputs) == 2
    assert len(outputs[0]) == 1
    assert outputs[0][0].target_logprobs is None
    assert outputs[1][0].target_logprobs is None
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
