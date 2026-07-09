from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")

from megatron.core import parallel_state as ps  # noqa: E402
from torch.distributed import destroy_process_group, init_process_group  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from art.loss import LossInputs, loss_fn, shift_tensor  # noqa: E402
from art.megatron.context_parallel.runtime import prepare_cp_micro  # noqa: E402
from art.megatron.context_parallel.types import (  # noqa: E402
    ArtContextParallelState,
    ContextParallelConfig,
    DispatchedPackedTensors,
    ParallelTopology,
)
from art.preprocessing.pack import PackedTensors  # noqa: E402

from .cases import default_phase0_cases  # noqa: E402
from .distributed_init import file_init_method  # noqa: E402
from .packed_layout import build_phase0_packed_tensors  # noqa: E402


def test_gdn_cp_training_batch_carries_prebuilt_rank_plan(tmp_path: Path) -> None:
    cp_size = 2
    if not torch.cuda.is_available() or torch.cuda.device_count() < cp_size:
        pytest.skip(f"requires {cp_size} CUDA devices")
    init_method = file_init_method(tmp_path, "gdn_cp_train_prepare")
    mp.spawn(
        _worker,
        args=(cp_size, init_method, str(tmp_path)),
        nprocs=cp_size,
        join=True,
    )
    for rank in range(cp_size):
        assert (tmp_path / f"rank_{rank}.ok").read_text() == "ok\n"


def _worker(rank: int, cp_size: int, init_method: str, output_dir: str) -> None:
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=cp_size,
    )
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
            expert_model_parallel_size=1,
        )
        micro = cast(
            PackedTensors,
            {
                key: value.cuda() if isinstance(value, torch.Tensor) else value
                for key, value in build_phase0_packed_tensors(
                    default_phase0_cases()[1]
                ).items()
            },
        )
        cast(Any, micro)["original_logprobs"] = micro["logprobs"] + 0.125
        ref_logprobs = torch.full_like(micro["logprobs"], -0.25)
        prepared = prepare_cp_micro(
            micro=micro,
            topology=ParallelTopology(cp=cp_size),
            config=ContextParallelConfig(),
            cp_group=ps.get_context_parallel_group(check_initialized=False),
            cp_rank=ps.get_context_parallel_rank(),
            build_gdn_execution_spec=True,
            ref_logprobs=ref_logprobs,
        )
        state = prepared.attention_state
        assert isinstance(state, ArtContextParallelState)
        plan = state.gdn_execution_plan
        assert plan is not None
        assert plan.cp_rank == rank
        assert plan.cp_size == cp_size
        assert state.gdn_execution_spec is not None
        assert prepared.tensors.tokens.shape == (1, int(plan.attention_token_count))
        assert prepared.tensors.labels.shape == prepared.tensors.tokens.shape
        assert prepared.tensors.input_pos.shape == prepared.tensors.tokens.shape
        assert prepared.tensors.group_ids.shape == prepared.tensors.tokens.shape
        assert prepared.tensors.original_logprobs is not None
        assert prepared.tensors.original_logprobs.shape == prepared.tensors.tokens.shape
        assert prepared.tensors.ref_logprobs is not None
        assert prepared.tensors.ref_logprobs.shape == prepared.tensors.tokens.shape
        assert prepared.tensors.loss_all_reduce_group is not None
        assert prepared.tensors.valid_lengths == (int(plan.attention_token_count),)
        Path(output_dir, f"rank_{rank}.ok").write_text("ok\n")
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def test_main_loss_matches_shifted_dispatched_loss_inputs() -> None:
    packed = cast(
        Any,
        {
            "tokens": torch.tensor([[10, 11, 12, 13, 14, 0]]),
            "group_ids": torch.tensor([[1, 1, 2, 2, 2, -1]]),
            "parent_ids": torch.tensor([[1, 1, 1, 1, 1, -1]]),
            "input_pos": torch.arange(6).reshape(1, 6),
            "assistant_mask": torch.tensor([[False, True, True, True, True, False]]),
            "logprobs": torch.tensor(
                [[float("nan"), -0.72, -0.65, -0.81, -0.52, float("nan")]]
            ),
            "original_logprobs": torch.tensor(
                [[float("nan"), -0.70, -0.60, -0.80, -0.55, float("nan")]]
            ),
            "advantages": torch.tensor([[0.0, 0.3, -0.2, 0.4, -0.5, 0.0]]),
            "weights": torch.tensor([[0.0, 1.0, 1.2, 0.8, 1.1, 0.0]]),
            "pixel_values": [None],
            "image_grid_thw": [None],
        },
    )
    ref_logprobs = torch.tensor([[-0.9, -0.7, -0.6, -0.8, -0.55, -0.5]])
    entropies = torch.tensor([[0.0, 0.2, 0.4, 0.6, 0.8, 0.0]])
    dispatched = DispatchedPackedTensors(
        tokens=packed["tokens"],
        labels=shift_tensor(packed["tokens"], -100),
        input_pos=packed["input_pos"],
        assistant_mask=shift_tensor(packed["assistant_mask"], False),
        group_ids=shift_tensor(packed["group_ids"], 0),
        old_logprobs=shift_tensor(packed["logprobs"], float("nan")),
        advantages=shift_tensor(packed["advantages"], 0.0),
        weights=shift_tensor(packed["weights"], 0.0),
        valid_lengths=(6,),
        original_logprobs=shift_tensor(packed["original_logprobs"], 0.0),
        ref_logprobs=ref_logprobs,
    )
    config = cast(
        Any,
        {
            "importance_sampling_level": "sequence",
            "truncated_importance_sampling": 1.4,
            "kl_penalty_coef": 0.15,
        },
    )
    dense_new_logprobs = torch.tensor(
        [[-0.85, -0.69, -0.66, -0.75, -0.51, -0.4]], requires_grad=True
    )
    dispatched_new_logprobs = dense_new_logprobs.detach().clone().requires_grad_()

    dense_loss = loss_fn(
        LossInputs(inputs=packed),
        new_logprobs=dense_new_logprobs,
        ref_logprobs=ref_logprobs,
        entropies=entropies,
        experimental_config=config,
        reduction="sum",
    )
    dispatched_loss = loss_fn(
        dispatched,
        new_logprobs=dispatched_new_logprobs,
        ref_logprobs=dispatched.ref_logprobs,
        entropies=shift_tensor(entropies, 0.0),
        experimental_config=config,
        reduction="sum",
    )
    dense_loss.policy_loss.backward()
    dispatched_loss.policy_loss.backward()

    torch.testing.assert_close(dispatched_loss.policy_loss, dense_loss.policy_loss)
    torch.testing.assert_close(
        dispatched_loss.policy_loss_sum,
        dense_loss.policy_loss_sum,
    )
    assert dispatched_loss.entropy is not None and dense_loss.entropy is not None
    torch.testing.assert_close(dispatched_loss.entropy, dense_loss.entropy)
    assert (
        dispatched_loss.kl_policy_ref is not None
        and dense_loss.kl_policy_ref is not None
    )
    torch.testing.assert_close(
        dispatched_loss.kl_policy_ref,
        dense_loss.kl_policy_ref,
    )
    torch.testing.assert_close(
        dispatched_new_logprobs.grad,
        dense_new_logprobs.grad,
    )
