from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from art.megatron import train
from art.megatron.composite_loss import (
    PreparedDistillationSidecars,
    PreparedPolicySidecars,
    composite_prepared_loss_from_logits,
)
from art.megatron.context_parallel.types import ParallelTopology
from art.megatron.distillation import (
    CispoObjectiveConfig,
    DistillationObjectiveConfig,
    PackedDistillationTensors,
)
from art.megatron.runtime.jobs import MegatronDistillationJob
from art.types import TrainConfig


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.2, -0.3, 0.5, 0.1, 50.0, -50.0],
                    [-0.1, 0.4, 0.3, -0.2, -50.0, 50.0],
                    [0.6, 0.1, -0.4, 0.2, 40.0, 30.0],
                    [0.0, -0.2, 0.1, 0.4, 20.0, 10.0],
                ]
            )
        )
        self.labels_seen: list[Any] = []

    def zero_grad_buffer(self) -> None:
        self.zero_grad(set_to_none=True)

    def forward(self, **kwargs: Any) -> torch.Tensor:
        self.labels_seen.append(kwargs.get("labels"))
        # Exercise Megatron's sequence-first raw-logit layout.
        return self.logits.unsqueeze(1)


class _FakeOptimizer:
    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.parameter = parameter
        self.param_groups: list[dict[str, Any]] = [{"params": [parameter], "lr": 0.0}]
        self.last_gradient: torch.Tensor | None = None

    def step(self) -> tuple[bool, float, int]:
        assert self.parameter.grad is not None
        self.last_gradient = self.parameter.grad.detach().clone()
        grad_norm = float(torch.linalg.vector_norm(self.parameter.grad).item())
        with torch.no_grad():
            self.parameter.add_(
                self.parameter.grad, alpha=-float(self.param_groups[0]["lr"])
            )
        return True, grad_norm, int(torch.count_nonzero(self.parameter.grad == 0))

    def zero_grad(self) -> None:
        self.parameter.grad = None


class _FakeHandler:
    build_gdn_execution_spec = False

    def get_forward_kwargs(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {}

    def zero_internal_padding_grads(self, _model: Any) -> None:
        return

    def zero_internal_padding_params(self, _model: Any) -> None:
        return


def _packed() -> PackedDistillationTensors:
    mask = torch.tensor([[False, True, False, True]])
    topk_ids = torch.full((1, 4, 2), -1, dtype=torch.long)
    topk_ids[0, 1] = torch.tensor([1, 2])
    topk_ids[0, 3] = torch.tensor([0, 3])
    teacher_logprobs = torch.zeros((1, 4, 2))
    teacher_logprobs[0, 1] = torch.tensor([0.6, 0.2]).log()
    teacher_logprobs[0, 3] = torch.tensor([0.5, 0.25]).log()
    tail_logprobs = torch.zeros((1, 4))
    tail_logprobs[0, 1] = math.log(0.2)
    tail_logprobs[0, 3] = math.log(0.25)
    return {
        "tokens": torch.tensor([[0, 1, 2, 3]]),
        "token_mask": torch.ones((1, 4), dtype=torch.bool),
        "input_pos": torch.arange(4).unsqueeze(0),
        "target_mask": mask,
        "distillation_weights": mask.to(torch.float32),
        "topk_token_ids": topk_ids,
        "teacher_logprobs": teacher_logprobs,
        "tail_logprobs": tail_logprobs,
        "temperatures": torch.ones((1, 4)),
    }


def _additive_packed() -> PackedDistillationTensors:
    first = _packed()
    mask = torch.tensor([[False, True, True, False], [False, True, False, False]])
    target_mask = torch.tensor(
        [[False, True, False, False], [False, False, True, False]]
    )
    topk_ids = torch.full((2, 4, 2), -1, dtype=torch.long)
    teacher_logprobs = torch.zeros((2, 4, 2))
    tail_logprobs = torch.zeros((2, 4))
    for row, position in ((0, 1), (1, 2)):
        topk_ids[row, position] = torch.tensor([1, 2])
        teacher_logprobs[row, position] = torch.tensor([0.6, 0.2]).log()
        tail_logprobs[row, position] = math.log(0.2)
    return {
        **first,
        "tokens": torch.tensor([[0, 1, 2, 3], [0, 2, 1, 3]]),
        "token_mask": torch.ones((2, 4), dtype=torch.bool),
        "input_pos": torch.arange(4).repeat(2, 1),
        "source_group_ids": torch.tensor([0, 0]),
        "policy_mask": mask,
        "old_logprobs": torch.where(
            mask,
            torch.tensor(-1.0),
            torch.tensor(float("nan")),
        ),
        "policy_advantages": torch.tensor(
            [[0.0, 1.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0]]
        ),
        "policy_weights": mask.to(torch.float32),
        "policy_group_ids": torch.zeros((2, 4), dtype=torch.long),
        "target_mask": target_mask,
        "distillation_weights": target_mask.to(torch.float32),
        "topk_token_ids": topk_ids,
        "teacher_logprobs": teacher_logprobs,
        "tail_logprobs": tail_logprobs,
        "temperatures": torch.ones((2, 4)),
    }


def _patch_single_rank_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> list[torch.Tensor]:
    finalized_denominators: list[torch.Tensor] = []
    monkeypatch.setattr(
        train,
        "_validate_distillation_worker_topology",
        lambda _model: None,
    )
    monkeypatch.setattr(train, "as_megatron_api_chunks", lambda chunks: chunks)
    monkeypatch.setattr(
        train,
        "finalize_model_grads_extended",
        lambda _chunks, num_tokens: finalized_denominators.append(
            num_tokens.detach().clone()
        ),
    )
    monkeypatch.setattr(
        train,
        "_causal_attention_state",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(train, "_art_flex_sliding_windows", lambda _provider: ())
    return finalized_denominators


def test_shift_preserves_rank_three_topk_width() -> None:
    values = torch.arange(8).reshape(1, 4, 2)

    shifted = train._shift_distillation_token_axis(values, pad=-1)

    assert shifted.shape == (1, 4, 2)
    assert shifted.tolist() == [[[2, 3], [4, 5], [6, 7], [-1, -1]]]


def test_fake_step_uses_raw_logits_exact_shift_and_independent_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalized_denominators = _patch_single_rank_worker(monkeypatch)
    model = _FakeModel()
    optimizer = _FakeOptimizer(model.logits)
    before = model.logits.detach().clone()

    result = train.run_megatron_distillation_step(
        model_chunks=cast(Any, [model]),
        provider=object(),
        model_support_handler=_FakeHandler(),
        optimizer=optimizer,
        learning_rate=0.1,
        packed_tensors=_packed(),
        sample_indices=[0],
        logical_vocab_size=4,
        temperature=1.0,
        coefficient=0.5,
        compensate_temperature_squared=False,
    )

    assert model.labels_seen == [None]
    assert result.target_token_count == 2
    assert len(finalized_denominators) == 1
    assert finalized_denominators[0].tolist() == [2.0]
    assert result.update_successful
    assert math.isfinite(result.reduced_loss.item())
    assert result.reduced_loss.item() > 0
    assert math.isfinite(result.grad_norm)
    assert result.grad_norm > 0
    changed_rows = torch.any(model.logits.detach() != before, dim=-1)
    assert changed_rows.tolist() == [True, False, True, False]
    assert torch.equal(model.logits.detach()[:, 4:], before[:, 4:])


def test_additive_step_uses_one_forward_and_independent_global_denominators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalized_denominators = _patch_single_rank_worker(monkeypatch)
    model = _FakeModel()
    optimizer = _FakeOptimizer(model.logits)

    def _divide_gradients(chunks: Any, num_tokens: torch.Tensor) -> None:
        finalized_denominators.append(num_tokens.detach().clone())
        for chunk in chunks:
            for parameter in chunk.parameters():
                assert parameter.grad is not None
                parameter.grad.div_(num_tokens.item())

    monkeypatch.setattr(train, "finalize_model_grads_extended", _divide_gradients)
    packed = _additive_packed()
    result = train.run_megatron_distillation_step(
        model_chunks=cast(Any, [model]),
        provider=object(),
        model_support_handler=_FakeHandler(),
        optimizer=optimizer,
        learning_rate=0.0,
        packed_tensors=packed,
        sample_indices=[0, 1],
        logical_vocab_size=4,
        temperature=1.0,
        coefficient=0.5,
        compensate_temperature_squared=False,
        policy_config=cast(
            Any,
            {
                "epsilon": 1.0,
                "epsilon_high": 4.0,
                "importance_sampling_level": "token",
            },
        ),
        expected_policy_count=3,
        expected_target_count=2,
    )
    assert optimizer.last_gradient is not None
    actual_gradient = optimizer.last_gradient

    reference_logits = model.logits.detach().clone().requires_grad_(True)
    policy_sum = reference_logits.new_zeros(())
    kd_sum = reference_logits.new_zeros(())
    for sample_index in (0, 1):
        micro = train._select_distillation_micro(
            packed,
            sample_index=sample_index,
            device=torch.device("cpu"),
        )
        logits = reference_logits.unsqueeze(0)
        composite = composite_prepared_loss_from_logits(
            logits,
            logical_vocab_size=4,
            policy=PreparedPolicySidecars(
                sampled_token_ids=micro["sampled_token_ids"],
                old_logprobs=micro["old_logprobs"],
                advantages=micro["policy_advantages"],
                weights=micro["policy_weights"],
                mask=micro["policy_mask"],
                group_ids=micro["policy_group_ids"],
            ),
            distillation=(
                PreparedDistillationSidecars(
                    teacher_topk_ids=micro["topk_token_ids"],
                    teacher_topk_logprobs=micro["teacher_logprobs"],
                    teacher_tail_logprob=micro["tail_logprobs"],
                    mask=micro["target_mask"],
                    weights=micro["distillation_weights"],
                )
                if bool(micro["target_mask"].any())
                else None
            ),
            policy_config=cast(
                Any,
                {
                    "epsilon": 1.0,
                    "epsilon_high": 4.0,
                    "importance_sampling_level": "token",
                },
            ),
            distillation_coefficient=0.5,
        )
        assert composite.policy is not None
        policy_sum = policy_sum + composite.policy.loss_sum
        if composite.distillation is not None:
            kd_sum = kd_sum + composite.distillation.loss_sum
    reference_loss = policy_sum / 3 + 0.5 * kd_sum / 2
    reference_loss.backward()

    assert reference_logits.grad is not None
    torch.testing.assert_close(actual_gradient, reference_logits.grad)
    assert model.labels_seen == [None, None]
    assert finalized_denominators[-1].tolist() == [3.0]
    assert result.policy_token_count == 3
    assert result.target_token_count == 2
    torch.testing.assert_close(result.reduced_loss, reference_loss.detach())


def test_additive_zero_denominator_fails_before_forward_or_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_rank_worker(monkeypatch)
    model = _FakeModel()
    optimizer = _FakeOptimizer(model.logits)
    packed = _additive_packed()
    packed["policy_mask"].zero_()

    with pytest.raises(RuntimeError, match="zero policy denominator"):
        train.run_megatron_distillation_step(
            model_chunks=cast(Any, [model]),
            provider=object(),
            model_support_handler=_FakeHandler(),
            optimizer=optimizer,
            learning_rate=0.1,
            packed_tensors=packed,
            sample_indices=[0, 1],
            logical_vocab_size=4,
            temperature=1.0,
            coefficient=0.5,
            compensate_temperature_squared=False,
            policy_config=cast(Any, {"importance_sampling_level": "token"}),
        )

    assert model.labels_seen == []
    assert model.logits.grad is None


def test_worker_topology_rejects_nonunit_data_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train,
        "_infer_parallel_topology",
        lambda _model: ParallelTopology(tp=1, dp=2, cp=1, pp=1),
    )
    monkeypatch.setattr(train.ps, "get_expert_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(train.ps, "get_expert_tensor_parallel_world_size", lambda: 1)

    with pytest.raises(ValueError, match="TP=DP=CP=PP=EP=ETP=1"):
        train._validate_distillation_worker_topology(cast(Any, []))


def test_job_dispatch_keeps_legacy_jobs_on_rl_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        train,
        "run_megatron_rl_job",
        lambda _runtime, _job: calls.append("rl"),
    )
    monkeypatch.setattr(
        train,
        "run_megatron_distillation_job",
        lambda _runtime, _job: calls.append("distill"),
    )
    train._run_megatron_job(cast(Any, object()), cast(Any, object()))
    assert calls == ["rl"]

    distillation_job = MegatronDistillationJob(
        step=1,
        source_policy_step=0,
        expected_source_revision=0,
        training_session_id="session",
        lora_path="/tmp/lora",
        optimizer_state_path="/tmp/optimizer",
        distillation_tensors={
            "schema_version": 1,
            "dir": "/tmp/tensors",
            "num_sequences": 1,
            "sequence_length": 4,
            "top_k_width": 2,
            "target_count": 2,
            "logical_vocab_size": 4,
            "tensors_sha256": "checksum",
        },
        config=TrainConfig(),
        objective=DistillationObjectiveConfig(coefficient=1.0),
        idempotency_key="update",
        preparation_id="preparation",
        payload_sha256="payload",
    )
    train._run_megatron_job(cast(Any, object()), distillation_job)
    assert calls == ["rl", "distill"]


def test_distillation_metrics_include_stable_public_keys(tmp_path: Path) -> None:
    result = train.DistillationTrainStepResult(
        reduced_loss=torch.tensor(0.75),
        raw_loss_sum=3.0,
        selected_loss_sum=2.0,
        tail_loss_sum=1.0,
        target_token_count=2,
        teacher_tail_mass_mean=0.2,
        student_tail_mass_mean=0.3,
        numerical_clamp_count=1,
        update_successful=True,
        grad_norm=0.4,
        num_zeros_in_grad=0,
    )
    log_path = tmp_path / "metrics.jsonl"

    train._log_distillation_step_result(
        0,
        str(log_path),
        result,
        coefficient=0.5,
        temperature=1.0,
        num_gradient_steps=1,
    )

    metrics = json.loads(log_path.read_text())
    assert metrics["loss/distillation_sum"] == 3.0
    assert metrics["loss/distillation_selected"] == 1.0
    assert metrics["loss/distillation_tail"] == 0.5
    assert metrics["data/distillation_tokens"] == 2.0
    assert metrics["distillation/coefficient"] == 0.5
    assert metrics["distillation/temperature"] == 1.0
    assert metrics["distillation/teacher_tail_mass_mean"] == 0.2
    assert metrics["distillation/student_tail_mass_mean"] == 0.3
    assert metrics["distillation/numerical_clamp_count"] == 1.0


def test_unsuccessful_optimizer_step_is_rejected_before_save() -> None:
    result = train.DistillationTrainStepResult(
        reduced_loss=torch.tensor(0.75),
        raw_loss_sum=3.0,
        selected_loss_sum=2.0,
        tail_loss_sum=1.0,
        target_token_count=2,
        teacher_tail_mass_mean=0.2,
        student_tail_mass_mean=0.3,
        numerical_clamp_count=0,
        update_successful=False,
        grad_norm=0.4,
        num_zeros_in_grad=0,
    )

    with pytest.raises(RuntimeError, match="optimizer step was not successful"):
        train._validate_distillation_step_result_finite(
            cast(Any, object()),
            result,
        )
