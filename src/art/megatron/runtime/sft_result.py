from typing import Any

import torch

from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY


def materialize_sft_command_result(result: dict[str, Any]) -> dict[str, Any]:
    """Resolve an already-staged SFT result outside the serialized GPU turn."""

    workload = result["workload"]
    metrics = {
        "loss/train": float(result["reduced_loss"].item()),
        "data/gradient_step_nonpadding_logical_tokens": float(
            workload.logical_nonpadding_tokens
        ),
        "data/gradient_step_loss_bearing_tokens": float(workload.loss_bearing_tokens),
        "data/gradient_step_executed_token_equivalents": float(
            workload.executed_token_equivalents
        ),
        "data/gradient_step_nominal_schedule_capacity_tokens": float(
            workload.nominal_schedule_capacity_tokens
        ),
        "data/gradient_step_dummy_executed_token_equivalents": float(
            workload.dummy_executed_token_equivalents
        ),
        "data/gradient_step_dummy_schedule_capacity_tokens": float(
            workload.dummy_schedule_capacity_tokens
        ),
        "pipeline/gradient_step_real_microbatches": float(workload.real_microbatches),
        "pipeline/gradient_step_dummy_microbatches": float(
            workload.dummy_microbatches
        ),
        (
            "time/forward_backward_s" if result["backward"] else "time/forward_s"
        ): result["elapsed_s"],
        **result["telemetry"].metrics(),
    }
    if result["backward"]:
        metrics[TRAIN_GRADIENT_STEPS_KEY] = 1.0
    token_logprobs = ()
    if result["logprob_values"] is not None:
        present = result["logprob_present"]
        if not torch.all(present == 1):
            raise RuntimeError("SFT forward did not materialize every trajectory")
        values = result["logprob_values"]
        token_logprobs = tuple(
            tuple(float(value) for value in values[index, :length].tolist())
            for index, length in enumerate(result["logprob_lengths"])
        )
    return {
        "operation_id": result["operation_id"],
        "metrics": metrics,
        "token_count": int(result["token_count"].item()),
        "token_logprobs": token_logprobs,
    }
