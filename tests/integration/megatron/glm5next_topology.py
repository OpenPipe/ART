"""CPU validation of sealed TrainerRank plans; no CUDA model execution."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch

from art.trainer_rank import ForwardInput, TrainerRank


class _ExecutionBoundary(Exception):
    """The next operation needs GPU buffers or CP collectives."""


def validate_plan(spec: dict, tokens: list[int], *, backward: bool) -> dict:
    """Real init, request validation, packing, flat-plan creation and topology guard.

    The scalar CPU model supplies metadata only. Parallel-state queries describe
    the sealed world rather than this singleton process. No TrainerRank planner
    or topology validator is replaced. Stop before HybridEP buffer allocation or
    CP device preparation; those remain GPU-window outputs.
    """
    from megatron.core import parallel_state as ps

    from art.megatron.model_support.handlers.glm5next import GLM5NEXT_HANDLER

    cp, ep, world = (int(spec[key]) for key in ("cp", "ep", "world_size"))
    if min(cp, ep, world) < 1 or world % cp or world % ep:
        raise ValueError("CP and EP must each divide the sealed world size")
    provider = SimpleNamespace(
        hidden_size=4096,
        num_layers=45,
        padded_vocab_size=154880,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    model: Any = torch.nn.Linear(1, 1, bias=False)
    model.config = provider
    model.decoder = SimpleNamespace(layers=[])
    model._preprocess = lambda **_: None

    def buffer_boundary(*_args: Any, **kwargs: Any) -> None:
        # Execute the real second guard too, before the CUDA buffer boundary.
        if ep > 1:
            rank._validate_hybridep_topology(kwargs["topology"])
        raise _ExecutionBoundary

    queries = {
        "get_tensor_model_parallel_world_size": 1,
        "get_pipeline_model_parallel_world_size": 1,
        "get_context_parallel_world_size": cp,
        "get_data_parallel_world_size": world // cp,
        "get_expert_model_parallel_world_size": ep,
        "get_expert_tensor_parallel_world_size": 1,
    }
    with ExitStack() as stack:
        for name, value in queries.items():
            stack.enter_context(patch.object(ps, name, return_value=value))
        rank = TrainerRank(
            cast(
                Any,
                SimpleNamespace(
                    model=[model],
                    provider=provider,
                    optimizer=None,
                    model_support_handler=GLM5NEXT_HANDLER,
                    rank=0,
                    world_size=world,
                ),
            )
        )
        request = ForwardInput(
            input_tokens=torch.tensor(tokens[:-1]),
            target_tokens=torch.tensor(tokens[1:]),
            top_k=1,
            no_grad=not backward,
            logits=spec["task"] == "parity" and len(tokens) <= 128,
        )
        # The worker sends one request per DP rank; retain that exact multiplicity
        # so the old CP1/EP8 plan reaches the actual DP8 refusal.
        plan = rank._plan_flat_forward([request] * (world // cp))
        with patch.object(rank, "_configure_hybridep", buffer_boundary):
            try:
                rank._execute_flat_plan(plan)
            except _ExecutionBoundary:
                pass
            else:
                raise AssertionError("CPU audit did not reach the execution boundary")
        assert not torch.cuda.is_initialized()
        return {
            **spec,
            "dp": world // cp,
            "backward": backward,
            "logical_tokens": plan.logical_tokens,
            "packed_tokens": plan.packed_tokens,
            "request_count": plan.request_count,
            "groups": len(plan.groups),
            "topology": plan.signature.topology,
            "guard": "TrainerRank._execute_flat_plan -> _validate_hybridep_topology",
            "boundary": "before CUDA HybridEP buffers / CP collectives",
            "cuda_initialized": False,
        }


def validate_stage_plans(inputs: Path, output: Path) -> dict:
    from tests.integration.megatron.glm5next_qualification import documents, write

    docs = documents(inputs)
    records = []
    for spec in json.loads((inputs / "stage-plans.json").read_text()):
        task = spec["task"]
        if task.startswith("context-"):
            length = int(task.split("-")[1])
            ids = (docs[0][1] * ((length + 1) // len(docs[0][1]) + 1))[: length + 1]
            batches = [(task, ids, True)]
        elif task == "learn":
            batches = [(f"repeat-{i}", docs[0][1], False) for i in (1, 2)]
            batches += [(f"learn-{i}", docs[0][1], True) for i in (1, 2, 3)]
        else:
            batches = [(name, ids, False) for name, ids in docs]
            if task == "forward":
                batches = [(f"repeat-{i}", docs[0][1], False) for i in (1, 2)] + batches
        for name, ids, backward in batches:
            records.append(
                {"batch": name, **validate_plan(spec, ids, backward=backward)}
            )
    result = {
        "status": "pass",
        "plans": records,
        "stage_count": len({row["name"] for row in records}),
        "scope": "real TrainerRank host plan/topology validation; metadata model and injected parallel-state sizes; GPU execution unmeasured",
    }
    write(output, result)
    return result
