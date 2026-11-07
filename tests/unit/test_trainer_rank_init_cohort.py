"""CPU/Gloo construction guards for execution scope and layout scoring."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_trainer_rank_active_memory import _Model, _rank
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import TrainerRank, TrainerRankRuntimeSupportError, _impl
from art.trainer_rank._planner_cost import (
    GDN_MOE_H2048_GEOMETRY,
    QWEN35_4B_GEOMETRY,
    ParallelShape,
    select_scoring,
)


class _DeviceFacadeParameter(torch.nn.Parameter):
    # Storage stays on CPU; only the constructor's accelerator queries are faked.
    @property
    def device(self):
        return torch.device("cuda", 0)


def _runtime():
    model = _Model()
    model.weight = _DeviceFacadeParameter(torch.zeros((), dtype=torch.bfloat16))
    assert not model.weight.is_cuda
    model.config = SimpleNamespace(
        hidden_size=2560,
        ffn_hidden_size=9216,
        num_attention_heads=16,
        num_query_groups=4,
        kv_channels=256,
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_layers=32,
        padded_vocab_size=32,
    )
    return cast(
        Any,
        SimpleNamespace(
            model=[model],
            optimizer=None,
            provider=model.config,
            model_support_handler=SimpleNamespace(build_gdn_execution_spec=True),
        ),
    )


def _worker(index, directory, mode, world_size):
    from megatron.core import parallel_state as ps

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=index,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    row: dict[str, Any] = dict(constructed=False, error=None, init_exchanges=0)
    try:
        tp = 2 if mode == "contained-etp" else 1
        cp = (
            8
            if mode in ("homogeneous-cp8", "global-aux-cp8")
            else (1 if mode in ("etp", "ep", "contained-etp", "global-aux") else 2)
        )
        ep = (
            2
            if mode in ("ep", "contained-ep", "expert-type-error", "expert-key-error")
            else 1
        )
        etp = 2 if mode in ("etp", "contained-etp") else 1
        ps.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            context_parallel_size=cp,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=etp,
            pipeline_model_parallel_size=1,
            pipeline_model_parallel_comm_backend=None,
        )
        cohort = ps.get_tensor_and_context_parallel_group()
        row["shape"] = [tp, cp, ep, etp]
        row["memory_ranks"] = dist.get_process_group_ranks(cohort)
        row["expert_ranks"] = dist.get_process_group_ranks(
            ps.get_expert_tensor_and_model_parallel_group()
        )
        memory_gib = 143
        capability = (9, 0)
        if mode == "heterogeneous" and index == 1:
            memory_gib = 80
        elif mode == "facts-only" and index == 1:
            memory_gib = 144  # Same scorer, different accelerator facts.
        elif mode == "capability" and index == 1:
            capability = (8, 0)
        elif mode == "isolated-cohorts" and index >= 2:
            memory_gib = 80
        row["capability"] = capability
        row["device_memory_bytes"] = memory_gib * 1024**3
        original_gather = dist.all_gather_object
        original_select = _impl.select_scoring

        def gather(output, value, *, group=None):
            assert group is cohort
            assert dist.get_backend(group) == "gloo"
            row["init_exchanges"] += 1
            return original_gather(output, value, group=group)

        def select(**kwargs):
            selection = original_select(**kwargs)
            if mode == "selection-only" and index == 1:
                selection = replace(selection, table_id="different-table")
            row["table"] = selection.table_id
            return selection

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
            patch.setattr(
                torch.cuda,
                "get_device_properties",
                lambda _: SimpleNamespace(total_memory=memory_gib * 1024**3),
            )
            patch.setattr(dist, "all_gather_object", gather)
            patch.setattr(_impl, "select_scoring", select)
            # Parameters only expose CUDA metadata; object transport is real Gloo.
            patch.setattr(torch.cuda, "device", lambda device: nullcontext())
            if mode == "partial-scope-error":
                patch.setattr(
                    TrainerRank,
                    "_expert_admission_scope_error",
                    lambda self, cohort: (
                        "expert group outside admission cohort" if index == 0 else None
                    ),
                )
            if mode in ("expert-type-error", "expert-key-error") and index == 0:

                def unavailable(**kwargs):
                    raise (TypeError if mode == "expert-type-error" else KeyError)(
                        "group metadata"
                    )

                patch.setattr(
                    ps, "get_expert_tensor_and_model_parallel_group", unavailable
                )
            runtime = _runtime()
            if mode in (
                "global-aux",
                "global-aux-cp8",
                "aux-overflow",
                "provider-overridden",
            ):
                from megatron.core.process_groups_config import ProcessGroupCollection
                from megatron.core.transformer.moe.router import TopKRouter
                from megatron.core.transformer.transformer_config import (
                    TransformerConfig,
                )

                config = TransformerConfig(
                    num_layers=40,
                    hidden_size=2048,
                    num_attention_heads=16,
                    num_moe_experts=256,
                    moe_router_topk=8,
                    moe_router_load_balancing_type="global_aux_loss",
                    moe_aux_loss_coeff=0.0 if mode == "provider-overridden" else 1e-3,
                    moe_router_fusion=False,
                )
                groups = ProcessGroupCollection()
                groups.tp = ps.get_tensor_model_parallel_group()
                groups.cp = ps.get_context_parallel_group()
                groups.tp_cp = cohort
                groups.tp_dp_cp = ps.get_tensor_and_data_parallel_group(
                    with_context_parallel=True
                )
                with patch.context() as router_patch:
                    router_patch.setattr(
                        torch.cuda, "current_device", lambda: torch.device("cpu")
                    )
                    runtime.model[0].router = TopKRouter(config, pg_collection=groups)
                    if mode == "provider-overridden":
                        runtime.model[0].second_router = TopKRouter(
                            config, pg_collection=groups
                        )
                if mode == "aux-overflow" and index == 0:
                    config.moe_aux_loss_coeff = 10**400
                row["router_ranks"] = dist.get_process_group_ranks(groups.tp_dp_cp)
                row["global_aux_coefficient"] = config.moe_aux_loss_coeff
                runtime.model_support_handler.is_moe = True
                if mode == "provider-overridden":
                    runtime.provider.moe_router_load_balancing_type = "global_aux_loss"
                    runtime.provider.moe_aux_loss_coeff = 0.3
                    row["router_count"] = 2
                    row["provider_coefficient"] = runtime.provider.moe_aux_loss_coeff
                if mode == "global-aux-cp8":
                    geometry = GDN_MOE_H2048_GEOMETRY
                    runtime.model[0].config.num_layers = 40
                    for name in (
                        "hidden_size",
                        "ffn_hidden_size",
                        "num_attention_heads",
                        "num_query_groups",
                        "kv_channels",
                    ):
                        setattr(runtime.model[0].config, name, getattr(geometry, name))
                    runtime.model[0].config.num_moe_experts = geometry.moe_experts
                    runtime.model[0].config.moe_router_topk = geometry.moe_topk
                    runtime.model[
                        0
                    ].config.moe_ffn_hidden_size = geometry.moe_ffn_hidden_size
                    runtime.model[
                        0
                    ].config.moe_shared_expert_intermediate_size = (
                        geometry.moe_shared_expert_ffn
                    )
            trainer = TrainerRank(runtime)
            if mode == "global-aux-cp8":
                assert trainer._geometry == GDN_MOE_H2048_GEOMETRY
            row["constructed"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (Path(directory) / f"rank-{index}.json").write_text(json.dumps(row))
        ps.destroy_model_parallel()
        dist.destroy_process_group()


def _run(tmp_path, mode, world_size=2):
    pytest.importorskip("megatron.core.parallel_state")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    _rank()  # Finish optional runtime imports before forking CPU workers.
    context = mp.start_processes(
        _worker,
        args=(str(tmp_path), mode, world_size),
        nprocs=world_size,
        join=False,
        start_method="fork",
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, "constructor workers timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
    return [
        json.loads((tmp_path / f"rank-{i}.json").read_text()) for i in range(world_size)
    ]


@pytest.mark.parametrize("mode", ["etp", "ep"])
def test_uncontained_expert_groups_are_rejected_at_construction(tmp_path, mode):
    rows = _run(tmp_path, mode)
    assert all(row["memory_ranks"] == [i] for i, row in enumerate(rows)), rows
    assert all(row["expert_ranks"] == [0, 1] for row in rows), rows
    assert all(not row["constructed"] for row in rows), rows
    assert all("TrainerRankRuntimeSupportError" in row["error"] for row in rows), rows
    assert all(
        "admission" in row["error"] and "expert" in row["error"] for row in rows
    ), rows


@pytest.mark.parametrize(
    "mode", ["heterogeneous", "facts-only", "capability", "selection-only"]
)
def test_mismatched_scoring_facts_raise_on_every_rank(tmp_path, mode):
    rows = _run(tmp_path, mode)
    assert all(not row["constructed"] for row in rows), rows
    assert all("TrainerRankRuntimeSupportError" in row["error"] for row in rows), rows
    assert all("layout-scoring" in row["error"] for row in rows), rows
    assert all(row["init_exchanges"] == 1 for row in rows), rows
    if mode == "heterogeneous":
        assert rows[0]["table"] != rows[1]["table"], rows
    elif mode == "facts-only":
        assert rows[0]["table"] == rows[1]["table"], rows


@pytest.mark.parametrize(
    "mode", ["contained-etp", "contained-ep", "isolated-cohorts", "homogeneous-cp8"]
)
def test_contained_groups_and_homogeneous_scoring_construct(tmp_path, mode):
    size = 8 if mode == "homogeneous-cp8" else (4 if mode == "isolated-cohorts" else 2)
    rows = _run(tmp_path, mode, size)
    assert all(row["constructed"] and row["error"] is None for row in rows), rows
    assert all(row["init_exchanges"] == 1 for row in rows), rows
    assert all(set(row["expert_ranks"]) <= set(row["memory_ranks"]) for row in rows), (
        rows
    )
    if mode == "homogeneous-cp8":
        assert all(row["memory_ranks"] == list(range(8)) for row in rows), rows
        assert all(row["shape"] == [1, 8, 1, 1] for row in rows), rows
        assert all(row["capability"] == [9, 0] for row in rows), rows
        assert all(row["device_memory_bytes"] == 143 * 1024**3 for row in rows), rows
        assert all(row["table"] == rows[0]["table"] for row in rows), rows


def test_local_scope_error_rejects_every_cohort_member(tmp_path):
    rows = _run(tmp_path, "partial-scope-error")
    assert all(not row["constructed"] for row in rows), rows
    assert all(
        "expert group outside admission cohort" in row["error"] for row in rows
    ), rows
    assert all(row["init_exchanges"] == 1 for row in rows), rows


@pytest.mark.parametrize("backend", ["nccl", "cuda:nccl", "cpu:gloo,cuda:nccl", "gloo"])
def test_nccl_identity_exchange_uses_model_device(monkeypatch, backend):
    trainer = TrainerRank.__new__(TrainerRank)
    trainer.device = torch.device("cuda", 1)
    trainer._geometry = QWEN35_4B_GEOMETRY
    trainer._parallel_shape = ParallelShape(cp=2)
    cohort = object()
    monkeypatch.setattr(trainer, "_forward_memory_group", lambda: cohort)
    monkeypatch.setattr(trainer, "_expert_admission_scope_error", lambda _: None)
    monkeypatch.setattr(trainer, "_global_aux_admission_scope_error", lambda _: None)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_backend", lambda group: backend)
    active = []
    calls = []

    @contextmanager
    def device_context(device):
        assert device == trainer.device
        active.append(device)
        try:
            yield
        finally:
            active.pop()

    def gather(output, value, *, group):
        assert active == [trainer.device]
        assert group is cohort
        calls.append(value)
        output[:] = [value, value]

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(dist, "all_gather_object", gather)
    facts: dict[str, Any] = dict(
        device_capability=(9, 0),
        device_memory_bytes=143 * 1024**3,
        param_dtype="torch.bfloat16",
    )
    selection = select_scoring(
        **facts, geometry=trainer._geometry, shape=trainer._parallel_shape
    )
    trainer._validate_planner_cohort(selection, **facts)
    assert len(calls) == 1 and not active


def test_uninitialized_expert_groups_fail_closed():
    runtime = _runtime()
    runtime.provider.expert_tensor_parallel_size = 2
    # Use real CPU metadata without the accelerator facade outside Gloo workers.
    runtime.model[0].weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
    with pytest.raises(TrainerRankRuntimeSupportError, match="expert.*admission"):
        TrainerRank(runtime)


def test_global_aux_group_outside_admission_rejects_every_rank(tmp_path):
    rows = _run(tmp_path, "global-aux")
    assert all(row["memory_ranks"] == [i] for i, row in enumerate(rows)), rows
    assert all(row["router_ranks"] == [0, 1] for row in rows), rows
    assert all(not row["constructed"] for row in rows), rows
    assert all(
        "TrainerRankRuntimeSupportError" in row["error"] and "global" in row["error"]
        for row in rows
    ), rows
    assert all(row["init_exchanges"] == 1 for row in rows), rows


def test_global_aux_production_cp8_constructs(tmp_path):
    rows = _run(tmp_path, "global-aux-cp8", 8)
    assert all(row["constructed"] and row["error"] is None for row in rows), rows
    assert all(
        row["router_ranks"] == row["memory_ranks"] == list(range(8)) for row in rows
    ), rows
    assert all(
        row["global_aux_coefficient"] == 1e-3 and row["init_exchanges"] == 1
        for row in rows
    ), rows


@pytest.mark.parametrize("mode", ["expert-type-error", "expert-key-error"])
def test_membership_lookup_errors_reject_every_peer(tmp_path, mode):
    rows = _run(tmp_path, mode)
    assert all(not row["constructed"] for row in rows), rows
    assert all("TrainerRankRuntimeSupportError" in row["error"] for row in rows), rows
    assert all(row["init_exchanges"] == 1 for row in rows), rows


def test_huge_aux_coefficient_rejects_every_peer_after_exchange(tmp_path):
    rows = _run(tmp_path, "aux-overflow")
    assert all(not row["constructed"] for row in rows), rows
    assert all("TrainerRankRuntimeSupportError" in row["error"] for row in rows), rows
    assert all(row["init_exchanges"] == 1 for row in rows), rows


def test_disabled_routers_override_enabled_provider(tmp_path):
    rows = _run(tmp_path, "provider-overridden", 4)
    assert all(row["router_ranks"] == list(range(4)) for row in rows), rows
    assert all(len(row["memory_ranks"]) == 2 for row in rows), rows
    assert all(
        row["router_count"] == 2 and row["global_aux_coefficient"] == 0 for row in rows
    ), rows
    assert all(row["provider_coefficient"] == 0.3 for row in rows), rows
    assert all(row["constructed"] and row["error"] is None for row in rows), rows
    assert all(row["init_exchanges"] == 1 for row in rows), rows
