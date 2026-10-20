"""CPU lifecycle coverage for ART's opt-in LoRA representations."""

from dataclasses import asdict
from datetime import timedelta
import hashlib
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")
from art.megatron import lora as lm
from art.megatron.model_support.handlers import qwen3_5 as qwen
from art.megatron.weights import adapter_export, lora_publish
from art.trainer_rank import _checkpoint as cp
from art.trainer_rank._impl import TrainerRank, _distributed_grad_norms

PREFIX = "base_model.model.model.layers.0.mlp.experts"


@pytest.fixture(autouse=True)
def local_parallel_state(monkeypatch):
    for name in (
        "get_expert_model_parallel_rank",
        "get_expert_data_parallel_rank",
        "get_data_parallel_rank",
    ):
        monkeypatch.setattr(lm.ps, name, lambda **_: 0)
    monkeypatch.setattr(lm.ps, "get_tensor_model_parallel_world_size", lambda: 1)


def expert_lora(projection="gate_up_proj", layout=None, *, rank=2):
    return lm.LoRA(
        f"{PREFIX}.{{expert}}.{projection}",
        3,
        4,
        rank,
        3.5,
        torch.float32,
        torch.device("cpu"),
        num_local_experts=3,
        lora_layout=layout,
    )


@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
def test_shared_outer_persistence_inspection_and_packed_publish(projection):
    layout = lm.LoraLayout(expert="shared_outer")
    lora = expert_lora(projection, layout)
    with torch.no_grad():
        for param in lora.parameters():
            param.copy_(torch.arange(param.numel()).reshape(param.shape))
    state = {
        key: tensor.clone() for key, tensor in lora.sharded_lora_state_dict().items()
    }
    restored = expert_lora(projection, layout)
    restored.load_lora(state)
    for param, actual in zip(lora.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(param, actual, rtol=0, atol=0)
    ref = lm.LoRASlotRef("checkpoint", "shared")
    assert restored.load_lora_slot(ref, state, alpha=3.5, requires_grad=True)
    assert {
        key: tuple(t.shape) for key, t in restored.sharded_lora_state_dict(ref).items()
    } == {key: tuple(t.shape) for key, t in state.items()}
    for expert in range(3):
        a, b = adapter_export._adapter_tensors(lora, expert)
        torch.testing.assert_close(
            a, state[f"{PREFIX}.{expert}.{projection}.lora_A.weight"]
        )
        torch.testing.assert_close(
            b, state[f"{PREFIX}.{expert}.{projection}.lora_B.weight"]
        )
    planner = lm.LoRAPublishPlanner([lora])
    assert {m.key: m.shape for m in planner.global_metadata({})} == {
        key: tuple(value.shape) for key, value in state.items()
    }
    # Capture physical singleton storage, then expand every optimizer component
    # using the same logical expert view as adapter/publish export.
    slot_params = restored.lora_slot_params(ref)
    masters = tuple(torch.nn.Parameter(p.detach().clone()) for p in slot_params)
    optimizer = torch.optim.Adam(masters)
    for master in masters:
        optimizer.state[master] = {
            "step": 7.0,
            "exp_avg": master.detach() + 10,
            "exp_avg_sq": master.detach() + 20,
        }
    trainer = SimpleNamespace(
        runtime=SimpleNamespace(model=[restored]),
        _slot_ref=lambda _: ref,
        _checkpoint_slots={
            "shared": SimpleNamespace(
                optimizer=SimpleNamespace(optimizer=optimizer, master_params=masters),
                params=slot_params,
                custom={},
                custom_payload=None,
            )
        },
    )
    captured, _, _ = cp._local_state(cast(Any, trainer), "shared", {})
    assert sorted(s.tensors["lora"].shape[0] for s in captured) == [1, 3]
    files = {}
    cp._expand_local_state(captured, files)
    payload = next(iter(files.values()))
    for key, value in state.items():
        for component, offset in (
            ("lora", 0),
            ("master", 0),
            ("exp_avg", 10),
            ("exp_avg_sq", 20),
        ):
            torch.testing.assert_close(payload[f"{component}/{key}"], value + offset)
    groups = qwen.QWEN3_5_MOE_HANDLER.expert_packed_lora_groups()
    tensors, metadata = lora_publish.collect_local_packed_expert_entries(
        cast(Any, [lora]),
        {},
        owner_rank=0,
        packed_expert_groups=groups,
    )
    assert all(tensor.shape[0] == 3 for tensor in tensors.values())
    packed = lora_publish.merge_packed_expert_adapter_entries(
        metadata, {(0, key): value for key, value in tensors.items()}
    )
    for meta in metadata:
        suffix = "lora_A" if "lora_A" in meta.key else "lora_B"
        values = [
            state[f"{PREFIX}.{expert}.{projection}.{suffix}.weight"]
            for expert in range(3)
        ]
        expected = (
            torch.cat(values)
            if suffix == "lora_A"
            else torch.stack(values).permute(1, 2, 0).flatten(1)
        )
        torch.testing.assert_close(packed[meta.key], expected)
    # Independent per-expert weights cannot silently be folded into shared storage.
    shared_suffix = "lora_A" if projection != "down_proj" else "lora_B"
    state[f"{PREFIX}.1.{projection}.{shared_suffix}.weight"].add_(1)
    with pytest.raises(ValueError, match="independent"):
        restored.load_lora(state)


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
@pytest.mark.parametrize("parameterization", ["tied_v0", "tinker_v1"])
def test_shared_outer_public_checkpoint_continuation(
    rank, projection, parameterization, tmp_path, monkeypatch
):
    from copy import deepcopy
    import json
    import shutil

    import safetensors.torch
    from test_trainer_rank_validation import _assert_nested_tensors_equal, _runtime

    from art.trainer_rank import AdamParams, MaterializedCheckpoint
    from art.trainer_rank._impl import _CheckpointSlot

    layout = lm.LoraLayout(
        expert="shared_outer", qwen_parameterization=parameterization
    )
    projections = (
        ("gate_proj", "up_proj")
        if parameterization == "tinker_v1" and projection == "gate_up_proj"
        else (projection,)
    )
    config = layout.record_config(
        {
            "base_model_name_or_path": "test/model",
            "r": rank,
            "lora_alpha": 3.5,
            "target_modules": list(projections),
        }
    )
    adam = AdamParams(learning_rate=3e-4, beta1=0.8, beta2=0.95, weight_decay=0.1)

    def make_trainer():
        model = torch.nn.ModuleList(
            [expert_lora(p, layout, rank=rank) for p in projections]
        )
        runtime = _runtime(model)
        runtime.lora_layout = layout
        trainer = TrainerRank(runtime)
        monkeypatch.setattr(
            trainer,
            "_reduce_dynamic_grads",
            lambda params, **_: tuple(p.grad.float() for p in params),
        )
        return trainer

    def step(trainer):
        ref = trainer._slot_ref("student")
        for lora in trainer.runtime.model[0]:
            a, b = lora.lora_slot_params(ref)
            sum(
                (lm.lora_factor_view(a, e) @ lm.lora_factor_view(b, e)).square().sum()
                for e in range(3)
            ).backward()
        trainer.optim_step(params=adam, checkpoints=["student"])

    def state(trainer):
        slot = trainer._checkpoint_slots["student"]
        dynamic = slot.optimizer
        assert dynamic is not None
        return deepcopy(
            (slot.params, dynamic.master_params, dynamic.optimizer.state_dict())
        )

    original = make_trainer()
    adapter = {}
    for lora in original.runtime.model[0]:
        with torch.no_grad():
            for p in lora.parameters():
                p.copy_(torch.arange(p.numel()).reshape(p.shape) / 16 + 0.25)
        adapter.update(lora.sharded_lora_state_dict())
    loaded = original._load_checkpoint_slot("student", adapter, alpha=3.5)
    params = original._validate_checkpoint_consistency("student", loaded, set(adapter))
    original._checkpoint_slots["student"] = _CheckpointSlot(params, cast(Any, config))
    step(original)
    before = state(original)
    output = tmp_path / "checkpoint"
    # Exercise both the asynchronous spill and final safetensors writers.
    original.save_checkpoint(str(output), "student")
    prepared = cp.prepare_checkpoint(str(output))
    assert prepared.manifest is not None
    assert len(prepared.keys) == len(projections) * 3 * 2
    assert not list(tmp_path.glob(".checkpoint.snapshot-*"))
    assert not (tmp_path / ".checkpoint.reserved").exists()

    step(original)
    continued = state(original)
    restored = make_trainer()
    for trainer in (original, restored):
        trainer.load_checkpoint(MaterializedCheckpoint("student", str(output)))
        _assert_nested_tensors_equal(state(trainer), before)
        assert [
            tuple(p.shape) for p in trainer._checkpoint_slots["student"].params
        ] == [tuple(p.shape) for p in params]
        step(trainer)
        _assert_nested_tensors_equal(state(trainer), continued)
    assert any(
        not torch.equal(after, prior)
        for after, prior in zip(state(original)[0], before[0], strict=True)
    )

    # Authenticate edited artifacts so restore must check the shared duplicates,
    # including masters and each Adam moment, rather than just file digests.
    shared_suffix = "lora_B" if projection == "down_proj" else "lora_A"
    key = f"{PREFIX}.1.{projections[0]}.{shared_suffix}.weight"
    survivor = state(original)
    for index, component in enumerate(("lora", "master", "exp_avg", "exp_avg_sq")):
        broken = tmp_path / component
        shutil.copytree(output, broken)
        filename = (
            "adapter_model.safetensors"
            if index == 0
            else prepared.manifest["parameters"][key][index - 1]
        )
        tensors = safetensors.torch.load_file(broken / filename)
        tensors[key].add_(1)
        safetensors.torch.save_file(tensors, broken / filename)
        manifest = deepcopy(prepared.manifest)
        manifest["files"][filename] = cp._file_digest(broken / filename)
        manifest["digest"] = cp._manifest_digest(manifest)
        (broken / cp.MANIFEST_FILE).write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="independent"):
            original.load_checkpoint(MaterializedCheckpoint("broken", str(broken)))
        assert set(original._checkpoint_slots) == {"student"}
        _assert_nested_tensors_equal(state(original), survivor)


@pytest.mark.parametrize("snapshot", [False, True])
def test_slot_reload_does_not_overwrite_surviving_slot(snapshot):
    lora = expert_lora()
    state = lora.sharded_lora_state_dict()
    a, b, c = [lm.LoRASlotRef("checkpoint", name) for name in ("a", "b", "c")]
    for ref in (a, b):
        lora.load_lora_slot(ref, state, requires_grad=True)
    survivor = lora._slot(b)
    lora._discard_lora_slot(a)
    if snapshot:
        assert lora._snapshot_lora_slot(b, c)
    else:
        assert lora.load_lora_slot(c, state, requires_grad=True)
    assert lora._slot(b) is survivor
    assert lora._slot(c) is not survivor
    assert len(set(lora._slot_keys.values())) == 2


class FakeInProj(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(14, 3), requires_grad=False)
        self.in_features = 3

    def forward(self, x):
        return (x.new_zeros((*x.shape[:-1], 14)), x), None


@pytest.mark.parametrize("parameterization", ["tied_v0", "tinker_v1"])
@pytest.mark.parametrize("expert", ["per_expert", "shared_outer"])
def test_qwen_producer_exporter_loader_and_inspection(
    parameterization, expert, tmp_path
):
    from art.megatron.model_support.lora_disk import (
        load_lora_tensors_for_megatron,
        save_vllm_lora_tensors,
    )
    from art.megatron.train import inspect_resident_lora
    from art.megatron.weights.param_name_canonicalization import (
        is_art_adapter_param_name,
    )

    layout = lm.LoraLayout(expert=expert, qwen_parameterization=parameterization)
    in_proj = lm.GatedDeltaNetInProjLoRA(
        "base_model.model.model.layers.0.linear_attn",
        FakeInProj(),
        SimpleNamespace(num_value_heads=1, qk_dim=2, v_dim=4),
        2,
        4,
        lora_layout=layout,
    )
    fc1 = lm.MLPExpertsLinearFC1LoRA(
        PREFIX,
        SimpleNamespace(out_features=4, in_features=3, weight0=torch.zeros(4, 3)),
        2,
        4,
        3,
        fused_gate_up=parameterization == "tied_v0",
        lora_layout=layout,
    )
    fc2 = lm.MLPExpertsLinearFC2LoRA(
        PREFIX,
        SimpleNamespace(out_features=3, in_features=2, weight0=torch.zeros(3, 2)),
        2,
        4,
        3,
        lora_layout=layout,
    )
    # A real TransformerLayer container lets the ordinary handler discover both
    # GDN and MoE wrappers; no forward kernel or model allocation is stubbed.
    layer = lm.TransformerLayer.__new__(lm.TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.layer_number = 1
    layer.self_attention = torch.nn.Module()
    layer.self_attention.in_proj = in_proj
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = torch.nn.Module()
    layer.mlp.experts.linear_fc1 = fc1
    layer.mlp.experts.linear_fc2 = fc2
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([layer])
    loras = list(lora_publish.iter_lora_modules(cast(Any, [model])))
    for lora in loras:
        with torch.no_grad():
            lora.B_T.normal_()
    handler = qwen.QWEN3_5_MOE_HANDLER
    exported = handler.build_adapter_weights_by_base([model])
    gdn_weights = exported["decoder.layers.0.self_attention.in_proj.weight"]
    assert {weight.adapter_key for weight in gdn_weights} == {
        "adapter_qkv",
        "adapter_z",
        "adapter_b",
        "adapter_a",
    }
    qkv_weight = next(
        weight for weight in gdn_weights if weight.adapter_key == "adapter_qkv"
    )
    assert qkv_weight.dim == (6 if parameterization == "tinker_v1" else 2)
    qkv_loras = (
        [in_proj.q_lora, in_proj.k_lora, in_proj.v_lora]
        if in_proj.split_qkv
        else [in_proj.qkv_lora]
    )
    expected_update = torch.cat(
        [lora.A_T @ lora.B_T * lora.scale for lora in qkv_loras], dim=-1
    )
    torch.testing.assert_close(
        qkv_weight.linear_in_weight.weight.T
        @ qkv_weight.linear_out_weight.weight.T
        * (qkv_weight.alpha / qkv_weight.dim),
        expected_update,
    )
    if parameterization == "tinker_v1":
        assert all(
            is_art_adapter_param_name(f"in_proj.{part}_lora.A_T")
            for part in ("q", "k", "v")
        )
        assert {
            f"linear_attn.in_proj_{part}.weight" for part in ("q", "k", "v")
        } <= set(
            handler._identity_lora_parameter_suffixes(
                ["in_proj_q", "in_proj_k", "in_proj_v"]
            )
        )
    inspection = inspect_resident_lora(
        cast(
            Any,
            SimpleNamespace(
                model=[model], model_support_handler=cast(Any, handler), rank=0
            ),
        ),
        cast(
            Any,
            SimpleNamespace(
                request_id="test",
                run_id="test",
                learner={
                    "training_session_id": "s",
                    "policy_step": 0,
                    "generation_id": "step-00000000-" + "0" * 32,
                    "adapter_path": "/tmp/adapter",
                },
                target_modules=("experts",),
            ),
        ),
    )
    assert inspection.module_count == len(loras)
    state = {
        key: value.clone()
        for lora in loras
        for key, value in lora.sharded_lora_state_dict().items()
    }
    config = layout.record_config(
        {
            "r": 2,
            "lora_alpha": 4.0,
            "target_modules": [
                "gate_proj",
                "up_proj",
                "down_proj",
                "in_proj_qkv",
                "in_proj_z",
            ],
        }
    )
    from art.trainer_rank import _lora_export

    plan = _lora_export._prepare_vllm_lora_publish(
        model=cast(Any, [model]),
        adapter_dtypes={},
        handler=handler,
        adapter_config=config,
        rank=0,
        world_size=1,
        runtime=(0, torch.device("cpu")),
    )
    inputs = _lora_export._exchange_vllm_lora_publish(plan)
    assert inputs is not None
    vllm, config = _lora_export._build_vllm_lora_tensors_from_inputs(inputs)
    vllm = {key: value.contiguous() for key, value in vllm.items()}
    if parameterization == "tinker_v1":
        from art.megatron.model_support.lora_disk import training_adapter_config

        assert config["r"] == 6 and config["lora_alpha"] == 12.0
        assert training_adapter_config(config)["r"] == 2
        tied = qwen._from_vllm_tied_lora_tensors(vllm, adapter_config=config)
        for prefix in {key.rsplit(".lora_", 1)[0] for key in tied}:
            a, b = [tied[prefix + f".lora_{factor}.weight"] for factor in ("A", "B")]
            if prefix.endswith(".in_proj_qkv"):
                source = [prefix.removesuffix("qkv") + part for part in ("q", "k", "v")]
            elif prefix.endswith(".gate_up_proj"):
                source = [
                    prefix.removesuffix("gate_up_proj") + part + "_proj"
                    for part in ("gate", "up")
                ]
            else:
                source = [prefix]
            expected = torch.cat(
                [
                    state[name + ".lora_A.weight"].T
                    @ state[name + ".lora_B.weight"].T
                    * 2
                    for name in source
                ],
                dim=-1,
            )
            torch.testing.assert_close(
                a.T @ b.T * (config["lora_alpha"] / config["r"]), expected
            )
        corrupted = {key: value.clone() for key, value in vllm.items()}
        qkv_b = next(
            key for key in corrupted if key.endswith("in_proj_qkv.lora_B.weight")
        )
        corrupted[qkv_b][0, 2] = 1.0
        with pytest.raises(ValueError, match="independent training factors"):
            handler.from_vllm_lora_tensors(corrupted, adapter_config=config)
    save_vllm_lora_tensors(tmp_path, vllm, config)
    loaded = load_lora_tensors_for_megatron(
        tmp_path,
        handler=cast(Any, handler),
        provider=SimpleNamespace(_art_lora_layout=layout),
    )
    assert state.keys() == loaded.keys()
    for lora in loras:
        with torch.no_grad():
            lora.A_T.zero_()
            lora.B_T.zero_()
        lora.load_lora(loaded)
        for key, value in lora.sharded_lora_state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    x = torch.randn(2, 1, 3)
    actual, _ = in_proj(x)
    qkv_loras = (
        [in_proj.q_lora, in_proj.k_lora, in_proj.v_lora]
        if in_proj.split_qkv
        else [in_proj.qkv_lora]
    )
    expected = torch.cat(
        [*(lora(x) for lora in qkv_loras), in_proj.z_lora(x), x.new_zeros(2, 1, 2)],
        dim=-1,
    )
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="layout mismatch"):
        load_lora_tensors_for_megatron(
            tmp_path,
            handler=cast(Any, handler),
            provider=SimpleNamespace(
                _art_lora_layout=lm.LoraLayout(shared_outer_kernel=False)
            ),
        )


def test_layout_is_explicit_frozen_and_restore_checked(monkeypatch):
    from dataclasses import FrozenInstanceError

    from art.trainer_rank import _slots

    for variable in (
        "ART_MEGATRON_EXPERT_LORA_LAYOUT",
        "ART_MEGATRON_QWEN_LORA_PARAMETERIZATION",
        "ART_MEGATRON_SHARED_OUTER_KERNEL",
    ):
        monkeypatch.setenv(variable, "invalid")
    assert expert_lora().shared_factor is None
    assert not lm.GatedDeltaNetInProjLoRA(
        "test",
        FakeInProj(),
        SimpleNamespace(num_value_heads=1, qk_dim=2, v_dim=4),
        2,
        4,
    ).split_qkv
    assert not hasattr(lm, "UnembeddingLoRA")
    layout = lm.LoraLayout(
        expert="shared_outer",
        qwen_parameterization="tinker_v1",
        shared_outer_kernel=False,
    )
    config = {
        "base_model_name_or_path": "test/model",
        "r": 2,
        "lora_alpha": 4.0,
        "target_modules": ["experts"],
    }
    trainer = SimpleNamespace(runtime=SimpleNamespace(lora_layout=layout))
    saved = _slots._validate_checkpoint_adapter_config(
        cast(Any, trainer), "x", config, alpha=4.0
    )
    assert saved is not None
    assert cast(Any, saved)["art_lora_layout"] == asdict(layout)
    layout.validate_config(dict(saved))
    with pytest.raises(ValueError, match="layout mismatch"):
        layout.validate_config(config)
    assert lm.LoraLayout().record_config(config) == config
    with pytest.raises(FrozenInstanceError):
        setattr(layout, "expert", "per_expert")
    with pytest.raises(ValueError, match="Unknown"):
        lm.LoraLayout(expert=cast(Any, "unknown"))


def _shared_reduction_worker(rank, path, etp_two):
    from megatron.core.transformer.module import param_is_not_shared
    import torch.distributed as dist

    from art.megatron.training import finalize_grads as fg

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    try:
        ep_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        dp_groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
        singletons = [dist.new_group([r]) for r in range(4)]
        ep, dp, etp = (
            (dist.group.WORLD, singletons[rank], ep_groups[rank // 2])
            if etp_two
            else (ep_groups[rank // 2], dp_groups[rank % 2], singletons[rank])
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                lm.ps, "get_expert_tensor_and_model_parallel_group", lambda **_: ep
            )
            patch.setattr(lm.ps, "get_expert_tensor_parallel_group", lambda **_: etp)
            patch.setattr(lm.ps, "get_tensor_model_parallel_group", lambda **_: etp)
            patch.setattr(lm.ps, "get_expert_data_parallel_group", lambda **_: dp)
            patch.setattr(
                lm.ps,
                "get_expert_model_parallel_rank",
                lambda: rank // 2 if etp_two else rank % 2,
            )
            patch.setattr(
                lm.ps,
                "get_expert_data_parallel_rank",
                lambda: 0 if etp_two else rank // 2,
            )
            patch.setattr(lm.ps, "get_tensor_model_parallel_world_size", lambda: 1)
            for projection in ("gate_up_proj", "down_proj"):
                torch.manual_seed(73 + rank)
                spec = lm.LoRAParallelSpec(shard_domain="expert_tp")
                lora = lm.LoRA(
                    f"{PREFIX}.{{expert}}.{projection}",
                    3,
                    4,
                    2,
                    2.0,
                    torch.float32,
                    torch.device("cpu"),
                    num_local_experts=2,
                    a_parallel_spec=spec,
                    b_parallel_spec=spec,
                    allreduce=False,
                    lora_layout=lm.LoraLayout(expert="shared_outer"),
                )
                shared = lora.A_T if projection == "gate_up_proj" else lora.B_T
                initial = [torch.empty_like(shared) for _ in range(ep.size())]
                dist.all_gather(initial, shared.detach(), group=ep)
                assert all(torch.equal(value, initial[0]) for value in initial)
                other = lora.B_T if projection == "gate_up_proj" else lora.A_T
                with torch.no_grad():
                    shared.fill_(0.25)
                    other.copy_(
                        torch.arange(other.numel()).reshape(other.shape) / 10 + 1
                    )
                # Independent per-expert leaves give a reference gradient, summed
                # over EP and expert-DP batches, including an empty expert rank.
                expected = torch.zeros_like(shared)
                for batch_rank in range(4):
                    a = lora.A_T.detach().expand(2, -1, -1).clone().requires_grad_()
                    b = lora.B_T.detach().expand(2, -1, -1).clone().requires_grad_()
                    xs = [
                        torch.arange(
                            (0 if batch_rank % 2 else 2) * 3, dtype=torch.float32
                        ).reshape(-1, 3)
                        + batch_rank
                        + e
                        for e in range(2)
                    ]
                    sum((x @ a[e] @ b[e]).sum() for e, x in enumerate(xs)).backward()
                    expected.add_(
                        (a.grad if projection == "gate_up_proj" else b.grad).sum(
                            0, keepdim=True
                        )
                    )
                active = lora.active_lora_tensors()
                assert active is not None
                a, b, _ = active
                xs = [
                    torch.arange(
                        (0 if rank % 2 else 2) * 3, dtype=torch.float32
                    ).reshape(-1, 3)
                    + rank
                    + e
                    for e in range(2)
                ]
                sum((x @ a[e] @ b[e]).sum() for e, x in enumerate(xs)).backward()
                local = shared.grad.clone()
                for p in lora.parameters():
                    p.main_grad = p.grad.clone()

                # MCore finalize owns expert-DP. Use the real Gloo collective for
                # that boundary, then call ART's actual extended finalizer.
                def finalize(model, **kwargs):
                    for p in model[0].parameters():
                        dist.all_reduce(p.main_grad, group=dp)

                patch.setattr(fg, "finalize_model_grads", finalize)
                fg.finalize_model_grads_extended(cast(Any, [lora]))
                torch.testing.assert_close(shared.main_grad, expected)
                shared.grad = local
                # Dynamic slots must inherit the same sync and norm metadata.
                dynamic = lm.LoRASlot(
                    ref=lm.LoRASlotRef("checkpoint", "x"),
                    a_t=lora.A_T,
                    b_t=lora.B_T,
                    alpha=2.0,
                    a_template=lora.A_T,
                    b_template=lora.B_T,
                    requires_grad=True,
                )
                p = dynamic.A_T if projection == "gate_up_proj" else dynamic.B_T
                p.grad = local
                reduced = TrainerRank._reduce_dynamic_grads(
                    cast(Any, None), [p], scale_grads=1.0
                )[0]
                torch.testing.assert_close(reduced, expected)
                (norm,) = _distributed_grad_norms([([p], [reduced])])
                assert norm == pytest.approx(expected.norm().item())
                included = torch.tensor(
                    int(param_is_not_shared(shared) and dp.rank() == 0)
                )
                dist.all_reduce(included)
                assert included.item() == 1
            patch.setattr(
                lm.ps,
                "get_expert_tensor_and_model_parallel_group",
                lambda **_: singletons[rank],
            )
            assert fg._resolve_domain_group("expert_tp_ep") is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("etp_two", [False, True], ids=["EP2-EDP2", "EP2-ETP2"])
def test_shared_outer_distributed_finalize_and_dynamic(tmp_path, etp_two):
    torch.multiprocessing.start_processes(
        _shared_reduction_worker,
        args=(str(tmp_path / "gloo"), etp_two),
        nprocs=4,
        start_method="fork",
    )


def _tensor_digest(values):
    import hashlib

    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize(
    "projection,initial,state_digest,packed_digest",
    [
        (
            "gate_up_proj",
            "6781668ea13c12587afea975807e373fb22cf96caa5626eb59dfb4d1aa294d31",
            "46b9bb995b5d158691ef18dc340c9012228eb50c2ab5649d300a04a958b69ed6",
            "8667e1ad50955f8472ea0e3fc2ebba9feca18176ebf0e863c19062286bdabd79",
        ),
        (
            "down_proj",
            "0a18651746df215132ec1836b8e0b0317cd7b0c4eb810945337fd4a8f29e78bf",
            "020f5fe648c42d04227799df9b4178f498ca6e712d6ac7038639a4f344de1cdc",
            "d4e15d45d8889a1e22d43a73c087531891ef48b45e5c3fbccace4b7a4ec409c6",
        ),
    ],
)
def test_default_representation_matches_main(
    projection, initial, state_digest, packed_digest
):
    # Frozen from e18c570's actual LoRA class, seed 17, FP32, E3/H3/O4/R2.
    # Hashes include names, dtypes, shapes and bytes, including the initial zeros.
    torch.manual_seed(17)
    lora = lm.LoRA(
        f"{PREFIX}.{{expert}}.{projection}",
        3,
        4,
        2,
        4.0,
        torch.float32,
        torch.device("cpu"),
        num_local_experts=3,
    )
    assert _tensor_digest(lora.sharded_lora_state_dict()) == initial
    assert (tuple(lora.A_T.shape), tuple(lora.B_T.shape)) == ((3, 3, 2), (3, 2, 4))
    with torch.no_grad():
        lora.B_T.copy_(torch.arange(lora.B_T.numel()).reshape(lora.B_T.shape) / 10)
    assert _tensor_digest(lora.sharded_lora_state_dict()) == state_digest
    views = {
        f"{expert}/{factor}": tensor
        for expert in range(3)
        for factor, tensor in zip(
            "AB", adapter_export._adapter_tensors(lora, expert), strict=True
        )
    }
    assert (
        _tensor_digest(views)
        == "3f876f5c21077554419b61f502949ccdb2c3a58eea2eb297c23eabee7a92f195"
    )
    tensors, metadata = lora_publish.collect_local_packed_expert_entries(
        cast(Any, [lora]),
        {},
        owner_rank=0,
        packed_expert_groups=qwen.QWEN3_5_MOE_HANDLER.expert_packed_lora_groups(),
    )
    packed = lora_publish.merge_packed_expert_adapter_entries(
        metadata, {(0, key): value for key, value in tensors.items()}
    )
    assert _tensor_digest(packed) == packed_digest


def test_layout_reaches_construction_hook_and_lm_head_is_ignored(monkeypatch):
    from art.megatron import train

    layout = lm.LoraLayout(expert="shared_outer", qwen_parameterization="tinker_v1")
    hooks, received = [], []
    provider = SimpleNamespace(register_pre_wrap_hook=hooks.append)
    monkeypatch.setattr(
        train,
        "apply_lora_adapters",
        lambda model, provider, **kwargs: received.append(kwargs["lora_layout"]),
    )
    train._register_trainable_parameter_mode(
        provider, trainable_parameter_mode="lora", lora_layout=layout
    )
    model = torch.nn.Module()
    model.output_layer = torch.nn.Linear(3, 4, bias=False)
    model.embedding = torch.nn.Embedding(4, 3)
    model.output_layer.weight = model.embedding.weight
    head, weight = model.output_layer, model.output_layer.weight
    for hook in hooks:
        hook([model])
    assert received == [layout]
    for handler in (qwen.QWEN3_5_DENSE_HANDLER, qwen.QWEN3_5_MOE_HANDLER):
        handler.apply_lora_adapters(
            [model],
            provider,
            target_modules=["lm_head"],
            rank=2,
            alpha=4,
            lora_layout=layout,
        )
        assert model.output_layer is head and model.output_layer.weight is weight
        assert list(lora_publish.iter_lora_modules(cast(Any, [model]))) == []


def test_checkpoint_restore_rejects_layout_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "_ensure_group", lambda _: None)
    monkeypatch.setattr(cp, "_gather", lambda value, _: [value])
    source = cp.PreparedCheckpoint(tmp_path, {}, (), None, "same")
    trainer = SimpleNamespace(
        runtime=SimpleNamespace(lora_layout=lm.LoraLayout(expert="shared_outer"))
    )
    with pytest.raises(ValueError, match="layout mismatch"):
        cp._load_checkpoint(cast(Any, trainer), source, "x")


@pytest.mark.parametrize("shared_a", [False, True])
def test_shared_outer_projection_uses_quack_override(shared_a, monkeypatch):
    from art.megatron.kernels import shared_outer_lora as kernel

    monkeypatch.setenv("ART_QUACK_PROJ_TILE_N", "16")
    monkeypatch.setenv("ART_QUACK_MATMUL_TILE_N", "64")
    calls = []

    def grouped(
        x, weight, *, out_features, expert_offsets, tile_n, alpha=1.0, **kwargs
    ):
        calls.append((out_features, tile_n))
        return (
            torch.cat(
                [
                    x[start:end] @ weight[e].T
                    for e, (start, end) in enumerate(
                        zip(expert_offsets[:-1], expert_offsets[1:], strict=True)
                    )
                ]
            )
            * alpha
        )

    def grad_weight(x, y, *, expert_offsets, alpha=1.0, **kwargs):
        return (
            torch.stack(
                [
                    x[:, start:end] @ y[:, start:end].T
                    for start, end in zip(
                        expert_offsets[:-1], expert_offsets[1:], strict=True
                    )
                ]
            )
            * alpha
        )

    monkeypatch.setattr(kernel, "_varlen_quack_gemm", grouped)
    monkeypatch.setattr(kernel, "_varlen_quack_gemm_k", grad_weight)
    x = torch.randn(3, 5, requires_grad=True)
    a = torch.randn(1 if shared_a else 2, 5, 4, requires_grad=True)
    b = torch.randn(2 if shared_a else 1, 4, 7, requires_grad=True)
    expected_inputs = [tensor.detach().clone().requires_grad_() for tensor in (x, a, b)]
    xx, aa, bb = expected_inputs
    expected = (
        torch.cat(
            [
                xx[:2] @ aa[0] @ bb[0],
                xx[2:] @ aa[0 if shared_a else 1] @ bb[1 if shared_a else 0],
            ]
        )
        * 0.75
    )
    expected.sum().backward()
    actual = kernel._SharedOuter.apply(x, a, b, torch.tensor([2, 1]), 0.75, shared_a)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected)
    for tensor, reference in zip((x, a, b), expected_inputs, strict=True):
        torch.testing.assert_close(tensor.grad, reference.grad)
    # The padded rank is 8. Both forward and backward projections use the
    # existing projection override; full-width products keep matmul tiles.
    assert (8, 16) in calls
    assert all(tile == 16 if width == 8 else tile == 64 for width, tile in calls)
