from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from pydantic import ValidationError
import pytest
import torch

from art.distributed.specs import GpuPlacement, TrainerMeshSpec
from art.megatron import identity_lora
from art.megatron.runtime import build as runtime_build
from art.megatron.runtime import monarch as runtime_monarch
from art.megatron.runtime.specs import TrainerRuntimeSpec
from art.megatron.training.bootstrap import LocalMegatronTrainingSlotConfig
from art.serverless import backend as serverless_backend
from art.serverless.contracts import AdapterSpec, TrainingRunSpec
from art.types import MegatronRuntimeConfig, MegatronTopologyConfig


def _runtime_spec(
    *,
    rank: int = 1,
    moe_parameterization: str = "per_expert",
) -> TrainerRuntimeSpec:
    topology = MegatronTopologyConfig(tp=1, cp=1, ep=1, pp=1, etp=1)
    return TrainerRuntimeSpec(
        art_revision="art",
        model_identifier="model",
        model_revision="revision",
        model_support_key="handler",
        handler_name="handler",
        lora_rank=rank,
        lora_target_modules=("experts",),
        lora_moe_parameterization=moe_parameterization,
        dtype="bfloat16",
        trainer_mesh=TrainerMeshSpec(
            ranks=(GpuPlacement(host_id="host", gpu_id=0),),
            topology=topology,
        ),
        packed_sequence_length=1024,
        compile_enabled=True,
        compile_fingerprint="compile",
        optimizer_semantic_fingerprint="0" * 64,
        optimizer_layout_fingerprint="optimizer",
    )


def test_adapter_contract_defaults_and_rejects_unknown_values() -> None:
    fields = {"rank": 1, "target_modules": ("experts",)}
    default = AdapterSpec(**fields)
    assert default.moe_parameterization == "per_expert"
    assert default.model_dump(mode="json")["moe_parameterization"] == "per_expert"
    shared_outer = AdapterSpec(**fields, moe_parameterization="shared_outer")
    assert shared_outer.moe_parameterization == "shared_outer"
    run = TrainingRunSpec(
        run_name="run",
        base_model="model",
        adapter=shared_outer,
    )
    persisted = TrainingRunSpec.model_validate_json(run.model_dump_json())
    assert persisted.adapter.moe_parameterization == "shared_outer"
    with pytest.raises(ValidationError, match="moe_parameterization"):
        AdapterSpec(**fields, moe_parameterization="invalid")


def test_serverless_backend_propagates_parameterization(monkeypatch) -> None:
    import art.megatron.model_support as model_support

    handler = SimpleNamespace(is_moe=True)
    monkeypatch.setattr(
        model_support,
        "get_model_support_spec",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        model_support,
        "get_model_support_handler_for_spec",
        lambda _spec: handler,
    )
    monkeypatch.setattr(
        model_support,
        "default_target_modules_for_model",
        lambda *_args, **_kwargs: ["experts"],
    )
    model = SimpleNamespace(
        base_model="model",
        lora_config={"moe_parameterization": "shared_outer"},
        _internal_config={},
    )

    assert (
        serverless_backend._adapter_spec(model).moe_parameterization == "shared_outer"
    )
    model.lora_config = {}
    assert serverless_backend._adapter_spec(model).moe_parameterization == "per_expert"
    model.lora_config = {"moe_parameterization": "invalid"}
    with pytest.raises(ValidationError, match="moe_parameterization"):
        serverless_backend._adapter_spec(model)


def test_runtime_identity_includes_parameterization() -> None:
    per_expert = _runtime_spec()
    shared_outer = _runtime_spec(moe_parameterization="shared_outer")

    assert per_expert.fingerprint != shared_outer.fingerprint
    assert (
        per_expert.compatibility_fingerprint == shared_outer.compatibility_fingerprint
    )
    assert (
        per_expert.compatibility_fingerprint
        == _runtime_spec(rank=32).compatibility_fingerprint
    )
    with pytest.raises(ValidationError, match="lora_moe_parameterization"):
        _runtime_spec(moe_parameterization="invalid")


def test_runtime_builder_propagates_parameterization(monkeypatch) -> None:
    topology = MegatronTopologyConfig(tp=1, cp=1, ep=1, pp=1, etp=1)
    runtime_config = MegatronRuntimeConfig(
        topology=topology,
        packed_sequence_length=1024,
    )
    mesh = TrainerMeshSpec(
        ranks=(GpuPlacement(host_id="host", gpu_id=0),),
        topology=topology,
    )
    runtime = SimpleNamespace(
        runtime_id="runtime",
        topology=SimpleNamespace(
            trainer=mesh,
            cluster=SimpleNamespace(cache_root=None, nixl_transport=None),
        ),
    )
    handler = SimpleNamespace(key="handler", is_moe=True)
    monkeypatch.setattr(runtime_build, "art_source_revision", lambda: "art")
    monkeypatch.setattr(
        runtime_build, "get_megatron_runtime_config", lambda: runtime_config
    )
    monkeypatch.setattr(
        runtime_build,
        "get_model_support_spec",
        lambda *_args, **_kwargs: SimpleNamespace(key="handler"),
    )
    monkeypatch.setattr(
        runtime_build,
        "get_model_support_handler_for_spec",
        lambda _spec: handler,
    )

    config = {
        "init_args": {"model_name": "model", "dtype": "bfloat16"},
        "lora_config": {
            "rank": 1,
            "alpha": 32,
            "target_modules": ["experts"],
            "moe_parameterization": "shared_outer",
        },
    }
    spec = runtime_build.build_trainer_runtime_spec(
        runtime,
        base_model="model",
        config=config,
        enable_expert_replay=False,
        offload_between_jobs=False,
    )

    assert spec.lora_moe_parameterization == "shared_outer"
    topology_cp2 = MegatronTopologyConfig(tp=1, cp=2, ep=1, pp=1, etp=1)
    runtime.topology.trainer = TrainerMeshSpec(
        ranks=(
            GpuPlacement(host_id="host", gpu_id=0),
            GpuPlacement(host_id="host", gpu_id=1),
        ),
        topology=topology_cp2,
    )
    monkeypatch.setattr(
        runtime_build,
        "get_megatron_runtime_config",
        lambda: MegatronRuntimeConfig(
            topology=topology_cp2,
            packed_sequence_length=2048,
        ),
    )
    migrated_spec = runtime_build.build_trainer_runtime_spec(
        runtime,
        base_model="model",
        config=config,
        enable_expert_replay=False,
        offload_between_jobs=False,
    )
    assert migrated_spec.optimizer_semantic_fingerprint == (
        spec.optimizer_semantic_fingerprint
    )
    assert migrated_spec.optimizer_layout_fingerprint != (
        spec.optimizer_layout_fingerprint
    )

    config["lora_config"]["target_modules"] = ["gate", "experts"]
    changed_targets_spec = runtime_build.build_trainer_runtime_spec(
        runtime,
        base_model="model",
        config=config,
        enable_expert_replay=False,
        offload_between_jobs=False,
    )
    assert changed_targets_spec.optimizer_semantic_fingerprint != (
        migrated_spec.optimizer_semantic_fingerprint
    )

    config["lora_config"]["moe_parameterization"] = "invalid"
    with pytest.raises(ValidationError, match="lora_moe_parameterization"):
        runtime_build.build_trainer_runtime_spec(
            runtime,
            base_model="model",
            config=config,
            enable_expert_replay=False,
            offload_between_jobs=False,
        )


def test_monarch_build_sets_explicit_provider_parameterization(monkeypatch) -> None:
    from art.megatron import train

    captured = {}

    def build_training_runtime(**kwargs):
        provider = SimpleNamespace()
        kwargs["provider_configure"](provider)
        captured["parameterization"] = provider._art_lora_moe_parameterization
        return SimpleNamespace()

    monkeypatch.setattr(train, "build_training_runtime", build_training_runtime)
    runtime_monarch._build_training_runtime(
        _runtime_spec(moe_parameterization="shared_outer"),
        rank=0,
    )

    assert captured["parameterization"] == "shared_outer"


def test_dense_local_slot_accepts_parameterization(tmp_path: Path) -> None:
    runtime = MegatronRuntimeConfig(
        topology=MegatronTopologyConfig(tp=1, cp=1, ep=1, pp=1, etp=1),
        packed_sequence_length=1024,
    )
    fields = {
        "slot_id": "slot",
        "artifact_root": str(tmp_path),
        "base_model": "dense-model",
        "trainer_gpu_ids": (0,),
        "runtime": runtime,
        "run_residency": {
            "limits": {
                "l1_gpu": {"max_bytes": 1},
                "l2_cpu": {"max_bytes": 1},
                "l3_nvme": {"max_bytes": 1},
            },
            "nvme": {"root": str(tmp_path / "nvme")},
        },
    }

    config = LocalMegatronTrainingSlotConfig(
        **fields, lora_moe_parameterization="shared_outer"
    )
    assert config.lora_moe_parameterization == "shared_outer"
    with pytest.raises(ValidationError, match="lora_moe_parameterization"):
        LocalMegatronTrainingSlotConfig(**fields, lora_moe_parameterization="invalid")


def test_identity_adapter_config_records_parameterization(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeLoraConfig:
        def __init__(self, **values):
            self.values = values

        def to_dict(self):
            return dict(self.values)

    class FakeModel(torch.nn.Module):
        pass

    class FakePeftModel:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    accelerate = ModuleType("accelerate")
    accelerate.init_empty_weights = nullcontext
    peft = ModuleType("peft")
    peft.get_peft_model = lambda *_args, **_kwargs: FakePeftModel()
    peft_tuners = ModuleType("peft.tuners")
    peft_lora = ModuleType("peft.tuners.lora")
    peft_config = ModuleType("peft.tuners.lora.config")
    peft_config.LoraConfig = FakeLoraConfig
    transformers = ModuleType("transformers")
    transformers.AutoConfig = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: SimpleNamespace()
    )
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_config=lambda *_args, **_kwargs: FakeModel()
    )
    for name, module in {
        "accelerate": accelerate,
        "peft": peft,
        "peft.tuners": peft_tuners,
        "peft.tuners.lora": peft_lora,
        "peft.tuners.lora.config": peft_config,
        "transformers": transformers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    recorded = {}
    monkeypatch.setattr(
        identity_lora,
        "normalize_lora_checkpoint_to_vllm",
        lambda _path, *, handler, adapter_config: recorded.update(adapter_config),
    )
    handler = SimpleNamespace(
        is_moe=False,
        identity_lora_model_config=lambda config: config,
        identity_lora_target_parameters=lambda _model, *, target_modules: [],
    )

    identity_lora.create_identity_lora(
        "model",
        str(tmp_path / "adapter"),
        rank=1,
        target_modules=["experts"],
        moe_parameterization="shared_outer",
        handler=handler,
    )
    assert recorded["moe_parameterization"] == "shared_outer"
    with pytest.raises(ValueError, match="unsupported MoE LoRA parameterization"):
        identity_lora.create_identity_lora(
            "model",
            str(tmp_path / "invalid"),
            moe_parameterization="invalid",
        )
