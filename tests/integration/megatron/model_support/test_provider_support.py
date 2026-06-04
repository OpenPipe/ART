from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("megatron.bridge")

from megatron.core.transformer.enums import AttnBackend

from art.megatron.flex_attention import FlexDotProductAttention
from art.megatron.lora import default_lora_rank_for_handler
from art.megatron.model_support.registry import (
    UnsupportedModelArchitectureError,
    get_model_support_handler,
    get_model_support_spec,
)
import art.megatron.provider as provider_module


class _FakeProvider:
    def __init__(self) -> None:
        self.transformer_layer_spec = self._base_layer_spec
        self.finalized = False
        self.overlap_moe_expert_parallel_comm = False
        self.num_moe_experts = 0

    def _base_layer_spec(
        self, config: object, vp_stage: int | None = None
    ) -> SimpleNamespace:
        del config, vp_stage
        return SimpleNamespace(
            submodules=SimpleNamespace(
                self_attention=SimpleNamespace(
                    submodules=SimpleNamespace(core_attention=object())
                )
            ),
        )

    def finalize(self) -> None:
        self.finalized = True


class _FakeHybridProvider(_FakeProvider):
    def _base_layer_spec(
        self, config: object, vp_stage: int | None = None
    ) -> SimpleNamespace:
        del config, vp_stage
        gdn_layer = SimpleNamespace(
            submodules=SimpleNamespace(
                self_attention=SimpleNamespace(submodules=SimpleNamespace())
            )
        )
        attention_layer = SimpleNamespace(
            submodules=SimpleNamespace(
                self_attention=SimpleNamespace(
                    submodules=SimpleNamespace(core_attention=object())
                )
            ),
        )
        return SimpleNamespace(layer_specs=[gdn_layer, attention_layer])


class _FakeBridge:
    def __init__(self, *, model_bridge: object, provider: _FakeProvider) -> None:
        self._model_bridge = model_bridge
        self._provider = provider
        self.hf_pretrained = SimpleNamespace(model_name_or_path="unused")

    def to_megatron_provider(self) -> _FakeProvider:
        return self._provider


def test_openpipe_qwen3_14b_instruct_uses_qwen3_dense_support() -> None:
    spec = get_model_support_spec("OpenPipe/Qwen3-14B-Instruct")
    handler = get_model_support_handler("OpenPipe/Qwen3-14B-Instruct")

    assert spec.key == "qwen3_dense"
    assert spec.native_vllm_lora_status == "validated"
    assert handler.key == "qwen3_dense"


def test_megatron_lora_rank_defaults_by_architecture() -> None:
    dense_handler = get_model_support_handler("OpenPipe/Qwen3-14B-Instruct")
    moe_handler = get_model_support_handler("Qwen/Qwen3-30B-A3B-Instruct-2507")

    assert default_lora_rank_for_handler(dense_handler) == 8
    assert default_lora_rank_for_handler(moe_handler) == 1


def test_get_provider_accepts_registry_supported_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider()
    provider.num_moe_experts = 8
    fake_bridge = _FakeBridge(
        model_bridge=object(),
        provider=provider,
    )
    monkeypatch.setattr(
        provider_module.AutoBridge,
        "from_hf_pretrained",
        lambda *args, **kwargs: fake_bridge,
    )
    monkeypatch.setattr(provider_module.torch.cuda, "device_count", lambda: 2)

    resolved = provider_module.get_provider("Qwen/Qwen3-30B-A3B-Instruct-2507")

    assert resolved is provider
    assert provider.finalized is True
    assert resolved.attention_backend is AttnBackend.auto
    assert resolved.recompute_granularity == "full"
    assert resolved.recompute_method == "uniform"
    assert resolved.recompute_num_layers == 1
    assert resolved.tensor_model_parallel_size == 2
    assert resolved.context_parallel_size == 1
    assert resolved.pipeline_model_parallel_size == 1
    assert resolved.expert_model_parallel_size == 2
    assert resolved.expert_tensor_parallel_size == 1
    assert resolved.sequence_parallel is True
    assert resolved.moe_shared_expert_overlap is False
    assert resolved.moe_router_dtype == "fp32"
    assert resolved.moe_aux_loss_coeff == 0.0
    assert resolved.calculate_per_token_loss is True

    layer_spec = cast(Any, resolved.transformer_layer_spec)(resolved, vp_stage=7)
    assert (
        layer_spec.submodules.self_attention.submodules.core_attention
        is FlexDotProductAttention
    )


def test_qwen35_provider_uses_handler_shared_expert_runtime_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron.model_support.handlers import qwen3_5 as qwen35_handler_module

    provider = _FakeProvider()
    fake_bridge = _FakeBridge(
        model_bridge=object(),
        provider=provider,
    )
    monkeypatch.setattr(
        provider_module.AutoBridge,
        "from_hf_pretrained",
        lambda *args, **kwargs: fake_bridge,
    )
    monkeypatch.setattr(provider_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        qwen35_handler_module,
        "_qwen35_provider_types",
        lambda: (_FakeProvider,),
    )
    monkeypatch.setattr(
        qwen35_handler_module,
        "_require_qwen35_provider_symbols",
        lambda: (
            object(),
            (_FakeProvider,),
            lambda block_spec, attention_module: None,
            provider._base_layer_spec,
        ),
    )

    resolved = provider_module.get_provider("Qwen/Qwen3.5-35B-A3B")

    assert resolved.moe_shared_expert_overlap is False
    assert resolved.scatter_embedding_sequence_parallel is True


def test_get_provider_rejects_unregistered_model_before_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def from_hf_pretrained(*args: object, **kwargs: object) -> object:
        raise AssertionError("AutoBridge should not be called for unsupported models")

    monkeypatch.setattr(
        provider_module.AutoBridge, "from_hf_pretrained", from_hf_pretrained
    )

    with pytest.raises(
        UnsupportedModelArchitectureError,
        match="has not passed the Megatron model-support workflow",
    ):
        provider_module.get_provider("unsupported/model")


def test_get_provider_preserves_hybrid_layer_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeHybridProvider()
    fake_bridge = _FakeBridge(
        model_bridge=object(),
        provider=provider,
    )
    monkeypatch.setattr(
        provider_module.AutoBridge,
        "from_hf_pretrained",
        lambda *args, **kwargs: fake_bridge,
    )
    monkeypatch.setattr(provider_module.torch.cuda, "device_count", lambda: 1)

    resolved = provider_module.get_provider(
        "unused-qwen",
        allow_unvalidated_arch=True,
    )
    layer_spec = cast(Any, resolved).transformer_layer_spec(resolved, vp_stage=0)

    assert hasattr(layer_spec, "layer_specs")
    gdn_layer, attention_layer = cast(Any, layer_spec).layer_specs
    assert not hasattr(gdn_layer.submodules.self_attention.submodules, "core_attention")
    assert (
        attention_layer.submodules.self_attention.submodules.core_attention
        is FlexDotProductAttention
    )


def test_finalize_provider_bundle_uses_post_prepare_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider()
    setattr(provider, "num_moe_experts", 8)
    fake_bridge = _FakeBridge(
        model_bridge=object(),
        provider=provider,
    )
    dispatcher_calls: list[tuple[int, int, str]] = []
    monkeypatch.setattr(
        provider_module.AutoBridge,
        "from_hf_pretrained",
        lambda *args, **kwargs: fake_bridge,
    )
    monkeypatch.setattr(provider_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        provider_module,
        "apply_flex_dispatcher_backend",
        lambda provider, moe_flex_dispatcher_backend: dispatcher_calls.append(
            (
                int(provider.tensor_model_parallel_size),
                int(provider.expert_model_parallel_size),
                cast(str, moe_flex_dispatcher_backend),
            )
        ),
    )

    bundle = provider_module.prepare_provider_bundle("Qwen/Qwen3-30B-A3B-Instruct-2507")

    assert provider.finalized is False
    assert getattr(provider, "tensor_model_parallel_size") == 2
    assert getattr(provider, "expert_model_parallel_size") == 2

    bundle.provider.tensor_model_parallel_size = 1
    bundle.provider.expert_model_parallel_size = 1
    bundle.provider.sequence_parallel = False
    provider_module.finalize_provider_bundle(bundle)

    assert dispatcher_calls == []
    assert provider.finalized is True
    assert getattr(provider, "sequence_parallel") is False


def test_get_provider_bundle_honors_single_gpu_env_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider()
    fake_bridge = _FakeBridge(
        model_bridge=object(),
        provider=provider,
    )
    monkeypatch.setattr(
        provider_module.AutoBridge,
        "from_hf_pretrained",
        lambda *args, **kwargs: fake_bridge,
    )
    monkeypatch.setattr(provider_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE", "1")
    monkeypatch.setenv("ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE", "1")
    monkeypatch.setenv("ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE", "1")

    bundle = provider_module.get_provider_bundle("Qwen/Qwen3-30B-A3B-Instruct-2507")
    resolved = bundle.provider

    assert resolved.tensor_model_parallel_size == 1
    assert resolved.context_parallel_size == 1
    assert resolved.pipeline_model_parallel_size == 1
    assert resolved.expert_model_parallel_size == 1
    assert resolved.expert_tensor_parallel_size == 1
    assert resolved.sequence_parallel is False
    assert resolved.recompute_granularity == "full"
    assert resolved.recompute_method == "uniform"
    assert resolved.recompute_num_layers == 1

    transformer_layer_spec = cast(Any, resolved.transformer_layer_spec)
    layer_spec = transformer_layer_spec(resolved, vp_stage=0)
    assert (
        layer_spec.submodules.self_attention.submodules.core_attention
        is FlexDotProductAttention
    )


def test_get_provider_bundle_disables_recompute_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider()
    fake_bridge = _FakeBridge(
        model_bridge=object(),
        provider=provider,
    )
    monkeypatch.setattr(
        provider_module.AutoBridge,
        "from_hf_pretrained",
        lambda *args, **kwargs: fake_bridge,
    )
    monkeypatch.setattr(provider_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("ART_MEGATRON_RECOMPUTE_GRANULARITY", "disabled")
    monkeypatch.setenv("ART_MEGATRON_RECOMPUTE_METHOD", "disabled")
    monkeypatch.setenv("ART_MEGATRON_RECOMPUTE_NUM_LAYERS", "disabled")
    monkeypatch.setenv("ART_MEGATRON_RECOMPUTE_MODULES", "disabled")

    resolved = provider_module.get_provider("Qwen/Qwen3-30B-A3B-Instruct-2507")

    assert resolved.recompute_granularity is None
    assert resolved.recompute_method is None
    assert resolved.recompute_num_layers is None
    assert resolved.recompute_modules is None


def test_get_provider_bundle_honors_expert_parallel_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FakeProvider()
    fake_bridge = _FakeBridge(
        model_bridge=object(),
        provider=provider,
    )
    monkeypatch.setattr(
        provider_module.AutoBridge,
        "from_hf_pretrained",
        lambda *args, **kwargs: fake_bridge,
    )
    monkeypatch.setattr(provider_module.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE", "2")
    monkeypatch.setenv("ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE", "1")
    monkeypatch.setenv("ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE", "2")

    resolved = provider_module.get_provider("Qwen/Qwen3-30B-A3B-Instruct-2507")

    assert resolved.tensor_model_parallel_size == 2
    assert resolved.expert_model_parallel_size == 1
    assert resolved.expert_tensor_parallel_size == 2
    assert resolved.sequence_parallel is True
