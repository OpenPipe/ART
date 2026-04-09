from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _ensure_package(name: str) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
    return module


def _stub_module(name: str, **attrs: object) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        sys.modules[name] = module
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_provider_module() -> ModuleType:
    module_name = "art_megatron_provider_under_test"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    for package_name in [
        "art",
        "art.megatron",
        "megatron",
        "megatron.bridge",
        "megatron.bridge.models",
        "megatron.bridge.models.hf_pretrained",
        "megatron.bridge.models.qwen",
        "megatron.bridge.training",
        "megatron.core",
        "megatron.core.transformer",
    ]:
        _ensure_package(package_name)

    class _StateSource:
        pass

    class _ModuleSpec:
        pass

    class _GPTModelProvider:
        pass

    class _AutoBridge:
        @staticmethod
        def from_hf_pretrained(*args: object, **kwargs: object) -> object:
            raise NotImplementedError

    _stub_module("art.megatron.flex_attention", FlexDotProductAttention=object)
    _stub_module("megatron.bridge", AutoBridge=_AutoBridge)
    _stub_module(
        "megatron.bridge.models.gpt_provider",
        GPTModelProvider=_GPTModelProvider,
    )
    _stub_module(
        "megatron.bridge.models.hf_pretrained.state",
        SafeTensorsStateSource=object,
        StateDict=object,
        StateSource=_StateSource,
    )
    _stub_module(
        "megatron.bridge.models.qwen.qwen3_moe_bridge",
        Qwen3MoEBridge=object,
    )
    _stub_module(
        "megatron.bridge.training.flex_dispatcher_backend",
        apply_flex_dispatcher_backend=lambda *args, **kwargs: None,
    )
    _stub_module(
        "megatron.core.transformer.enums",
        AttnBackend=SimpleNamespace(auto="auto"),
    )
    _stub_module("megatron.core.transformer.spec_utils", ModuleSpec=_ModuleSpec)

    provider_path = Path(__file__).resolve().parents[2] / "src/art/megatron/provider.py"
    spec = spec_from_file_location(module_name, provider_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


megatron_provider = _load_provider_module()


def _fake_provider() -> SimpleNamespace:
    return SimpleNamespace(
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
        ep_overlap_early_attn_memory_release=False,
        moe_deepep_num_sms=None,
        moe_apply_probs_on_input=False,
        bias_activation_fusion=False,
        fine_grained_activation_offloading=False,
        offload_modules=[],
        tensor_model_parallel_size=1,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        recompute_modules=None,
        moe_shared_expert_overlap=True,
    )


def test_apply_runtime_env_overrides_warns_on_shared_expert_overlap_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _fake_provider()
    monkeypatch.setenv("ART_MEGATRON_OVERLAP_MOE_EXPERT_PARALLEL_COMM", "true")
    monkeypatch.setenv("ART_MEGATRON_MOE_SHARED_EXPERT_OVERLAP", "true")
    monkeypatch.setattr(
        megatron_provider, "_resolve_default_deepep_num_sms", lambda _: 20
    )

    with pytest.warns(UserWarning, match="moe_shared_expert_overlap=False"):
        megatron_provider._apply_runtime_env_overrides(provider)

    assert provider.moe_shared_expert_overlap is False


@pytest.mark.parametrize("raw_value", [None, "default"])
def test_apply_runtime_env_overrides_uses_resolved_default_num_sms(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str | None,
) -> None:
    provider = _fake_provider()
    monkeypatch.setattr(
        megatron_provider, "_resolve_default_deepep_num_sms", lambda _: 42
    )
    if raw_value is None:
        monkeypatch.delenv("ART_MEGATRON_MOE_DEEPEP_NUM_SMS", raising=False)
    else:
        monkeypatch.setenv("ART_MEGATRON_MOE_DEEPEP_NUM_SMS", raw_value)

    megatron_provider._apply_runtime_env_overrides(provider)

    assert provider.moe_deepep_num_sms == 42


@pytest.mark.parametrize("raw_value", ["none", "3", "0", "-2"])
def test_env_default_or_even_positive_int_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    monkeypatch.setenv("ART_MEGATRON_MOE_DEEPEP_NUM_SMS", raw_value)

    with pytest.raises(ValueError, match="positive, even integer"):
        megatron_provider._env_default_or_even_positive_int(
            "ART_MEGATRON_MOE_DEEPEP_NUM_SMS"
        )
