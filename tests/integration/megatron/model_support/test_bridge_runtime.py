from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
import weakref

import pytest
import torch

pytest.importorskip("megatron.bridge")

from megatron.bridge.models.hf_pretrained.state import StateDict, StateSource

from art.megatron.model_support.spec import HfWeightSource
from art.megatron.runtime.bridge_runtime import (
    _optimized_load_weights_hf_to_megatron,
    load_unique_hf_keys_once,
)


class _Mapping:
    def __init__(self, megatron_param: str, hf_param: str) -> None:
        self.megatron_param = megatron_param
        self.hf_param = hf_param
        self.tp_size = 1

    def hf_to_megatron(
        self, hf_weights: torch.Tensor, megatron_module: torch.nn.Module
    ) -> torch.Tensor:
        del megatron_module
        return hf_weights


class _Bridge:
    def __init__(self, tasks: list[Any]) -> None:
        self.tasks = tasks

    def build_conversion_tasks(
        self, hf_pretrained: Any, megatron_model: Any
    ) -> list[Any]:
        del hf_pretrained, megatron_model
        return self.tasks

    def _share_embeddings_and_output_weights(self, config: Any) -> bool:
        return bool(config.share_embeddings_and_output_weights)

    def _is_adapter_param_name(self, name: str) -> bool:
        return ".adapter." in name

    def _with_progress_tracking(self, tasks: list[Any], description: str) -> list[Any]:
        del description
        return tasks

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str, state: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return state[hf_param]

    def _broadcast_shared_embeddings(self, megatron_model: Any) -> None:
        del megatron_model


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(share_embeddings_and_output_weights=False)
        self.local = torch.nn.Linear(1, 1, bias=False)


def _task(
    mapping: _Mapping,
    *,
    module: torch.nn.Module | None = None,
    weight: torch.Tensor | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        mapping=mapping,
        megatron_module=module,
        param_weight=weight,
        param_name=mapping.megatron_param,
    )


def test_pretrained_load_rejects_placeholder_for_required_local_parameter() -> None:
    model = _Model()
    bridge = _Bridge([_task(_Mapping("local.weight", "hf.weight"))])
    pretrained = SimpleNamespace(state={}, model_name_or_path="empty-checkpoint")

    with pytest.raises(
        RuntimeError,
        match=r"1 required local parameter\(s\): local.weight",
    ):
        _optimized_load_weights_hf_to_megatron(cast(Any, bridge), pretrained, model)


def test_pretrained_load_allows_nonlocal_placeholder_tasks() -> None:
    model = _Model()
    local_mapping = _Mapping("local.weight", "hf.weight")
    remote_mapping = _Mapping("remote.weight", "hf.remote_weight")
    bridge = _Bridge(
        [
            _task(local_mapping, module=model.local, weight=model.local.weight),
            _task(remote_mapping),
        ]
    )
    expected = torch.tensor([[7.0]])
    pretrained = SimpleNamespace(
        state={"hf.weight": expected}, model_name_or_path="checkpoint"
    )

    result = _optimized_load_weights_hf_to_megatron(
        cast(Any, bridge), pretrained, model
    )

    assert result == [model]
    assert torch.equal(model.local.weight, expected)


class _LazySource(StateSource):
    def __init__(self, values: dict[str, int]) -> None:
        self.values = values
        self.reads: list[list[str]] = []
        self.refs: dict[str, weakref.ReferenceType[torch.Tensor]] = {}
        self.live_before_read: list[set[str]] = []

    def get_all_keys(self) -> list[str]:
        return list(self.values)

    def load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
        self.reads.append(list(keys))
        self.live_before_read.append(
            {key for key, ref in self.refs.items() if ref() is not None}
        )
        tensors = {key: torch.tensor([self.values[key]], device="cpu") for key in keys}
        self.refs.update({key: weakref.ref(value) for key, value in tensors.items()})
        return tensors


def _load_direct(keys: list[str], state: Any, bridge: Any = None) -> dict[str, Any]:
    tasks = [_task(_Mapping(key, key), module=cast(Any, object())) for key in keys]
    return load_unique_hf_keys_once(tasks, state, bridge=bridge)


@pytest.mark.parametrize("key", ["ordinary", "literal*", "literal?", "literal[0]"])
@pytest.mark.parametrize("dictionary", [False, True])
def test_direct_load_preserves_literal_physical_keys(
    key: str, dictionary: bool
) -> None:
    source = _LazySource({key: 11, "literal0": 22, "literalX": 33})
    state = (
        source.load_tensors(source.get_all_keys()) if dictionary else StateDict(source)
    )

    cache = _load_direct([key], state)

    assert cache[key].tolist() == [11]
    if not dictionary:
        assert source.reads == [[key]]


@pytest.mark.parametrize("fail_calls", [(), (1,), (1, 3)])
@pytest.mark.parametrize("dictionary", [False, True])
def test_direct_load_releases_sources_after_last_alias(
    monkeypatch: pytest.MonkeyPatch, fail_calls: tuple[int, ...], dictionary: bool
) -> None:
    source = _LazySource({"shared": 11, "other": 22, "tail": 33})
    state = (
        source.load_tensors(source.get_all_keys()) if dictionary else StateDict(source)
    )
    aliases = {"a": "shared", "b": "other", "c": "shared", "d": "tail"}
    bridge = SimpleNamespace(
        _art_hf_weight_source=lambda key, **kwargs: HfWeightSource(
            logical_key=key, physical_key_options=((aliases[key],),)
        )
    )
    pin_calls = []

    def fake_pin(tensor: torch.Tensor) -> torch.Tensor:
        pin_calls.append(tensor.item())
        if len(pin_calls) in fail_calls:
            raise RuntimeError("injected pin failure")
        return tensor.clone()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin)

    cache = _load_direct(list(reversed(aliases)), state, bridge)

    assert list(cache) == list(aliases)
    assert [cache[key].item() for key in aliases] == [11, 22, 11, 33]
    assert pin_calls == [11, 22, 11, 33]
    assert (cache["a"] is cache["c"]) == (fail_calls == (1, 3))
    if not dictionary:
        assert source.reads == [["shared"], ["other"], ["tail"]]
        assert source.live_before_read == [
            set(),
            {"shared"},
            {"shared"} if fail_calls else set(),
        ]
        assert source.refs["other"]() is None and source.refs["tail"]() is None
        assert (source.refs["shared"]() is not None) == bool(fail_calls)


@pytest.mark.parametrize("key", ["missing", "literal*", "literal?", "literal[0]"])
def test_direct_load_rejects_missing_literal_key(key: str) -> None:
    source = _LazySource({"literal0": 11, "literalX": 22})

    with pytest.raises(KeyError, match="HF tensor source"):
        _load_direct([key], StateDict(source))

    assert source.reads == []


def test_direct_load_preserves_missing_key_lookup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _LazySource({"literal*": 11})
    key_censuses = iter([["literal*"], []])
    monkeypatch.setattr(source, "get_all_keys", lambda: next(key_censuses))

    with pytest.raises(KeyError) as caught:
        _load_direct(["literal*"], StateDict(source))

    assert caught.value.args == ("Keys not found: ['literal*']",)
    assert source.reads == []


def test_direct_load_preserves_reader_error(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _LazySource({"first": 11, "second": 22})
    original = source.load_tensors
    error = OSError("injected checkpoint read failure")

    def load(keys: list[str]) -> dict[str, torch.Tensor]:
        if "second" in keys:
            raise error
        return original(keys)

    monkeypatch.setattr(source, "load_tensors", load)
    with pytest.raises(OSError) as caught:
        _load_direct(["first", "second"], StateDict(source))

    assert caught.value is error


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("replace_buffer", [False, True])
def test_external_binding_pre_wrap_mp_and_ddp(
    monkeypatch, tmp_path, replace_buffer, device
):
    """Real MP with CPU gloo DDP, or the full Megatron DDP stack on one GPU."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires one CUDA GPU")
    from megatron.bridge.models import model_provider
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.transformer_config import TransformerConfig
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from art.megatron.runtime.bridge_runtime import _art_get_model

    config = TransformerConfig(
        num_layers=1,
        hidden_size=2,
        num_attention_heads=1,
        bf16=True,
        init_model_with_meta_device=True,
        use_cpu_initialization=True,
    )

    class Model(MegatronModule):
        _keep_fp32_parameters = ("A_log",)
        _keep_fp32_buffers = ("state",)

        def __init__(self):
            super().__init__(config)
            self.weight = torch.nn.Parameter(torch.empty(2, 2, dtype=torch.bfloat16))
            self.A_log = torch.nn.Parameter(torch.empty(2))
            self.D = torch.nn.Parameter(torch.empty(2))
            setattr(self.D, "_keep_fp32", True)
            self.register_buffer("state", torch.empty(2))
            self.register_buffer("cache", torch.empty(2, dtype=torch.bfloat16))

    sources = {
        "0:weight": torch.randn(2, 2, dtype=torch.bfloat16, device=device),
        **{
            f"0:{name}": torch.randn(2, device=device)
            for name in ("A_log", "D", "state")
        },
        "0:cache": torch.randn(2, dtype=torch.bfloat16, device=device),
    }
    setattr(config, "_art_external_weights", sources)
    stages = []

    def create(*args, **kwargs):
        stages.append("meta")
        return [Model()]

    def check_aliases(model):
        for name, source in sources.items():
            actual = getattr(model, name.removeprefix("0:"))
            assert actual.data_ptr() == source.data_ptr()
            assert actual.dtype == source.dtype
            assert torch.equal(actual, source)

    def pre_wrap(chunks):
        stages.append("pre_wrap")
        check_aliases(chunks[0])
        chunks[0].requires_grad_(False)
        chunks[0].register_parameter(
            "adapter",
            torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16, device=device)),
        )

    original_ddp_wrap = model_provider._ddp_wrap

    def ddp_wrap(chunks, *args, **kwargs):
        stages.append("ddp")
        check_aliases(chunks[0].module)
        if replace_buffer:
            chunks[0].module.cache = chunks[0].module.cache.clone()
        if device == "cuda":
            return original_ddp_wrap(chunks, *args, **kwargs)
        return [DistributedDataParallel(chunk) for chunk in chunks]

    monkeypatch.setattr(model_provider, "_create_model", create)
    monkeypatch.setattr(
        model_provider, "_print_num_params", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(model_provider, "_ddp_wrap", ddp_wrap)
    monkeypatch.setattr(model_provider, "correct_amax_history_if_needed", None)
    dist.init_process_group(
        "nccl" if device == "cuda" else "gloo",
        init_method=f"file://{tmp_path / 'rendezvous'}",
        rank=0,
        world_size=1,
    )
    try:
        pg_collection = None
        if device == "cuda":
            parallel_state.initialize_model_parallel()
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        def build():
            return _art_get_model(
                cast(Any, config),
                DistributedDataParallelConfig(),
                init_model_with_meta_device=True,
                use_cpu_initialization=True,
                pre_wrap_hook=pre_wrap,
                pg_collection=cast(Any, pg_collection),
            )

        if replace_buffer:
            with pytest.raises(RuntimeError, match="detached an external base tensor"):
                build()
        else:
            wrapped = build()
            check_aliases(wrapped[0].module.module)
        assert stages == ["meta", "pre_wrap", "ddp"]
    finally:
        if device == "cuda":
            parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize("loader_count", [0, 1, 2])
def test_external_weights_remove_exactly_one_hf_loader(monkeypatch, loader_count):
    from art.megatron import train

    def loader():
        pass

    loader.__name__ = "load_weights_hf_to_megatron"
    other = lambda: None
    provider = SimpleNamespace(_pre_wrap_hooks=[other, *([loader] * loader_count)])
    bundle = SimpleNamespace(provider=provider)
    monkeypatch.setattr(train, "install_fast_frozen_output_backward", lambda: None)
    monkeypatch.setattr(
        train, "prepare_provider_bundle", lambda *args, **kwargs: bundle
    )

    class Validated(Exception):
        pass

    def finish_validation(bundle):
        assert bundle.provider._pre_wrap_hooks == [other]
        assert bundle.provider._art_external_weights == {}
        raise Validated

    monkeypatch.setattr(train, "finalize_provider_bundle", finish_validation)
    expected = Validated if loader_count == 1 else RuntimeError
    with pytest.raises(
        expected, match=None if loader_count == 1 else "exactly one HF loader hook"
    ):
        train.build_training_runtime(
            model_identifier="test/model", external_weights={}, print_env=False
        )
    if loader_count != 1:
        assert not hasattr(provider, "_art_external_weights")
        assert len(provider._pre_wrap_hooks) == loader_count + 1
