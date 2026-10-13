import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


def _install_stub(monkeypatch: pytest.MonkeyPatch, name: str, module: types.ModuleType):
    monkeypatch.setitem(sys.modules, name, module)


def _load_tinker_service_module(monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[2]
    service_path = repo_root / "src" / "art" / "tinker" / "service.py"

    art_pkg = types.ModuleType("art")
    art_pkg.__path__ = [str(repo_root / "src" / "art")]  # type: ignore[attr-defined]
    _install_stub(monkeypatch, "art", art_pkg)

    tinker_pkg = types.ModuleType("art.tinker")
    tinker_pkg.__path__ = [str(repo_root / "src" / "art" / "tinker")]  # type: ignore[attr-defined]
    _install_stub(monkeypatch, "art.tinker", tinker_pkg)

    preprocessing_pkg = types.ModuleType("art.preprocessing")
    preprocessing_pkg.__path__ = [str(repo_root / "src" / "art" / "preprocessing")]  # type: ignore[attr-defined]
    _install_stub(monkeypatch, "art.preprocessing", preprocessing_pkg)

    dev_mod = types.ModuleType("art.dev")
    dev_mod.InternalModelConfig = dict
    dev_mod.OpenAIServerConfig = dict
    dev_mod.TrainConfig = dict
    _install_stub(monkeypatch, "art.dev", dev_mod)

    types_mod = types.ModuleType("art.types")
    types_mod.TrainConfig = dict
    _install_stub(monkeypatch, "art.types", types_mod)

    loss_mod = types.ModuleType("art.loss")
    loss_mod.loss_fn = lambda *args, **kwargs: None
    loss_mod.shift_tensor = lambda tensor, _: tensor
    _install_stub(monkeypatch, "art.loss", loss_mod)

    inputs_mod = types.ModuleType("art.preprocessing.inputs")
    inputs_mod.TrainInputs = dict
    inputs_mod.create_train_inputs = lambda *args, **kwargs: {}
    _install_stub(monkeypatch, "art.preprocessing.inputs", inputs_mod)

    pack_mod = types.ModuleType("art.preprocessing.pack")
    pack_mod.DiskPackedTensors = dict
    pack_mod.packed_tensors_from_dir = lambda **kwargs: kwargs
    _install_stub(monkeypatch, "art.preprocessing.pack", pack_mod)

    server_mod = types.ModuleType("art.tinker.server")
    server_mod.OpenAICompatibleTinkerServer = type(
        "OpenAICompatibleTinkerServer", (), {}
    )
    _install_stub(monkeypatch, "art.tinker.server", server_mod)

    yaml_mod = types.ModuleType("yaml")

    def safe_load(stream_or_text):
        if hasattr(stream_or_text, "read"):
            return json.loads(stream_or_text.read())
        return json.loads(stream_or_text)

    def safe_dump(data, stream=None):
        text = json.dumps(data)
        if stream is None:
            return text
        stream.write(text)
        return None

    yaml_mod.safe_load = safe_load
    yaml_mod.safe_dump = safe_dump
    _install_stub(monkeypatch, "yaml", yaml_mod)

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = type("Tensor", (), {})
    torch_mod.float32 = "float32"
    _install_stub(monkeypatch, "torch", torch_mod)

    tinker_mod = types.ModuleType("tinker")
    tinker_mod.ServiceClient = type("ServiceClient", (), {})
    tinker_mod.TrainingClient = type("TrainingClient", (), {})
    tinker_mod.Datum = type("Datum", (), {})
    tinker_mod.TensorData = type(
        "TensorData", (), {"from_torch": staticmethod(lambda value: value)}
    )
    tinker_mod.ModelInput = type(
        "ModelInput", (), {"from_ints": staticmethod(lambda ints: ints)}
    )
    tinker_mod.AdamParams = type("AdamParams", (), {})
    _install_stub(monkeypatch, "tinker", tinker_mod)

    tinker_lib_pkg = types.ModuleType("tinker.lib")
    tinker_lib_pkg.__path__ = []  # type: ignore[attr-defined]
    _install_stub(monkeypatch, "tinker.lib", tinker_lib_pkg)

    public_interfaces_pkg = types.ModuleType("tinker.lib.public_interfaces")
    public_interfaces_pkg.__path__ = []  # type: ignore[attr-defined]
    _install_stub(monkeypatch, "tinker.lib.public_interfaces", public_interfaces_pkg)

    rest_client_mod = types.ModuleType("tinker.lib.public_interfaces.rest_client")
    rest_client_mod.RestClient = type("RestClient", (), {})
    _install_stub(monkeypatch, "tinker.lib.public_interfaces.rest_client", rest_client_mod)

    spec = importlib.util.spec_from_file_location("art.tinker.service", service_path)
    assert spec is not None and spec.loader is not None
    service_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "art.tinker.service", service_module)
    spec.loader.exec_module(service_module)
    return service_module


def test_get_state_reuses_nested_user_metadata_from_training_client_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service_module = _load_tinker_service_module(monkeypatch)

    checkpoint_dir = tmp_path / "checkpoints" / "0001"
    checkpoint_dir.mkdir(parents=True)
    info_path = checkpoint_dir / "info.yaml"
    info_path.write_text(
        json.dumps(
            {
                "state_with_optimizer_path": "tinker://state/0001",
                "sampler_weights_path": "tinker://sampler/0001",
            }
        )
    )

    observed: dict[str, object] = {}
    fake_training_client = object()

    class FakeServiceClient:
        def create_rest_client(self) -> object:
            return object()

        async def create_training_client_from_state_with_optimizer_async(
            self,
            *,
            path: str,
            user_metadata: dict[str, str] | None = None,
        ) -> object:
            observed["path"] = path
            observed["user_metadata"] = user_metadata
            return fake_training_client

    monkeypatch.setattr(
        service_module.tinker,
        "ServiceClient",
        FakeServiceClient,
    )

    service = service_module.TinkerService(
        model_name="test-model",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        config={
            "tinker_args": {
                "renderer_name": "qwen3_5",
                "training_client_args": {
                    "user_metadata": {"tenant": "test-tenant"},
                },
            }
        },
        output_dir=str(tmp_path),
    )

    state = asyncio.run(service._get_state())

    assert observed["path"] == "tinker://state/0001"
    assert observed["user_metadata"] == {"tenant": "test-tenant"}
    assert state.training_client is fake_training_client
    assert state.models["test-model"] == "tinker://sampler/0001"
