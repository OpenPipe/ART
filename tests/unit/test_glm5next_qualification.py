"""Launch readiness rejects changed inputs and unsafe placement before allocation."""

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from tests.integration.megatron.glm5next_qualification_launch import (
    manifest,
    preflight,
    process_tree_pss,
    run,
)


def test_seal_binding_and_holds(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"sealed code")
    seal = tmp_path / "seal.json"
    value = {
        "holds": [],
        "launch_ready": True,
        "files": [
            {
                "path": str(source),
                "size": 11,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
    }
    seal.write_text(json.dumps(value))
    approved = hashlib.sha256(seal.read_bytes()).hexdigest()
    assert preflight(seal, approved) == value
    source.write_bytes(b"changed src")
    with pytest.raises(ValueError, match="changed"):
        preflight(seal, approved)
    value["holds"] = ["grant"]
    seal.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="launch ready"):
        preflight(seal, hashlib.sha256(seal.read_bytes()).hexdigest())


@pytest.mark.parametrize("gpu", [False, True])
def test_cluster_manifest_uses_amended_b300_placement(gpu):
    pod = manifest(
        {"namespace": "glm-m4-b300", "chunks": ["source"], "source_sha256": "0" * 64},
        "m4-test",
        gpu,
    )
    spec = pod["spec"]
    assert "preemptionPolicy" not in spec
    assert spec["priorityClassName"] == "caladan-serving"
    assert spec["nodeSelector"] == {"caladan.openpipe.ai/pool": "caladan-b300-spot"}
    assert spec["tolerations"] == [
        {
            "key": "caladan.openpipe.ai/spot",
            "operator": "Exists",
            "effect": "NoSchedule",
        }
    ]
    resources = spec["containers"][0]["resources"]
    assert resources["requests"] == resources["limits"]
    assert resources["requests"].get("nvidia.com/gpu", "0") == ("8" if gpu else "0")
    assert spec["activeDeadlineSeconds"] <= 14400


def test_pss_reads_rollup_and_timeout_cleans_worker(tmp_path):
    value = process_tree_pss(os.getpid(), {})
    rss = int(Path(f"/proc/{os.getpid()}/statm").read_text().split()[1]) * os.sysconf(
        "SC_PAGE_SIZE"
    )
    assert 0 < value <= rss + (1 << 20)
    with pytest.raises(TimeoutError):
        run(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            1,
            tmp_path / "timeout.log",
        )


def test_pss_allows_kernel_confirmed_process_exit(monkeypatch):
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path.name == "smaps_rollup":
            raise ProcessLookupError("process exited after discovery")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert process_tree_pss(os.getpid(), {}) == 0


def test_layer_capture_unwraps_ddp_and_precision_wrappers():
    from types import SimpleNamespace

    import torch

    from tests.integration.megatron.glm5next_qualification import capture_layer_outputs

    layer: Any = torch.nn.Linear(2, 2, bias=False)
    layer.layer_number = 1
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2))
    wrapped = SimpleNamespace(
        module=SimpleNamespace(
            module=SimpleNamespace(decoder=SimpleNamespace(layers=[layer]))
        )
    )
    values = {}
    handles = capture_layer_outputs([wrapped], values)
    x = torch.arange(16, dtype=torch.float32).reshape(8, 2).requires_grad_()
    layer(x)
    sample = values["layers.0.streams"]
    assert sample.shape == (4, 2) and not sample.requires_grad
    torch.testing.assert_close(sample, x[:4])
    for handle in handles:
        handle.remove()


def test_qualification_pins_rank32_and_preserves_cp_ep():
    from types import SimpleNamespace

    from tests.integration.megatron.glm5next_qualification import (
        configure_qualification_provider,
    )

    cfg = SimpleNamespace(
        _art_lora_rank=1,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_tensor_parallel_size=2,
        context_parallel_size=8,
        expert_model_parallel_size=8,
        _art_model_support_spec=SimpleNamespace(default_target_modules=("lm_head",)),
    )
    configure_qualification_provider(cfg)
    assert cfg._art_lora_rank == 32
    assert (
        cfg.tensor_model_parallel_size,
        cfg.pipeline_model_parallel_size,
        cfg.expert_tensor_parallel_size,
    ) == (1, 1, 1)
    assert (cfg.context_parallel_size, cfg.expert_model_parallel_size) == (8, 8)
    assert cfg._art_lora_target_modules == ["lm_head"]


@pytest.mark.parametrize(
    "step",
    [
        "stage0",
        "volumes",
        "acquire",
        "oracle-setup",
        "identity",
        "identity-F4",
        "identity-F8",
    ],
)
def test_cpu_stages_never_receive_gpu_visibility(step):
    from tests.integration.megatron.glm5next_qualification_launch import visible_devices

    seal = {"kind": "workstation", "uuids": ["GPU-two", "GPU-three"]}
    assert visible_devices(seal, step, tiny=False) == ""
    assert visible_devices(seal, "F4", tiny=False) == "GPU-two,GPU-three"
    assert visible_devices(seal, "F4", tiny=True) == ""
