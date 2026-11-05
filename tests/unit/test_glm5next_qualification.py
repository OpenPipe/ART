"""Launch readiness rejects changed inputs and unsafe placement before allocation."""

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from tests.integration.megatron.glm5next_qualification_launch import (
    KubeClient,
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


def test_pss_reads_rollup_and_timeout_cleans_worker(tmp_path, monkeypatch):
    original = Path.read_text
    observed = []

    def read(path, *args, **kwargs):
        if path.name == "smaps_rollup":
            observed.append(path)
            return "Rss: 10000000 kB\nPss: 3000 kB\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    value = process_tree_pss(os.getpid(), {})
    assert observed and value == len(observed) * 3000 * 1024
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
        "hybridep-setup",
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


def test_b200_manifest_uses_primary_pool_and_cuda13():
    seal = {
        "namespace": "qual-flash",
        "chunks": ["source"],
        "source_sha256": "0" * 64,
        "pool": "caladan-b200-spot",
        "pool_selector": "compute.coreweave.com/node-pool",
        "priorityClassName": "yield",
        "pod_cpu": 96,
        "image": "ghcr.io/openpipe/caladan-stark-trainer-cuda13@sha256:" + "0" * 64,
    }
    pod = manifest(seal, "gpu", True)
    assert pod["spec"]["priorityClassName"] == "yield"
    assert pod["spec"]["nodeSelector"] == {
        "compute.coreweave.com/node-pool": "caladan-b200-spot"
    }
    assert pod["spec"]["containers"][0]["image"] == seal["image"]
    assert "preemptionPolicy" not in pod["spec"]
    assert manifest(seal, "cpu-stage0", False)["spec"]["activeDeadlineSeconds"] >= 1800


def test_cluster_cleanup_only_deletes_owned_resources(tmp_path):
    class Client(KubeClient):
        def __init__(self):
            super().__init__(
                {"window_seconds": 14400, "namespace": "qual-flash"}, tmp_path
            )
            self.calls = []

        def call(self, args, data=None, seconds=120):
            self.calls.append(args)
            return ""

    client = Client()
    client.create({"kind": "ConfigMap", "metadata": {"name": "m4-source"}})
    client.create({"kind": "Pod", "metadata": {"name": "m4-worker"}})
    client.cleanup()
    deletes = [args[:3] for args in client.calls if args[0] == "delete"]
    assert deletes == [
        ["delete", "pod", "m4-worker"],
        ["delete", "configmap", "m4-source"],
    ]
    assert all("namespace" not in args for args in client.calls)


def test_literal_grant_and_atomic_single_controller(tmp_path):
    from tests.integration.megatron.glm5next_qualification_launch import (
        claim_controller,
        has_grant,
    )

    approved = "f" * 64
    seal = {"kind": "cluster", "controller_owner": "named-continuation"}
    grant = f"GRANT glm-m4-model B200 seal_sha256={approved} — Jarvis: exact terms"
    assert has_grant(seal, approved, grant)
    for invalid in (
        grant.replace("GRANT", "AMEND"),
        "NEEDS-LEAD " + grant,
        grant.replace("B200", "workstation"),
        grant.replace(approved, "0" * 64),
    ):
        assert not has_grant(seal, approved, invalid)
    with pytest.raises(ValueError, match="sealed controller owner"):
        claim_controller(seal, approved, "other-continuation", tmp_path, tmp_path)
    assert not list(tmp_path.iterdir())
    claim_controller(seal, approved, "named-continuation", tmp_path, tmp_path)
    with pytest.raises(FileExistsError):
        claim_controller(seal, approved, "named-continuation", tmp_path, tmp_path)


def test_old_ep8_cp1_is_refused_by_real_trainer_on_cpu():
    from tests.integration.megatron.glm5next_topology import validate_plan

    with pytest.raises(
        NotImplementedError, match="combining data parallelism with expert"
    ):
        validate_plan(
            dict(name="old-EP8-CP1", task="forward", cp=1, ep=8, world_size=8),
            [1, 2, 3],
            backward=False,
        )


@pytest.mark.parametrize("ep", [1, 8])
@pytest.mark.parametrize("backward", [False, True])
def test_cp8_real_trainer_plans_pass_on_cpu(ep, backward):
    from tests.integration.megatron.glm5next_topology import validate_plan

    row = validate_plan(
        dict(name="CP8", task="learn", cp=8, ep=ep, world_size=8),
        [154820, 2, 3],
        backward=backward,
    )
    assert row["dp"] == 1 and row["packed_tokens"] == 64
    assert row["topology"] == (1, 1, 8, 1)
    assert not row["cuda_initialized"]


@pytest.mark.parametrize("mean,hard_bug", [(0.05, False), (0.11, True)])
def test_first_forward_repeat_preserves_evidence_and_stops(tmp_path, mean, hard_bug):
    import torch

    from tests.integration.megatron.glm5next_qualification import first_forward_repeat

    class Engine:
        rank = 0
        calls = 0

        def forward(self, ids):
            self.calls += 1
            assert ids == [1, 2, 3]
            return {
                "logprobs": torch.tensor([0.0, 0.0]) + (mean if self.calls == 2 else 0)
            }

    engine = Engine()
    if hard_bug:
        with pytest.raises(RuntimeError, match="repeat hard bug"):
            first_forward_repeat(engine, [1, 2, 3], tmp_path)
    else:
        first_forward_repeat(engine, [1, 2, 3], tmp_path)
    assert engine.calls == 2
    report = json.loads((tmp_path / "first-forward-repeat-rank0.json").read_text())
    assert report["hard_bug"] == hard_bug
    assert report["mean"] == pytest.approx(mean)
    assert (tmp_path / "first-forward-repeat-rank0.safetensors").exists()


def test_b200_blocks_before_context_and_has_no_dp8_stage():
    from tests.integration.megatron.glm5next_qualification_launch import steps

    commands = steps(
        {"kind": "cluster", "fixtures": "/fixtures"},
        Path("/source"),
        Path("/work"),
        tiny=False,
    )
    names = [step["name"] for step in commands]
    assert "EP8-CP1" not in names
    assert (
        names.index("CP8-EP8")
        < names.index("rank32")
        < names.index("kda-cp8")
        < names.index("context-32768")
    )
    for step in commands:
        if (plan := step.get("trainer_plan")) is not None:
            assert (plan["world_size"], plan["cp"], plan["ep"]) == (8, 8, 8)


@pytest.mark.parametrize("oom", [False, True])
def test_context_stops_at_first_oom_and_other_errors_block(tmp_path, monkeypatch, oom):
    import tests.integration.megatron.glm5next_qualification_launch as launch

    commands = [
        {"name": f"context-{length}", "argv": [str(length)], "seconds": 10, "env": {}}
        for length in (32768, 65536, 131072, 262144)
    ]
    monkeypatch.setattr(launch, "steps", lambda *_args, **_kwargs: commands)
    calls = []

    def run(argv, *_args, **_kwargs):
        calls.append(int(argv[0]))
        if argv == ["65536"]:
            if oom:
                launch.write(
                    tmp_path / "results/context-65536/failure-rank0.json",
                    {"status": "oom"},
                )
            raise RuntimeError("failed")
        return {}

    monkeypatch.setattr(launch, "run", run)
    seal = {"kind": "cluster", "window_seconds": 14400, "pss_limit_bytes": 120 << 30}
    if oom:
        result = launch.pipeline(seal, tmp_path, tmp_path, tiny=False)
        assert result["context"]["largest_fit"] == 32768
        assert result["context"]["first_oom"] == 65536
        assert [s["status"] for s in result["steps"]] == [
            "pass",
            "oom",
            "not_run",
            "not_run",
        ]
    else:
        with pytest.raises(RuntimeError, match="failed"):
            launch.pipeline(seal, tmp_path, tmp_path, tiny=False)
    assert calls == [32768, 65536]


def test_context_recognizes_wrapped_cuda_oom_but_not_admission_refusal():
    import torch

    from art.trainer_rank import TrainerRankMemoryError
    from tests.integration.megatron.glm5next_qualification import oom_evidence

    raw = torch.cuda.OutOfMemoryError("allocation failed")
    wrapped = TrainerRankMemoryError("CUDA OOM occurred despite planner estimate")
    wrapped.__cause__ = raw
    record = oom_evidence(wrapped, "context-65536")
    assert record is not None and record["status"] == "oom"
    assert record["chain"][1]["type"] == "OutOfMemoryError"
    assert oom_evidence(raw, "context-65536") is not None
    assert (
        oom_evidence(
            TrainerRankMemoryError("predicted to exceed available memory"),
            "context-65536",
        )
        is None
    )


def test_kda_mvp_numeric_gradient_failure_does_not_block():
    from tests.integration.megatron.glm5next_kda_cp import gate_policy

    row = {
        "packing": "single-root",
        "output": {"rel_l2": 1e-4},
        "gradients": {"A": {"rel_l2": 0.003, "max_abs": 0.01}},
    }
    gate_policy(row, 8)
    assert row["blocking_pass"] and not row["report_only_pass"]
    assert row["gradients"]["A"]["rel_l2"] == 0.003


@pytest.mark.parametrize("forward_error,blocking", [(1e-4, True), (0.003, False)])
def test_dsa_numeric_gradient_failure_retains_forward_gate(forward_error, blocking):
    from tests.integration.megatron.glm5next_dsa_cp import apply_gate_policy

    row = {
        "passed": False,
        "selection_matching_fraction": 1.0,
        "output": {"cosine": 1.0, "rel_l2": 1e-4},
        "lora_gradients": {"A": {"rel_l2": 0.03}},
        "cp1": {"output": {"rel_l2": forward_error}, "A": {"rel_l2": 0.003}},
    }
    apply_gate_policy(row)
    assert row["passed"] is blocking
    assert row["blocking_pass"] is blocking
    assert not row["report_only_pass"] and not row["original_all_tolerances_passed"]
    assert row["lora_gradients"]["A"]["rel_l2"] == 0.03
