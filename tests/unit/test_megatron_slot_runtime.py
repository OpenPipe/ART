from types import SimpleNamespace
from typing import Any, cast

import pytest
from safetensors.torch import save_file
import torch

import art
from art.distributed import (
    ClusterSpec,
    GpuPlacement,
    HostSpec,
    RuntimeTopology,
    TrainerMeshSpec,
)
from art.distributed.host_admission import GpuIdentity, HostAdmissionReport
from art.megatron import MegatronSlotLaunchConfig, launch_megatron_slot
from art.megatron.model_support.lora_disk import (
    load_adapter_config,
    normalize_lora_checkpoint_to_vllm,
)
from art.runtime_attestation import (
    RuntimeArchitectureAttestation,
    RuntimeHostAttestation,
)


def test_host_attestation_retains_driver_and_gpu_identity() -> None:
    report = cast(
        HostAdmissionReport,
        SimpleNamespace(
            host_id="host",
            hostname="trainer-host",
            boot_id="boot",
            runtime=SimpleNamespace(sha256="a" * 64),
            nvidia_driver_version="580.159.03",
            assigned_gpus=(
                GpuIdentity(
                    index=0,
                    product_name="NVIDIA B300 SXM6 PC",
                    compute_capability="10.3",
                    uuid="GPU-f7f178ce-8dac-92a7-71fc-b5ea257b8ff7",
                    parent_uuid="GPU-f7f178ce-8dac-92a7-71fc-b5ea257b8ff7",
                    pci_bus_id="00000000:1A:00.0",
                ),
            ),
        ),
    )

    attestation = RuntimeHostAttestation.from_admission(report)

    assert attestation.nvidia_driver_version == "580.159.03"
    assert attestation.assigned_gpus == report.assigned_gpus


def test_serving_normalization_retains_exact_training_targets(tmp_path) -> None:
    save_file({"weight": torch.ones(1)}, tmp_path / "adapter_model.safetensors")
    normalize_lora_checkpoint_to_vllm(
        tmp_path,
        handler=SimpleNamespace(
            to_vllm_lora_tensors=lambda tensors, *, adapter_config: (
                tensors,
                {**adapter_config, "target_modules": ["q_proj", "experts"]},
            )
        ),  # ty: ignore[invalid-argument-type]
        adapter_config={
            "r": 1,
            "target_modules": ["q_proj", "gate_proj", "up_proj", "down_proj"],
        },
    )
    config = load_adapter_config(tmp_path)
    assert config["target_modules"] == ["q_proj", "experts"]
    assert config["art_training_target_modules"] == [
        "q_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


@pytest.mark.asyncio
async def test_launch_megatron_slot_owns_one_shared_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import art.megatron.slot_runtime as slot_runtime

    topology_config = art.MegatronTopologyConfig(tp=1, cp=1, ep=1, pp=1)
    topology = RuntimeTopology(
        cluster=ClusterSpec(
            hosts=(
                HostSpec(
                    host_id="host",
                    node_rank=0,
                    worker_address="tcp://127.0.0.1:0",
                    cpu_slots=1,
                    gpu_ids=(0,),
                ),
            ),
            controller_host_id="host",
            artifact_root="/tmp/art-slot",
        ),
        rollout_host_ids=(),
        trainer=TrainerMeshSpec(
            ranks=(GpuPlacement(host_id="host", gpu_id=0),),
            topology=topology_config,
        ),
    )
    runtime_spec = SimpleNamespace(fingerprint="a" * 64)
    architecture = RuntimeArchitectureAttestation.create(
        runtime_kind="trainer",
        base_model="model",
        model_source="model",
        model_revision="default",
        model_support_key="test",
        handler_name="test",
        canonical_config_sha256="b" * 64,
        loaded_layer_count=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        data_parallel_size=1,
        world_size=1,
        runtime_identity="a" * 64,
    )

    class Runtime:
        def __init__(self) -> None:
            self.topology = topology
            self.runtime_id = "runtime"
            self.closeables: list[Any] = []
            self.shared_starts = 0
            self.closed = False

        async def start_shared_trainer(self, spec: Any, **kwargs: Any) -> Any:
            assert spec is runtime_spec
            assert kwargs == {
                "launch_id": "slot",
                "command_timeout_s": 30.0,
                "shutdown_timeout_s": 20.0,
            }
            self.shared_starts += 1
            return SimpleNamespace(
                runtime_spec=spec, architecture_attestation=architecture
            )

        def register_closeable(self, closeable: Any) -> None:
            self.closeables.append(closeable)

        async def close(self) -> None:
            for closeable in self.closeables:
                await closeable.aclose()
            self.closed = True

    runtime = Runtime()

    async def start_local(*_args: Any, **_kwargs: Any) -> Runtime:
        return runtime

    monkeypatch.setattr(
        slot_runtime.ArtRuntime,
        "start_local",
        staticmethod(start_local),
    )
    monkeypatch.setattr(
        slot_runtime, "init_megatron_runtime_config", lambda value: value
    )
    monkeypatch.setattr(
        slot_runtime,
        "build_trainer_runtime_spec",
        lambda *_args, **_kwargs: runtime_spec,
    )

    launched = await launch_megatron_slot(
        MegatronSlotLaunchConfig(
            slot_id="slot",
            runtime_source_epoch=7,
            topology=topology,
            megatron=art.MegatronRuntimeConfig(
                topology=topology_config,
                packed_sequence_length=128,
            ),
            base_model="model",
            command_timeout_s=30.0,
            shutdown_timeout_s=20.0,
        )
    )

    assert runtime.shared_starts == 1
    assert runtime.closeables == [launched.coordinator]
    assert launched.descriptor.runtime_source_id == "slot"
    assert launched.descriptor.runtime_source_epoch == 7
    assert launched.descriptor.runtime_fingerprint == "a" * 64
    assert launched.descriptor.trainer_architecture == architecture
    assert launched.descriptor.paired_attestation is None

    await launched.aclose()
    assert runtime.closed
    assert launched.coordinator._closed
