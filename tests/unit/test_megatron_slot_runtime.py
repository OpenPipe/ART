from types import SimpleNamespace
from typing import Any

import pytest

import art
from art.distributed import (
    ClusterSpec,
    GpuPlacement,
    HostSpec,
    RuntimeTopology,
    TrainerMeshSpec,
)
from art.megatron import MegatronSlotLaunchConfig, launch_megatron_slot


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
            return SimpleNamespace(runtime_spec=spec)

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
    assert launched.descriptor.model_dump() == {
        "runtime_source_id": "slot",
        "runtime_source_epoch": 7,
        "runtime_fingerprint": "a" * 64,
    }

    await launched.aclose()
    assert runtime.closed
    assert launched.coordinator._closed
