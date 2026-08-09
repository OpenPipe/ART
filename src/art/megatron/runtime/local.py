from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import threading

from art import dev
from art.distributed.specs import (
    CUDA_DEVICE_UUID_PATTERN,
    ClusterSpec,
    EndpointSpec,
    GpuPlacement,
    HostSpec,
    ModelServiceMemberSpec,
    ModelServiceSpec,
    RuntimeTopology,
    TrainerMeshSpec,
    VllmParallelSpec,
)

from ..runtime_config import get_megatron_runtime_config

LocalServicePorts = tuple[int, int]


def _bind_loopback_port(port: int = 0) -> socket.socket:
    reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        reservation.bind(("127.0.0.1", port))
    except BaseException:
        reservation.close()
        raise
    return reservation


class LocalEndpointAllocator:
    """Owns unique API and rendezvous ports for backend-local runtimes."""

    _lock = threading.Lock()
    _reserved: set[int] = set()

    def __init__(self) -> None:
        self._owned: set[int] = set()

    def reserve(self) -> LocalServicePorts:
        with self._lock:
            sockets: list[socket.socket] = []
            try:
                while len(sockets) < 2:
                    reservation = _bind_loopback_port()
                    if reservation.getsockname()[1] in self._reserved:
                        reservation.close()
                        continue
                    sockets.append(reservation)
                ports = tuple(reservation.getsockname()[1] for reservation in sockets)
                assert len(ports) == 2
                self._reserved.update(ports)
                self._owned.update(ports)
                return ports
            finally:
                for reservation in sockets:
                    reservation.close()

    def replace_api_port(
        self, ports: LocalServicePorts, api_port: int
    ) -> LocalServicePorts:
        with self._lock:
            if ports[0] == api_port:
                return ports
            if not 1 <= api_port <= 65535:
                raise ValueError("OpenAI server port must be between 1 and 65535")
            if not set(ports) <= self._owned:
                raise RuntimeError("local service endpoint ownership was lost")
            if api_port in self._reserved:
                raise ValueError(f"local service port {api_port} is already reserved")
            api = _bind_loopback_port(api_port)
            try:
                configured = (api_port, ports[1])
                self._reserved.difference_update(ports)
                self._reserved.update(configured)
                self._owned.difference_update(ports)
                self._owned.update(configured)
                return configured
            finally:
                api.close()

    def release(self, ports: LocalServicePorts) -> None:
        with self._lock:
            if not set(ports) <= self._owned:
                raise RuntimeError("local service endpoint ownership was lost")
            self._reserved.difference_update(ports)
            self._owned.difference_update(ports)


def _host_gpu_ids(
    gpu_ids: tuple[int, ...], *, visible_gpu_count: int
) -> tuple[int | str, ...]:
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visible is None:
        return gpu_ids
    visible = tuple(part.strip() for part in raw_visible.split(",") if part.strip())
    if len(visible) != visible_gpu_count or len(
        {value.casefold() for value in visible}
    ) != len(visible):
        raise RuntimeError(
            "local Monarch requires unique CUDA_VISIBLE_DEVICES matching the "
            f"visible CUDA count, got {raw_visible!r} for {visible_gpu_count} GPUs"
        )
    if any(
        not (value.isdecimal() or re.fullmatch(CUDA_DEVICE_UUID_PATTERN, value))
        for value in visible
    ):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain only numeric, full GPU UUID, or MIG "
            "tokens"
        )
    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= len(visible)]
    if invalid:
        raise ValueError(
            f"GPU ids {invalid} exceed the controller's visible CUDA devices"
        )
    return tuple(
        int(visible[gpu_id]) if visible[gpu_id].isdecimal() else visible[gpu_id]
        for gpu_id in gpu_ids
    )


def with_local_serving_port(
    topology: RuntimeTopology,
    *,
    model_name: str,
    port: int,
    rendezvous_port: int | None = None,
) -> RuntimeTopology:
    services = tuple(
        service for service in topology.model_services if service.name == model_name
    )
    if len(services) != 1:
        raise ValueError(f"runtime topology has no unique service {model_name!r}")
    service = services[0]
    endpoint = EndpointSpec(host=service.leader_endpoint.host, port=port)
    if endpoint == service.leader_endpoint and rendezvous_port is None:
        return topology
    if (
        len(topology.cluster.hosts) != 1
        or len(service.members) != 1
        or not service.leader_endpoint.is_loopback
    ):
        raise ValueError("OpenAI server port conflicts with the compiled topology")
    rendezvous = service.rendezvous
    if rendezvous_port is not None:
        rendezvous = EndpointSpec(host=rendezvous.host, port=rendezvous_port)
    elif endpoint.port == rendezvous.port:
        reservation = _bind_loopback_port()
        try:
            rendezvous = EndpointSpec(
                host=rendezvous.host, port=reservation.getsockname()[1]
            )
        finally:
            reservation.close()
    configured = service.model_copy(
        update={"leader_endpoint": endpoint, "rendezvous": rendezvous}
    )
    return RuntimeTopology(
        cluster=topology.cluster,
        rollout_host_ids=topology.rollout_host_ids,
        trainer=topology.trainer,
        model_services=tuple(
            configured if value is service else value
            for value in topology.model_services
        ),
    )


def compile_local_runtime_topology(
    config: dev.BackendModelConfig,
    *,
    model_name: str,
    base_model: str,
    artifact_root: str,
    visible_gpu_count: int,
    service_ports: LocalServicePorts | None = None,
) -> RuntimeTopology:
    if visible_gpu_count < 1:
        raise RuntimeError("MegatronBackend requires at least one visible CUDA GPU")
    trainer_gpu_ids = _host_gpu_ids(
        tuple(map(int, config.get("trainer_gpu_ids", range(visible_gpu_count)))),
        visible_gpu_count=visible_gpu_count,
    )
    if not trainer_gpu_ids:
        raise ValueError("Megatron trainer GPU placement must not be empty")
    from art.dev.validate import is_dedicated_mode, is_external_vllm_mode

    engine = config.get("engine_args", {})
    parallel = VllmParallelSpec(
        tp=int(engine.get("tensor_parallel_size", 1)),
        pp=int(engine.get("pipeline_parallel_size", 1)),
        dp=int(engine.get("data_parallel_size", 1)),
        enable_expert_parallel=bool(engine.get("enable_expert_parallel", False)),
    )
    dedicated = is_dedicated_mode(config)
    external = is_external_vllm_mode(config)
    inference_gpu_ids = ()
    if not external:
        inference_gpu_ids = _host_gpu_ids(
            tuple(map(int, config.get("inference_gpu_ids", ()))),
            visible_gpu_count=visible_gpu_count,
        )
        candidates = inference_gpu_ids if dedicated else trainer_gpu_ids
        if len(candidates) < parallel.world_size:
            raise ValueError("vLLM parallelism exceeds local inference GPU placement")
        inference_gpu_ids = candidates[: parallel.world_size]
    available_gpu_ids = tuple(dict.fromkeys((*trainer_gpu_ids, *inference_gpu_ids)))
    host_id = "local"
    init_args = config.get("init_args", {})
    provider_model = str(init_args.get("model_name", base_model))
    configured_revision = init_args.get("revision")
    revision = str(configured_revision) if configured_revision is not None else None
    model_services = ()
    if not external:
        if service_ports is None:
            reservations = (_bind_loopback_port(), _bind_loopback_port())
            try:
                service_ports = tuple(
                    reservation.getsockname()[1] for reservation in reservations
                )
            finally:
                for reservation in reservations:
                    reservation.close()
        api_port, rendezvous_port = service_ports
        if api_port == rendezvous_port:
            raise ValueError("local API and rendezvous ports must differ")
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "base_model": provider_model,
                    "parallel": parallel.model_dump(mode="json"),
                    "revision": revision or "<default>",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        model_services = (
            ModelServiceSpec(
                name=model_name,
                members=(
                    ModelServiceMemberSpec(
                        member_id=host_id,
                        host_id=host_id,
                        node_rank=0,
                        gpu_ids=inference_gpu_ids,
                    ),
                ),
                leader_endpoint=EndpointSpec(host="127.0.0.1", port=api_port),
                rendezvous=EndpointSpec(host="127.0.0.1", port=rendezvous_port),
                base_model=provider_model,
                model_revision=revision,
                runtime_fingerprint=fingerprint,
                parallel=parallel,
                update_mode=config.get("rollout_weights_mode", "lora"),
                temporal_gpu_sharing=not dedicated,
            ),
        )
    return RuntimeTopology(
        cluster=ClusterSpec(
            hosts=(
                HostSpec(
                    host_id=host_id,
                    node_rank=0,
                    worker_address="tcp://127.0.0.1:0",
                    cpu_slots=max(1, os.cpu_count() or 1),
                    gpu_ids=available_gpu_ids,
                ),
            ),
            controller_host_id=host_id,
            artifact_root=artifact_root,
        ),
        rollout_host_ids=(),
        trainer=TrainerMeshSpec(
            ranks=tuple(
                GpuPlacement(host_id=host_id, gpu_id=gpu_id)
                for gpu_id in trainer_gpu_ids
            ),
            topology=get_megatron_runtime_config().topology,
        ),
        model_services=model_services,
    )
