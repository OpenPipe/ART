from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Literal, cast
import uuid

import httpx

from art import dev, types
from art.dev.get_model_config import default_target_modules
from art.distributed.art_runtime import ArtRuntime, DistributedPackedBatch
from art.distributed.vllm_replica import (
    ReplicaLaunchTemplate,
    ReplicaUpdateReport,
)
from art.distributed.vllm_router import (
    ReplicaTelemetry,
    RoutableReplica,
    RoutingTable,
)
from art.serving_capabilities import (
    ServingCapabilities,
    discover_serving_capabilities,
)
from art.utils.output_dirs import get_step_checkpoint_dir
from art.vllm_runtime import (
    get_external_vllm_runtime_config,
    map_checkpoint_path_for_vllm,
    normalize_vllm_server_url,
    wait_for_vllm_http_runtime,
)

from .lora import LORA_ALPHA, default_lora_rank_for_handler
from .model_support import get_model_support_handler, model_uses_expert_parallel
from .optimizer_state import (
    format_megatron_resume_message,
    prepare_megatron_resume_state,
)
from .runtime.specs import (
    AdapterReady,
    CurrentTrainConfig,
    DurableTrainOutput,
    ExperimentalTrainConfig,
    TrainAccepted,
    TrainCancelled,
    TrainCompleted,
    TrainerRuntimeSpec,
    TrainFailed,
    TrainingRunSpec,
    TrainJobSpec,
    TrainProgress,
)
from .runtime_config import get_megatron_runtime_config
from .service import create_identity_lora


class DistributedMegatronService:
    """One model's durable checkpoints and run-scoped distributed runtimes."""

    def __init__(
        self,
        *,
        model_name: str,
        base_model: str,
        config: dev.BackendModelConfig,
        output_dir: str,
        runtime: ArtRuntime,
        enable_expert_replay: bool,
    ) -> None:
        self.model_name = model_name
        self.base_model = base_model
        self.config = config
        self.output_dir = output_dir
        self.runtime = runtime
        self.enable_expert_replay = enable_expert_replay
        self._latest_step = 0
        self._training_session_id = uuid.uuid4().hex
        self._trainer: Any = None
        self._trainer_lock = asyncio.Lock()
        self._replica_ids: tuple[str, ...] = ()
        self._base_urls: tuple[str, ...] = ()
        self._gateway: Any = None
        self._gateway_endpoint: tuple[str, int] | None = None
        self._policy_generation = 0
        self._serving_capabilities: ServingCapabilities | None = None
        self._api_key_value: str | None = None
        self._loaded_adapter_steps: set[int] = set()
        self._loaded_exact_adapter_steps: set[int] = set()
        self._exact_adapter_refcounts: dict[int, int] = {}
        self._exact_adapter_lock = asyncio.Lock()
        self._closed = False

    @property
    def rollout_weights_mode(self) -> str:
        return self.config.get("rollout_weights_mode", "lora")

    @property
    def rollout_weight_update_mode(self) -> str:
        return self.config.get("rollout_weight_update_mode", "step_lora")

    @property
    def _allow_unvalidated_arch(self) -> bool:
        return bool(self.config.get("allow_unvalidated_arch", False))

    @property
    def _optimizer_state_path(self) -> str:
        path = f"{self.output_dir}/optimizer_states_rl"
        os.makedirs(path, exist_ok=True)
        return path

    def _lora_config(self) -> dev.LoRAConfig:
        return cast(dev.LoRAConfig, self.config.get("lora_config") or {})

    def _random_state(self) -> int | None:
        for key in ("lora_config", "init_args"):
            value = self.config.get(key, {}).get("random_state")
            if value is not None:
                return int(value)
        return None

    def _resolve_current_lora_path(self) -> str:
        resume = prepare_megatron_resume_state(
            output_dir=self.output_dir,
            optimizer_state_path=self._optimizer_state_path,
        )
        print(format_megatron_resume_message(resume))
        self._latest_step = max(self._latest_step, resume.step)
        path = get_step_checkpoint_dir(self.output_dir, self._latest_step)
        if not (Path(path) / "adapter_model.safetensors").is_file():
            lora = self._lora_config()
            create_identity_lora(
                self.base_model,
                path,
                rank=lora.get("rank"),
                target_modules=lora.get("target_modules"),
                random_state=self._random_state(),
                allow_unvalidated_arch=self._allow_unvalidated_arch,
            )
        return path

    def _runtime_spec(self) -> TrainerRuntimeSpec:
        mesh = self.runtime.topology.trainer
        if mesh is None:
            raise RuntimeError("ART runtime has no trainer mesh")
        runtime_config = get_megatron_runtime_config()
        if runtime_config.topology != mesh.topology:
            raise ValueError(
                "Megatron runtime topology does not match the ART trainer mesh"
            )
        lora = self._lora_config()
        handler = get_model_support_handler(
            self.base_model,
            allow_unvalidated_arch=self._allow_unvalidated_arch,
        )
        targets = lora.get("target_modules") or default_target_modules(self.base_model)
        revision = str(self.config.get("init_args", {}).get("revision") or "default")
        compile_enabled = os.environ.get(
            "ART_DISABLE_MEGATRON_COMPILE", "0"
        ).lower() not in {"1", "true", "yes", "on"}
        identity = {
            "art": _art_source_revision(),
            "model": self.base_model,
            "revision": revision,
            "handler": handler.key,
            "mesh": mesh.model_dump(mode="json"),
        }
        return TrainerRuntimeSpec(
            art_revision=identity["art"],
            model_identifier=self.base_model,
            model_revision=revision,
            handler_name=handler.key,
            lora_rank=int(lora.get("rank") or default_lora_rank_for_handler(handler)),
            lora_alpha=float(lora.get("alpha", LORA_ALPHA)),
            lora_target_modules=tuple(targets),
            dtype=_trainer_dtype(self.config),
            trainer_mesh=mesh,
            packed_sequence_length=runtime_config.packed_sequence_length,
            compile_enabled=compile_enabled,
            compile_fingerprint=_digest({**identity, "compile": compile_enabled}),
            optimizer_layout_fingerprint=_digest(
                {"mesh": mesh.model_dump(mode="json")}
            ),
            allow_unvalidated_arch=self._allow_unvalidated_arch,
            enable_moe_routing_replay=self.enable_expert_replay
            and model_uses_expert_parallel(
                self.base_model,
                allow_unvalidated_arch=self._allow_unvalidated_arch,
            ),
            streaming_weight_offload=runtime_config.streaming_weight_offload,
            random_state=self._random_state(),
        )

    async def _ensure_trainer(self) -> Any:
        if self._trainer is not None:
            return self._trainer
        async with self._trainer_lock:
            if self._trainer is None:
                current = await asyncio.to_thread(self._resolve_current_lora_path)
                runtime_spec = self._runtime_spec()
                run_spec = TrainingRunSpec(
                    run_id=uuid.uuid4().hex,
                    runtime_fingerprint=runtime_spec.fingerprint,
                    training_session_id=self._training_session_id,
                    initial_learner_version=self._latest_step,
                    initial_adapter_path=current,
                    optimizer_state_path=self._optimizer_state_path,
                )
                self._trainer = await self.runtime.start_trainer(runtime_spec, run_spec)
        return self._trainer

    async def train_packed(
        self,
        batch: DistributedPackedBatch,
        config: types.TrainConfig,
        experimental_config: dev.TrainConfig,
    ) -> AsyncIterator[dict[str, float]]:
        trainer = await self._ensure_trainer()
        next_step = self._latest_step + 1
        source = get_step_checkpoint_dir(self.output_dir, self._latest_step)
        staging = f"{self.output_dir}/megatron_runtime/staging/{next_step:04d}"
        await asyncio.to_thread(_copy_checkpoint, source, staging)
        values = {
            key: value
            for key, value in experimental_config.items()
            if key in ExperimentalTrainConfig.model_fields and value is not None
        }
        job = TrainJobSpec(
            job_id=uuid.uuid4().hex,
            run_id=trainer.run_spec.run_id,
            training_session_id=self._training_session_id,
            expected_learner_version=self._latest_step,
            learner_version=next_step,
            batch=batch.leases.ref,
            config=CurrentTrainConfig.model_validate(config.model_dump()),
            experimental_config=ExperimentalTrainConfig.model_validate(values),
            output=DurableTrainOutput(
                adapter_path=staging,
                optimizer_state_path=self._optimizer_state_path,
            ),
        )
        checkpoint: str | None = None
        async for event in trainer.train(job, batch.leases):
            if isinstance(event, (TrainAccepted, TrainCompleted)):
                continue
            if isinstance(event, TrainProgress):
                yield event.metrics
                continue
            if isinstance(event, AdapterReady):
                checkpoint = _publish_checkpoint(staging, self.output_dir, next_step)
                await self.register_lora_for_step(next_step, checkpoint)
                continue
            if isinstance(event, TrainFailed):
                raise RuntimeError(
                    f"distributed Megatron job failed ({event.error_type}): "
                    f"{event.message}"
                )
            if isinstance(event, TrainCancelled):
                raise asyncio.CancelledError(event.reason)
        if checkpoint is None:
            checkpoint = _publish_checkpoint(staging, self.output_dir, next_step)
            await self.register_lora_for_step(next_step, checkpoint)
        self._latest_step = next_step

    async def resolve_global_grad_accumulation_sequences(
        self, config: types.TrainConfig
    ) -> int:
        if config.grad_accumulation_sequences is not None:
            return int(config.grad_accumulation_sequences)
        mesh = self.runtime.topology.trainer
        assert mesh is not None
        topology = mesh.topology
        return len(mesh.ranks) // (topology.tp * topology.cp * topology.pp)

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        if self._base_urls:
            return self._gateway_endpoint or _host_port(self._base_urls[0])
        self._api_key_value = self._api_key(config)
        lora_path = await asyncio.to_thread(self._resolve_current_lora_path)
        external = get_external_vllm_runtime_config(self.config)
        if external is not None:
            base_url = normalize_vllm_server_url(external.server_url)
            headers = _headers(external.api_key)
            await wait_for_vllm_http_runtime(
                base_url=base_url,
                timeout=external.health_timeout_s,
                headers=headers,
            )
            self._base_urls = (base_url,)
            self._serving_capabilities = await discover_serving_capabilities(
                base_url=base_url,
                headers=headers,
                allow_openai_compatible=True,
            )
            await self._load_adapter(lora_path, self._latest_step)
            return _host_port(base_url)

        services = tuple(
            service
            for service in self.runtime.topology.model_services
            if service.name == self.model_name
        )
        if len(services) != 1:
            raise ValueError(
                f"runtime topology must define one model service named "
                f"{self.model_name!r}"
            )
        service = services[0]
        if self.rollout_weights_mode != "lora":
            raise RuntimeError(
                "distributed Megatron currently requires LoRA rollout serving"
            )
        template = ReplicaLaunchTemplate(
            served_model_name=f"{self.model_name}@{self._latest_step}",
            lora_path=lora_path,
            engine_args=self._engine_args(config),
            server_args=self._server_args(config),
        )
        self._replica_ids = tuple(replica.replica_id for replica in service.replicas)
        starts = await asyncio.gather(
            *(
                self.runtime.start_replica(replica, template)
                for replica in service.replicas
            ),
            return_exceptions=True,
        )
        failures = [value for value in starts if isinstance(value, BaseException)]
        if failures:
            await asyncio.gather(
                *(
                    self.runtime.stop_replica(replica_id)
                    for replica_id, value in zip(self._replica_ids, starts, strict=True)
                    if not isinstance(value, BaseException)
                ),
                return_exceptions=True,
            )
            self._replica_ids = ()
            raise BaseExceptionGroup("vLLM replica startup failed", failures)
        self._base_urls = tuple(
            f"http://{replica.leader_endpoint.host}:{replica.leader_endpoint.port}"
            for replica in service.replicas
        )
        capabilities = await asyncio.gather(
            *(
                discover_serving_capabilities(
                    base_url=url,
                    headers=_headers(self._api_key(config)),
                    allow_openai_compatible=False,
                )
                for url in self._base_urls
            )
        )
        if len(set(capabilities)) != 1:
            raise RuntimeError("vLLM replicas expose different ART capabilities")
        self._serving_capabilities = capabilities[0]
        digest = await asyncio.to_thread(_checkpoint_digest, lora_path)
        update_identity = uuid.uuid4().hex
        reports = []
        for replica_id in self._replica_ids:
            manager = self.runtime.replica(replica_id)
            state = manager.prepare_update(update_identity=update_identity)
            report = ReplicaUpdateReport(
                replica_id=replica_id,
                generation=state.generation,
                generation_digest=state.generation_digest,
                policy_version=str(self._latest_step),
                policy_digest=digest,
                update_identity=update_identity,
            )
            if manager.verify_update(report).phase != "ready":
                raise RuntimeError(f"replica {replica_id!r} rejected initial policy")
            reports.append(report)
        self._loaded_adapter_steps.add(self._latest_step)
        if len(self._replica_ids) > 1:
            from art.distributed.vllm_gateway import VllmGateway

            table = self._routing_table(
                service,
                policy_version=self._latest_step,
                policy_digest=digest,
                update_identity=update_identity,
            )
            self._gateway = VllmGateway(
                table,
                upstream_headers=_headers(self._api_key(config)),
                max_queued=self.runtime.config.gateway_max_queued,
                route_timeout_s=self.runtime.config.gateway_route_timeout_s,
            )
            port = await self._gateway.start(self.runtime.config.gateway_bind_host)
            self._gateway_endpoint = (self._gateway_advertise_host(), port)
        return self._gateway_endpoint or _host_port(self._base_urls[0])

    def _routing_table(
        self,
        service: Any,
        *,
        policy_version: int,
        policy_digest: str,
        update_identity: str,
    ) -> RoutingTable:
        now = asyncio.get_running_loop().time()
        specs = {replica.replica_id: replica for replica in service.replicas}
        replicas = []
        for replica_id in self._replica_ids:
            manager = self.runtime.replica(replica_id)
            state = manager.state
            spec = specs[replica_id]
            if state.phase != "ready":
                raise RuntimeError(
                    f"replica {replica_id!r} is not routable: {state.phase}"
                )
            replicas.append(
                RoutableReplica(
                    replica_id=replica_id,
                    endpoint=spec.leader_endpoint,
                    phase="ready",
                    generation=state.generation,
                    generation_digest=state.generation_digest,
                    committed_version=str(policy_version),
                    policy_digest=policy_digest,
                    update_identity=update_identity,
                    telemetry=ReplicaTelemetry(
                        observed_at=now,
                        in_flight=0,
                        capacity=int(
                            self.config.get("engine_args", {}).get("max_num_seqs")
                            or 256
                        ),
                    ),
                )
            )
        return RoutingTable(
            policy_generation=self._policy_generation,
            policy_version=str(policy_version),
            policy_digest=policy_digest,
            update_identity=update_identity,
            replicas=tuple(replicas),
        )

    def _gateway_advertise_host(self) -> str:
        if host := self.runtime.config.gateway_advertise_host:
            return host
        controller = next(
            host
            for host in self.runtime.topology.cluster.hosts
            if host.host_id == self.runtime.topology.cluster.controller_host_id
        )
        from urllib.parse import urlparse

        host = urlparse(controller.worker_address).hostname
        if host is None:
            raise ValueError("controller worker address has no routable hostname")
        return host

    def _engine_args(self, server: dev.OpenAIServerConfig | None) -> dict[str, object]:
        handler = get_model_support_handler(
            self.base_model,
            allow_unvalidated_arch=self._allow_unvalidated_arch,
        )
        values = dict(self.config.get("engine_args", {}))
        values.update(dict((server or {}).get("engine_args", {})))
        for key, value in handler.vllm_engine_args(rollout_weights_mode="lora").items():
            values.setdefault(key, value)
        values["enable_sleep_mode"] = False
        values.pop("enable_lora", None)
        values.setdefault("max_loras", 2)
        values.setdefault("generation_config", "vllm")
        for key in ("model", "served_model_name"):
            values.pop(key, None)
        return values

    def _server_args(self, server: dev.OpenAIServerConfig | None) -> dict[str, object]:
        handler = get_model_support_handler(
            self.base_model,
            allow_unvalidated_arch=self._allow_unvalidated_arch,
        )
        values: dict[str, object] = {
            "return_tokens_as_token_ids": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
            **handler.vllm_server_args(),
            **dict((server or {}).get("server_args", {})),
        }
        for key in ("port", "host", "lora_modules"):
            values.pop(key, None)
        return values

    def _api_key(self, server: dev.OpenAIServerConfig | None = None) -> str | None:
        value = dict((server or {}).get("server_args", {})).get("api_key")
        return cast(str | None, value) if value is not None else self._api_key_value

    async def _load_adapter(
        self, checkpoint: str, step: int, *, exact: bool = False
    ) -> None:
        name = (
            f"{self.model_name}:eval@{step}"
            if exact and self.rollout_weight_update_mode == "in_flight_lora"
            else f"{self.model_name}@{step}"
        )
        path = map_checkpoint_path_for_vllm(self.config, checkpoint)
        in_flight = (
            not exact
            and self.rollout_weight_update_mode == "in_flight_lora"
            and step != self._latest_step
        )
        endpoint = (
            "/art/in_flight_lora_update" if in_flight else "/v1/load_lora_adapter"
        )
        payload = (
            {
                "model_name": f"{self.model_name}@{step}",
                "lora_slot": f"{self.model_name}:active",
                "lora_path": path,
                "policy_version": step,
            }
            if in_flight
            else {"lora_name": name, "lora_path": path}
        )
        async with httpx.AsyncClient(timeout=60.0) as client:
            responses = await asyncio.gather(
                *(
                    client.post(
                        f"{url}{endpoint}",
                        json=payload,
                        headers=_headers(self._api_key()),
                    )
                    for url in self._base_urls
                )
            )
        for response in responses:
            response.raise_for_status()

    async def register_lora_for_step(self, step: int, checkpoint: str) -> None:
        if not self._base_urls:
            self._latest_step = step
            return
        digest = await asyncio.to_thread(_checkpoint_digest, checkpoint)
        update_identity = uuid.uuid4().hex
        try:
            states = {
                replica_id: self.runtime.replica(replica_id).prepare_update(
                    update_identity=update_identity
                )
                for replica_id in self._replica_ids
            }
            if (
                self._gateway is not None
                and self.rollout_weight_update_mode == "in_flight_lora"
            ):
                await self._gateway.pause("in-flight LoRA update")
            await self._load_adapter(checkpoint, step)
            reports = []
            for replica_id in self._replica_ids:
                manager = self.runtime.replica(replica_id)
                state = states[replica_id]
                report = ReplicaUpdateReport(
                    replica_id=replica_id,
                    generation=state.generation,
                    generation_digest=state.generation_digest,
                    policy_version=str(step),
                    policy_digest=digest,
                    update_identity=update_identity,
                )
                if manager.verify_update(report).phase != "ready":
                    raise RuntimeError(f"replica {replica_id!r} rejected LoRA update")
                reports.append(report)
            if self._gateway is not None:
                service = next(
                    service
                    for service in self.runtime.topology.model_services
                    if service.name == self.model_name
                )
                self._policy_generation += 1
                await self._gateway.commit(
                    self._routing_table(
                        service,
                        policy_version=step,
                        policy_digest=digest,
                        update_identity=update_identity,
                    ),
                    tuple(reports),
                )
        except BaseException:
            for replica_id in self._replica_ids:
                self.runtime.replica(replica_id).quarantine(
                    "partial or failed LoRA update"
                )
            if self._gateway is not None:
                await self._gateway.pause("partial or failed LoRA update")
            raise
        self._latest_step = step
        self._loaded_adapter_steps.add(step)

    async def acquire_exact_adapter(self, step: int, checkpoint: str) -> str:
        async with self._exact_adapter_lock:
            if step not in self._loaded_exact_adapter_steps:
                if (
                    self.rollout_weight_update_mode == "in_flight_lora"
                    or step not in self._loaded_adapter_steps
                ):
                    await self._load_adapter(checkpoint, step, exact=True)
                self._loaded_exact_adapter_steps.add(step)
                self._exact_adapter_refcounts[step] = 0
                if self._gateway is not None:
                    service = next(
                        service
                        for service in self.runtime.topology.model_services
                        if service.name == self.model_name
                    )
                    digest = await asyncio.to_thread(_checkpoint_digest, checkpoint)
                    self._gateway.add_policy(
                        self._routing_table(
                            service,
                            policy_version=step,
                            policy_digest=digest,
                            update_identity=f"exact:{step}:{digest}",
                        )
                    )
            self._exact_adapter_refcounts[step] += 1
        return (
            f"{self.model_name}:eval@{step}"
            if self.rollout_weight_update_mode == "in_flight_lora"
            else f"{self.model_name}@{step}"
        )

    async def release_exact_adapter(self, step: int) -> None:
        async with self._exact_adapter_lock:
            count = self._exact_adapter_refcounts.get(step, 0)
            if count <= 1:
                self._exact_adapter_refcounts.pop(step, None)
                if self._gateway is not None:
                    self._gateway.remove_policy(str(step))
                if self.rollout_weight_update_mode == "in_flight_lora":
                    await self._unload_adapter(f"{self.model_name}:eval@{step}")
                self._loaded_exact_adapter_steps.discard(step)
            else:
                self._exact_adapter_refcounts[step] = count - 1

    async def prune_loaded_adapters(self, *, retain_steps: set[int]) -> None:
        for step in sorted(
            self._loaded_adapter_steps - retain_steps - {self._latest_step}
        ):
            await self._unload_adapter(f"{self.model_name}@{step}")
            self._loaded_adapter_steps.discard(step)
            if self._gateway is not None:
                self._gateway.remove_policy(str(step))

    async def _unload_adapter(self, name: str) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            responses = await asyncio.gather(
                *(
                    client.post(
                        f"{url}/v1/unload_lora_adapter",
                        json={"lora_name": name},
                        headers=_headers(self._api_key()),
                    )
                    for url in self._base_urls
                )
            )
        for response in responses:
            if response.status_code != 404:
                response.raise_for_status()

    async def get_serving_capabilities(self) -> ServingCapabilities:
        if self._serving_capabilities is None:
            raise RuntimeError("vLLM serving capabilities have not been discovered")
        return self._serving_capabilities

    async def vllm_engine_is_sleeping(self) -> bool:
        return False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        if self._gateway is not None:
            try:
                await self._gateway.close()
            except BaseException as error:
                failures.append(error)
        if self._trainer is not None:
            try:
                await self._trainer.close()
            except BaseException as error:
                failures.append(error)
        for replica_id in self._replica_ids:
            try:
                await self.runtime.stop_replica(replica_id)
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("distributed model service close failed", failures)


def _copy_checkpoint(source: str, destination: str) -> None:
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _publish_checkpoint(staging: str, output_dir: str, step: int) -> str:
    destination = get_step_checkpoint_dir(output_dir, step)
    if os.path.exists(destination):
        raise RuntimeError(f"refusing to replace checkpoint {destination}")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(staging).rename(destination)
    return destination


def _trainer_dtype(
    config: dev.BackendModelConfig,
) -> Literal["bfloat16", "float16", "float32"]:
    value = str(config.get("init_args", {}).get("dtype") or "bfloat16").lower()
    value = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
        "torch.bfloat16": "bfloat16",
        "torch.float16": "float16",
        "torch.float32": "float32",
    }.get(value, value)
    if value not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported Megatron trainer dtype {value!r}")
    return cast(
        Literal["bfloat16", "float16", "float32"],
        value,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _art_source_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _checkpoint_digest(path: str) -> str:
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        digest.update((Path(path) / name).read_bytes())
    return digest.hexdigest()


def _headers(api_key: str | None) -> dict[str, str] | None:
    return {"Authorization": f"Bearer {api_key}"} if api_key else None


def _host_port(base_url: str) -> tuple[str, int]:
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    assert parsed.hostname is not None
    return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
