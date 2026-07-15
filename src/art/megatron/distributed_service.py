from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
from art.distributed.specs import vllm_kv_event_topic
from art.distributed.vllm_replica import (
    ReplicaLaunchTemplate,
    ReplicaUpdateReport,
)
from art.distributed.vllm_router import (
    ReplicaTelemetry,
    RoutableReplica,
    RoutingTable,
    VllmPrefixHashConfig,
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

from .identity_lora import create_identity_lora
from .lora_config import LORA_ALPHA, default_lora_rank_for_handler
from .model_support import (
    get_model_support_handler,
    get_model_support_handler_for_spec,
    get_model_support_spec,
    model_uses_expert_parallel,
)
from .optimizer_state import (
    OptimizerAdapter,
    async_optimizer_model_lease,
    format_megatron_resume_message,
    hash_adapter_checkpoint,
    prepare_megatron_resume_state,
    publish_adapter_checkpoint,
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


class DistributedMegatronService:
    """One model's durable checkpoints and run-scoped distributed runtimes."""

    propagate_close_errors = True
    close_timeout_s = 60.0

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
        self._mutation_lock = asyncio.Lock()
        self._replica_ids: tuple[str, ...] = ()
        self._base_urls: tuple[str, ...] = ()
        self._gateway: Any = None
        self._gateway_endpoint: tuple[str, int] | None = None
        self._policy_generation = 0
        self._serving_capabilities: ServingCapabilities | None = None
        self._api_key_value: str | None = None
        self._published_adapters: dict[int, OptimizerAdapter] = {}
        self._loaded_adapter_steps: set[int] = set()
        self._loaded_exact_adapter_steps: set[int] = set()
        self._exact_adapter_refcounts: dict[int, int] = {}
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

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("distributed model service is closed")

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

    @property
    def _model_identifier(self) -> str:
        value = self.config.get("init_args", {}).get("model_name", self.base_model)
        if not isinstance(value, str) or not value:
            raise ValueError("init_args.model_name must be a non-empty string")
        return value

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
            handler = get_model_support_handler(
                self.base_model,
                allow_unvalidated_arch=self._allow_unvalidated_arch,
            )
            create_identity_lora(
                self._model_identifier,
                path,
                rank=lora.get("rank"),
                target_modules=lora.get("target_modules"),
                random_state=self._random_state(),
                allow_unvalidated_arch=self._allow_unvalidated_arch,
                handler=handler,
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
        support_spec = get_model_support_spec(
            self.base_model,
            allow_unvalidated_arch=self._allow_unvalidated_arch,
        )
        handler = get_model_support_handler_for_spec(support_spec)
        targets = lora.get("target_modules") or default_target_modules(self.base_model)
        revision = str(self.config.get("init_args", {}).get("revision") or "default")
        compile_enabled = os.environ.get(
            "ART_DISABLE_MEGATRON_COMPILE", "0"
        ).lower() not in {"1", "true", "yes", "on"}
        identity = {
            "art": _art_source_revision(),
            "model": self._model_identifier,
            "support_model": self.base_model,
            "revision": revision,
            "handler": handler.key,
            "mesh": mesh.model_dump(mode="json"),
        }
        return TrainerRuntimeSpec(
            art_revision=identity["art"],
            model_identifier=self._model_identifier,
            model_revision=revision,
            model_support_key=support_spec.key,
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

    async def _ensure_trainer_locked(self) -> Any:
        if self._trainer is not None:
            return self._trainer
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

    @asynccontextmanager
    async def _trainer_transaction(
        self,
        trainer: Any,
        job: TrainJobSpec,
        leases: Any,
        *,
        source: str,
        staging: str,
    ) -> AsyncIterator[AsyncIterator[Any]]:
        async with async_optimizer_model_lease(job.output.optimizer_state_path):
            await asyncio.to_thread(_copy_checkpoint, source, staging)
            yield trainer.train(job, leases)

    async def train_packed(
        self,
        batch: DistributedPackedBatch,
        config: types.TrainConfig,
        experimental_config: dev.TrainConfig,
    ) -> AsyncIterator[dict[str, float]]:
        async with self._mutation_lock:
            self._require_open()
            trainer = await self._ensure_trainer_locked()
            next_step = self._latest_step + 1
            source = get_step_checkpoint_dir(self.output_dir, self._latest_step)
            staging = f"{self.output_dir}/megatron_runtime/staging/{next_step:04d}"
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
                    optimizer_lease_owner="controller",
                ),
            )
            adapter_ready = False
            completed = False
            checkpoint: str | None = None
            async with self._trainer_transaction(
                trainer,
                job,
                batch.leases,
                source=source,
                staging=staging,
            ) as events:
                async for event in events:
                    if event.job_id != job.job_id or event.run_id != job.run_id:
                        raise RuntimeError(
                            "trainer returned an event for a different job"
                        )
                    if isinstance(event, TrainAccepted):
                        continue
                    if isinstance(event, TrainProgress):
                        yield event.metrics
                        continue
                    if isinstance(event, AdapterReady):
                        if adapter_ready:
                            raise RuntimeError(
                                "trainer returned duplicate AdapterReady events"
                            )
                        if (
                            event.learner_version != next_step
                            or event.adapter_path != staging
                        ):
                            raise RuntimeError("trainer prepared the wrong adapter")
                        published = _publish_checkpoint(
                            staging, self.output_dir, next_step
                        )
                        checkpoint = published.identity
                        self._published_adapters[next_step] = published
                        adapter_ready = True
                        continue
                    if isinstance(event, TrainCompleted):
                        if completed:
                            raise RuntimeError(
                                "trainer returned duplicate TrainCompleted events"
                            )
                        if not adapter_ready:
                            raise RuntimeError(
                                "trainer completed before preparing the adapter"
                            )
                        if event.learner_version != next_step:
                            raise RuntimeError(
                                "trainer completed the wrong learner version"
                            )
                        completed = True
                        continue
                    if isinstance(event, TrainFailed):
                        raise RuntimeError(
                            f"distributed Megatron job failed ({event.error_type}): "
                            f"{event.message}"
                        )
                    if isinstance(event, TrainCancelled):
                        raise asyncio.CancelledError(event.reason)
            if not adapter_ready or not completed:
                raise RuntimeError(
                    "trainer ended without preparing and durably completing the adapter"
                )
            assert checkpoint is not None
            try:
                await self._register_lora_for_step_locked(next_step, checkpoint)
            except BaseException:
                # Training is durable before serving publication. Preserve that
                # lineage even when the serving generation fails closed.
                self._latest_step = next_step
                raise
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
        async with self._mutation_lock:
            self._require_open()
            if self._base_urls:
                return self._gateway_endpoint or _host_port(self._base_urls[0])
            return await self._start_openai_server_locked(config)

    async def _start_openai_server_locked(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        api_key = self._api_key(config)
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
            capabilities = await discover_serving_capabilities(
                base_url=base_url,
                headers=headers,
                allow_openai_compatible=True,
            )
            base_urls = (base_url,)
            await self._load_adapter_at(
                lora_path,
                self._latest_step,
                base_urls=base_urls,
                api_key=api_key,
                latest_step=self._latest_step,
            )
            self._publish_serving_state(
                replica_ids=(),
                base_urls=base_urls,
                gateway=None,
                gateway_endpoint=None,
                capabilities=capabilities,
                api_key=api_key,
            )
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
        replica_ids = tuple(replica.replica_id for replica in service.replicas)
        start_tasks = tuple(
            asyncio.create_task(self.runtime.start_replica(replica, template))
            for replica in service.replicas
        )
        try:
            starts = await asyncio.gather(*start_tasks, return_exceptions=True)
        except BaseException as error:
            for task in start_tasks:
                task.cancel()
            starts = await asyncio.gather(*start_tasks, return_exceptions=True)
            started = tuple(
                replica_id
                for replica_id, value in zip(replica_ids, starts, strict=True)
                if not isinstance(value, BaseException)
            )
            cleanup = await self._rollback_server_start(None, started)
            if cleanup:
                raise BaseExceptionGroup(
                    "vLLM replica startup cancellation and rollback failed",
                    [error, *cleanup],
                ) from None
            raise
        failures = [value for value in starts if isinstance(value, BaseException)]
        if failures:
            started = tuple(
                replica_id
                for replica_id, value in zip(replica_ids, starts, strict=True)
                if not isinstance(value, BaseException)
            )
            failures.extend(await self._rollback_server_start(None, started))
            raise BaseExceptionGroup("vLLM replica startup failed", failures)
        base_urls = tuple(
            f"http://{replica.leader_endpoint.host}:{replica.leader_endpoint.port}"
            for replica in service.replicas
        )
        gateway: Any = None
        try:
            capabilities = await asyncio.gather(
                *(
                    discover_serving_capabilities(
                        base_url=url,
                        headers=_headers(api_key),
                        allow_openai_compatible=False,
                    )
                    for url in base_urls
                )
            )
            if len(set(capabilities)) != 1:
                raise RuntimeError("vLLM replicas expose different ART capabilities")
            capabilities[0].require(
                "exact_lora_worker_state", operation="distributed LoRA publication"
            )
            for replica_id, capability in zip(replica_ids, capabilities, strict=True):
                hash_block_size = capability.prefix_hash_block_size
                if hash_block_size is None:
                    raise RuntimeError(
                        "distributed prefix routing requires the effective vLLM "
                        "hash block size"
                    )
                self.runtime.replica(replica_id).confirm_prefix_hash_block_size(
                    hash_block_size
                )
            digest = await self._checkpoint_digest(lora_path, self._latest_step)
            update_identity = uuid.uuid4().hex
            states = {
                replica_id: self.runtime.replica(replica_id).prepare_update(
                    update_identity=update_identity
                )
                for replica_id in replica_ids
            }
            await self._acknowledge_lora_workers(
                lora_name=template.served_model_name,
                lora_path=lora_path,
                step=self._latest_step,
                replica_ids=replica_ids,
                base_urls=base_urls,
                api_key=api_key,
            )
            for replica_id, state in states.items():
                manager = self.runtime.replica(replica_id)
                report = ReplicaUpdateReport(
                    replica_id=replica_id,
                    generation=state.generation,
                    generation_digest=state.generation_digest,
                    policy_version=str(self._latest_step),
                    policy_digest=digest,
                    update_identity=update_identity,
                )
                if manager.verify_update(report).phase != "ready":
                    raise RuntimeError(
                        f"replica {replica_id!r} rejected initial policy"
                    )
            gateway_endpoint = None
            if len(replica_ids) > 1:
                from art.distributed.vllm_gateway import VllmGateway

                gateway = VllmGateway(
                    self._routing_table(
                        service,
                        policy_version=self._latest_step,
                        policy_digest=digest,
                        update_identity=update_identity,
                        lora_name=template.served_model_name,
                        replica_ids=replica_ids,
                    ),
                    upstream_headers=_headers(api_key),
                    max_queued=self.runtime.config.gateway_max_queued,
                    route_timeout_s=self.runtime.config.gateway_route_timeout_s,
                    kv_event_sources=self._kv_event_sources(service, replica_ids),
                )
                port = await gateway.start(self.runtime.config.gateway_bind_host)
                gateway_endpoint = (self._gateway_advertise_host(), port)
        except BaseException as error:
            cleanup = await self._rollback_server_start(gateway, replica_ids)
            if cleanup:
                raise BaseExceptionGroup(
                    "vLLM startup validation and rollback failed", [error, *cleanup]
                ) from None
            raise
        self._publish_serving_state(
            replica_ids=replica_ids,
            base_urls=base_urls,
            gateway=gateway,
            gateway_endpoint=gateway_endpoint,
            capabilities=capabilities[0],
            api_key=api_key,
        )
        return gateway_endpoint or _host_port(base_urls[0])

    def _publish_serving_state(
        self,
        *,
        replica_ids: tuple[str, ...],
        base_urls: tuple[str, ...],
        gateway: Any,
        gateway_endpoint: tuple[str, int] | None,
        capabilities: ServingCapabilities,
        api_key: str | None,
    ) -> None:
        self._replica_ids = replica_ids
        self._base_urls = base_urls
        self._gateway = gateway
        self._gateway_endpoint = gateway_endpoint
        self._serving_capabilities = capabilities
        self._api_key_value = api_key
        self._loaded_adapter_steps.add(self._latest_step)

    def _clear_serving_state(self) -> None:
        self._replica_ids = ()
        self._base_urls = ()
        self._gateway = None
        self._gateway_endpoint = None
        self._serving_capabilities = None
        self._api_key_value = None
        self._loaded_adapter_steps.clear()
        self._loaded_exact_adapter_steps.clear()
        self._exact_adapter_refcounts.clear()

    async def _rollback_server_start(
        self, gateway: Any, replica_ids: tuple[str, ...]
    ) -> list[BaseException]:
        cleanup = ([gateway.close()] if gateway is not None else []) + [
            self.runtime.stop_replica(replica_id) for replica_id in replica_ids
        ]
        return [
            result
            for result in await asyncio.gather(*cleanup, return_exceptions=True)
            if isinstance(result, BaseException)
        ]

    def _routing_table(
        self,
        service: Any,
        *,
        policy_version: int,
        policy_digest: str,
        update_identity: str,
        lora_name: str,
        replica_ids: tuple[str, ...] | None = None,
        policy_generation: int | None = None,
    ) -> RoutingTable:
        now = asyncio.get_running_loop().time()
        specs = {replica.replica_id: replica for replica in service.replicas}
        replicas = []
        for replica_id in self._replica_ids if replica_ids is None else replica_ids:
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
                    kv_event_publishers=spec.parallel.dp,
                )
            )
        target_replica_ids = self._replica_ids if replica_ids is None else replica_ids
        hash_block_sizes = {
            self.runtime.replica(replica_id).prefix_hash_block_size
            for replica_id in target_replica_ids
        }
        if len(hash_block_sizes) != 1:
            raise RuntimeError("vLLM replicas use inconsistent hash block sizes")
        policy_cache_key = (
            f"{lora_name}:{policy_version}"
            if lora_name == f"{self.model_name}:active"
            else None
        )
        return RoutingTable(
            policy_generation=(
                self._policy_generation
                if policy_generation is None
                else policy_generation
            ),
            policy_version=str(policy_version),
            policy_digest=policy_digest,
            update_identity=update_identity,
            replicas=tuple(replicas),
            prefix_hash=VllmPrefixHashConfig(
                block_size=hash_block_sizes.pop(),
                lora_name=lora_name,
                policy_cache_key=policy_cache_key,
            ),
        )

    def _kv_event_sources(
        self, service: Any, replica_ids: tuple[str, ...]
    ) -> tuple[Any, ...]:
        from art.distributed.vllm_kv_events import KvEventSource

        hosts = {
            host.host_id: _worker_hostname(host.worker_address)
            for host in self.runtime.topology.cluster.hosts
        }
        sources = []
        specs = {replica.replica_id: replica for replica in service.replicas}
        for replica_id in replica_ids:
            spec = specs[replica_id]
            state = self.runtime.replica(replica_id).state
            topic = vllm_kv_event_topic(
                replica_id, state.generation, state.generation_digest
            )
            for publisher_rank in range(spec.parallel.dp):
                host = hosts[spec.kv_event_member(publisher_rank).host_id]
                sources.append(
                    KvEventSource(
                        replica_id=replica_id,
                        generation=state.generation,
                        publisher_rank=publisher_rank,
                        endpoint=_zmq_endpoint(
                            host, spec.kv_event_base_port + publisher_rank
                        ),
                        replay_endpoint=_zmq_endpoint(
                            host, spec.kv_replay_base_port + publisher_rank
                        ),
                        topic=topic,
                    )
                )
        return tuple(sources)

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
    ) -> tuple[str, str]:
        return await self._load_adapter_at(
            checkpoint,
            step,
            exact=exact,
            base_urls=self._base_urls,
            api_key=self._api_key(),
            latest_step=self._latest_step,
        )

    async def _load_adapter_at(
        self,
        checkpoint: str,
        step: int,
        *,
        base_urls: tuple[str, ...],
        api_key: str | None,
        latest_step: int,
        exact: bool = False,
    ) -> tuple[str, str]:
        name = (
            f"{self.model_name}:eval@{step}"
            if exact and self.rollout_weight_update_mode == "in_flight_lora"
            else f"{self.model_name}@{step}"
        )
        path = map_checkpoint_path_for_vllm(self.config, checkpoint)
        in_flight = (
            not exact
            and self.rollout_weight_update_mode == "in_flight_lora"
            and step != latest_step
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
                        headers=_headers(api_key),
                    )
                    for url in base_urls
                )
            )
        for response in responses:
            response.raise_for_status()
        return str(payload.get("lora_slot", name)), path

    async def _acknowledge_lora_workers(
        self,
        *,
        lora_name: str,
        lora_path: str,
        step: int,
        replica_ids: tuple[str, ...] | None = None,
        base_urls: tuple[str, ...] | None = None,
        api_key: str | None = None,
    ) -> None:
        replica_ids = self._replica_ids if replica_ids is None else replica_ids
        base_urls = self._base_urls if base_urls is None else base_urls
        if not replica_ids:
            return
        if len(replica_ids) != len(base_urls):
            raise RuntimeError("replica IDs and endpoints are inconsistent")
        payloads = tuple(
            {
                "expected_workers": self.runtime.replica(
                    replica_id
                ).expected_worker_identities(),
                "expected_lora": {
                    "lora_name": lora_name,
                    "lora_path": lora_path,
                    "policy_version": step,
                },
            }
            for replica_id in replica_ids
        )
        async with httpx.AsyncClient(timeout=60.0) as client:
            responses = await asyncio.gather(
                *(
                    client.post(
                        f"{url}/art/lora_worker_state",
                        json=payload,
                        headers=_headers(
                            api_key if api_key is not None else self._api_key()
                        ),
                    )
                    for url, payload in zip(base_urls, payloads, strict=True)
                )
            )
        for response in responses:
            response.raise_for_status()

    async def register_lora_for_step(self, step: int, checkpoint: str) -> None:
        async with self._mutation_lock:
            self._require_open()
            await self._register_lora_for_step_locked(step, checkpoint)
            self._latest_step = step

    async def _register_lora_for_step_locked(self, step: int, checkpoint: str) -> None:
        if not self._base_urls:
            return
        digest = await self._checkpoint_digest(checkpoint, step)
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
            lora_name, lora_path = await self._load_adapter(checkpoint, step)
            await self._acknowledge_lora_workers(
                lora_name=lora_name,
                lora_path=lora_path,
                step=step,
            )
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
                policy_generation = self._policy_generation + 1
                await self._gateway.commit(
                    self._routing_table(
                        service,
                        policy_version=step,
                        policy_digest=digest,
                        update_identity=update_identity,
                        lora_name=lora_name,
                        policy_generation=policy_generation,
                    ),
                    tuple(reports),
                )
                self._policy_generation = policy_generation
        except BaseException as error:
            for replica_id in self._replica_ids:
                self.runtime.replica(replica_id).quarantine(
                    "partial or failed LoRA update"
                )
            cleanup = await self._rollback_server_start(
                self._gateway, self._replica_ids
            )
            self._clear_serving_state()
            if cleanup:
                raise BaseExceptionGroup(
                    "LoRA publication and serving rollback failed", [error, *cleanup]
                ) from None
            raise
        self._loaded_adapter_steps.add(step)

    async def _checkpoint_digest(self, checkpoint: str, step: int) -> str:
        published = self._published_adapters.get(step)
        if published is not None and published.identity == str(
            Path(checkpoint).absolute()
        ):
            return published.sha256
        return await asyncio.to_thread(hash_adapter_checkpoint, checkpoint)

    async def acquire_exact_adapter(self, step: int, checkpoint: str) -> str:
        async with self._mutation_lock:
            self._require_open()
            lora_name = (
                f"{self.model_name}:eval@{step}"
                if self.rollout_weight_update_mode == "in_flight_lora"
                else f"{self.model_name}@{step}"
            )
            if step not in self._loaded_exact_adapter_steps:
                if (
                    self.rollout_weight_update_mode == "in_flight_lora"
                    or step not in self._loaded_adapter_steps
                ):
                    lora_name, lora_path = await self._load_adapter(
                        checkpoint, step, exact=True
                    )
                    await self._acknowledge_lora_workers(
                        lora_name=lora_name,
                        lora_path=lora_path,
                        step=step,
                    )
                self._loaded_exact_adapter_steps.add(step)
                self._exact_adapter_refcounts[step] = 0
                if self._gateway is not None:
                    service = next(
                        service
                        for service in self.runtime.topology.model_services
                        if service.name == self.model_name
                    )
                    digest = await self._checkpoint_digest(checkpoint, step)
                    self._gateway.add_policy(
                        self._routing_table(
                            service,
                            policy_version=step,
                            policy_digest=digest,
                            update_identity=f"exact:{step}:{digest}",
                            lora_name=lora_name,
                        )
                    )
            self._exact_adapter_refcounts[step] += 1
        return (
            f"{self.model_name}:eval@{step}"
            if self.rollout_weight_update_mode == "in_flight_lora"
            else f"{self.model_name}@{step}"
        )

    async def release_exact_adapter(self, step: int) -> None:
        async with self._mutation_lock:
            self._require_open()
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
        async with self._mutation_lock:
            self._require_open()
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
        async with self._mutation_lock:
            if self._closed:
                return
            self._closed = True
            operations = []
            if self._gateway is not None:
                operations.append(self._gateway.close())
            if self._trainer is not None:
                operations.append(self._trainer.close())
            operations.extend(
                self.runtime.stop_replica(replica_id)
                for replica_id in self._replica_ids
            )
            results = await asyncio.gather(*operations, return_exceptions=True)
            failures = [
                result for result in results if isinstance(result, BaseException)
            ]
            if failures:
                raise BaseExceptionGroup(
                    "distributed model service close failed", failures
                )


def _copy_checkpoint(source: str, destination: str) -> None:
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _publish_checkpoint(staging: str, output_dir: str, step: int) -> OptimizerAdapter:
    adapter = publish_adapter_checkpoint(staging, step=step)
    expected = str(Path(get_step_checkpoint_dir(output_dir, step)).absolute())
    if adapter.identity != expected:
        raise RuntimeError(
            "Published adapter path does not match the service output: "
            f"published={adapter.identity}, expected={expected}"
        )
    return adapter


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


def _headers(api_key: str | None) -> dict[str, str] | None:
    return {"Authorization": f"Bearer {api_key}"} if api_key else None


def _worker_hostname(worker_address: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(worker_address)
    if parsed.scheme != "tcp" or parsed.hostname is None:
        raise ValueError("model-service worker addresses must use routable tcp:// URLs")
    return parsed.hostname


def _zmq_endpoint(host: str, port: int) -> str:
    return f"tcp://[{host}]:{port}" if ":" in host else f"tcp://{host}:{port}"


def _host_port(base_url: str) -> tuple[str, int]:
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    assert parsed.hostname is not None
    return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
