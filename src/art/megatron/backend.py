import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import secrets
from typing import Any, AsyncIterator, Iterable, Literal, cast
import uuid

from pydantic import BaseModel, ConfigDict, Field

from mp_actors import move_to_child_process

from .. import dev
from ..backend import AnyTrainableModel
from ..distributed.art_runtime import ArtRuntime
from ..local.backend import LocalBackend, _PackedTrainingBatch
from ..local.service import ModelService
from ..model import TrainableModel
from ..trajectories import TrajectoryGroup
from ..types import LocalTrainResult
from ..utils.output_dirs import get_model_dir, get_step_checkpoint_dir
from ..vllm_runtime import get_external_vllm_runtime_config
from .optimizer_state import (
    format_megatron_resume_message,
    prepare_megatron_resume_state,
)
from .runtime_config import get_megatron_runtime_config


class _DistributedBatchPayload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    packed: Any
    selections: tuple[Any, ...]
    generation_id: str = Field(min_length=1)


class _PipelinePreparedBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    batch: Any
    groups: tuple[Any, ...]
    include_moe_routing: bool
    metrics: dict[str, float]


class MegatronBackend(LocalBackend):
    def __init__(
        self,
        *,
        in_process: bool = False,
        path: str | None = None,
        enable_expert_replay: bool = True,
        runtime: ArtRuntime | None = None,
    ) -> None:
        if runtime is not None:
            artifact_root = runtime.topology.cluster.artifact_root
            if artifact_root is None:
                raise ValueError("distributed Megatron requires cluster.artifact_root")
            if (
                path is not None
                and Path(path).resolve() != Path(artifact_root).resolve()
            ):
                raise ValueError("backend path must match cluster.artifact_root")
            path = artifact_root
        super().__init__(
            in_process=in_process,
            path=path,
            enable_expert_replay=enable_expert_replay,
        )
        self._requires_explicit_packed_sequence_length = True
        self._packed_sequence_length_requires_chunk_alignment = False
        self._supports_result_packing = True
        self._resume_prepared_models: set[str] = set()
        self._runtime = runtime
        self._managed_api_key = (
            secrets.token_urlsafe(32) if runtime is not None else None
        )

    def __enter__(self) -> "MegatronBackend":
        if self._runtime is not None:
            raise TypeError("distributed MegatronBackend requires 'async with'")
        return super().__enter__()

    async def train(
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        **kwargs: Any,
    ) -> LocalTrainResult:
        for removed_kwarg in ("packed_sequence_length", "megatron_topology"):
            if removed_kwarg in kwargs:
                raise TypeError(
                    f"MegatronBackend.train gets {removed_kwarg} from "
                    "art.init_megatron_runtime_config(...)."
                )
        result = await super().train(
            model,
            trajectory_groups,
            packed_sequence_length=get_megatron_runtime_config().packed_sequence_length,
            **kwargs,
        )
        if self._runtime is None or not kwargs.get("save_checkpoint", True):
            return result
        from .distributed_service import DistributedMegatronService

        service = cast(DistributedMegatronService, await self._get_service(model))
        result.checkpoint_path = get_step_checkpoint_dir(
            get_model_dir(model=model, art_path=self._path), result.step
        )
        if not Path(result.checkpoint_path).exists():
            result.checkpoint_ready = service.checkpoint_materialization(result.step)
        return result

    def _supports_concurrent_training_and_inference(
        self, model: AnyTrainableModel
    ) -> bool:
        return (
            self._runtime is not None
            or super()._supports_concurrent_training_and_inference(model)
        )

    @asynccontextmanager
    async def adapter_lease(
        self,
        model: AnyTrainableModel,
        step: int,
    ) -> AsyncIterator[None]:
        if self._runtime is not None:
            from .distributed_service import DistributedMegatronService

            service = cast(DistributedMegatronService, await self._get_service(model))
            await service.wait_for_serving(step)
        async with super().adapter_lease(model, step):
            yield

    @asynccontextmanager
    async def exact_adapter_lease(
        self,
        model: AnyTrainableModel,
        step: int,
    ) -> AsyncIterator[None]:
        if self._runtime is not None:
            from .distributed_service import DistributedMegatronService

            service = cast(DistributedMegatronService, await self._get_service(model))
            await service.wait_for_serving(step)
        async with super().exact_adapter_lease(model, step):
            yield

    async def _get_service(self, model: TrainableModel) -> ModelService:
        from ..dev.get_model_config import get_model_config

        if model.name not in self._services:
            config = get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
                lora_config=model.lora_config,
            )
            if self._runtime is not None:
                from .distributed_service import DistributedMegatronService

                config["init_args"]["model_name"] = (
                    (model._internal_config or {})
                    .get("init_args", {})
                    .get("model_name", model.base_model)
                )

                service = cast(
                    ModelService,
                    DistributedMegatronService(
                        model_name=model.name,
                        base_model=model.base_model,
                        config=config,
                        output_dir=get_model_dir(model=model, art_path=self._path),
                        runtime=self._runtime,
                        enable_expert_replay=self._enable_expert_replay,
                    ),
                )
                self._runtime.register_closeable(service)
                self._services[model.name] = service
            else:
                from .service import MegatronService

                self._services[model.name] = MegatronService(
                    model_name=model.name,
                    base_model=model.base_model,
                    config=config,
                    output_dir=get_model_dir(model=model, art_path=self._path),
                    enable_expert_replay=self._enable_expert_replay,
                )
            if self._runtime is None and not self._in_process:
                self._services[model.name] = move_to_child_process(
                    self._services[model.name],
                    process_name="megatron-service",
                )
        return self._services[model.name]

    async def _prepare_backend_for_training(
        self,
        model: AnyTrainableModel,
        config: dev.OpenAIServerConfig | None = None,
    ) -> tuple[str, str]:
        if (
            self._managed_api_key is None
            or get_external_vllm_runtime_config(model._internal_config or {})
            is not None
        ):
            return await super()._prepare_backend_for_training(model, config)
        config_dict = dict(config or {})
        server_args = dict(config_dict.get("server_args", {}))
        server_args.setdefault("api_key", self._managed_api_key)
        config_dict["server_args"] = server_args
        return await super()._prepare_backend_for_training(
            model, cast(dev.OpenAIServerConfig, config_dict)
        )

    async def _prepare_training_batch(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        dev_config: Any,
        *,
        include_moe_routing: bool,
    ) -> _PackedTrainingBatch | None:
        prepared = tuple(group._prepared_training_batch for group in trajectory_groups)
        if any(value is not None for value in prepared):
            first = prepared[0]
            if (
                not isinstance(first, _PipelinePreparedBatch)
                or any(value is not first for value in prepared)
                or first.groups != tuple(trajectory_groups)
                or first.include_moe_routing != include_moe_routing
            ):
                raise RuntimeError("pipeline prepared batch does not match training")
            for group in trajectory_groups:
                group._prepared_training_batch = None
            return cast(_PackedTrainingBatch, first.batch)
        if self._runtime is None:
            return await super()._prepare_training_batch(
                model,
                trajectory_groups,
                dev_config,
                include_moe_routing=include_moe_routing,
            )
        from ..distributed.packing import PackingRequest
        from ..distributed.rollout import (
            DistributedTrajectorySelection,
            RolloutModelSpec,
        )
        from ..distributed.trajectory_store import TrajectoryGroupBundle

        selections = tuple(group._distributed_lease for group in trajectory_groups)
        selected = tuple(
            selection
            for selection in selections
            if isinstance(selection, DistributedTrajectorySelection)
        )
        if selected and len(selected) != len(trajectory_groups):
            raise RuntimeError("distributed batch mixes owned and controller groups")
        queue = selected[0].queue if selected else None
        if queue is not None and any(
            selection.queue is not queue for selection in selected
        ):
            raise RuntimeError("distributed batch spans trajectory queues")

        versions = [
            version
            for group in trajectory_groups
            for trajectory in group.trajectories
            for version in (
                trajectory.initial_policy_version,
                trajectory.final_policy_version,
            )
            if version is not None
        ]
        current_step = min(versions) if versions else await self._get_step(model)
        if selected:
            group_ids = tuple(
                selection.lease.item.ref.result_id for selection in selected
            )
            record_ids = tuple(
                record.record_id
                for selection in selected
                for record in selection.lease.item.ref.records
            )
        else:
            group_ids = tuple(
                f"{group.metadata.get('scenario_id', 'group')}:{index}"
                for index, group in enumerate(trajectory_groups)
            )
            record_ids = tuple(
                f"{group_id}:{trajectory_index}"
                for group_id, group in zip(group_ids, trajectory_groups, strict=True)
                for trajectory_index, _ in enumerate(group.trajectories)
            )
        generation_id = uuid.uuid4().hex
        trajectory_log_path = (
            str(
                Path(get_model_dir(model=model, art_path=self._path))
                / "trajectories"
                / ".staging"
                / f"{generation_id}.parquet"
            )
            if selected
            else None
        )
        request = PackingRequest(
            model=RolloutModelSpec.from_model(model),
            generation_id=generation_id,
            trajectory_groups=(
                ()
                if selected
                else tuple(
                    TrajectoryGroupBundle.from_group(group)
                    for group in trajectory_groups
                )
            ),
            trajectory_sources=tuple(selection.lease.item for selection in selected),
            trajectory_log_path=trajectory_log_path,
            advantage_balance=dev_config.get("advantage_balance", 0.0),
            allow_training_without_logprobs=dev_config.get(
                "allow_training_without_logprobs", False
            ),
            scale_rewards=dev_config.get("scale_rewards", True),
            plot_tensors=dev_config.get("plot_tensors", False),
            packed_sequence_length=dev_config["packed_sequence_length"],
            logprob_calculation_chunk_size=dev_config.get(
                "logprob_calculation_chunk_size", 1024
            ),
            include_moe_routing=include_moe_routing,
            collect_packing_shapes=any(
                group._collect_packing_shape for group in trajectory_groups
            ),
            group_ids=group_ids,
            record_ids=record_ids,
            min_source_version=min(versions, default=current_step),
            max_source_version=max(versions, default=current_step),
        )
        try:
            packed = await self._runtime.pack(request)
        except BaseException as error:
            cleanup = await asyncio.gather(
                *(
                    (queue.release_selections(selected, disposition="discarded"),)
                    if queue is not None
                    else ()
                ),
                *(
                    (
                        asyncio.to_thread(
                            Path(trajectory_log_path).unlink, missing_ok=True
                        ),
                    )
                    if trajectory_log_path is not None
                    else ()
                ),
                return_exceptions=True,
            )
            for group in trajectory_groups:
                group._distributed_lease = None
            failures = [
                result for result in cleanup if isinstance(result, BaseException)
            ]
            if failures:
                raise BaseExceptionGroup(
                    "packing and source cleanup failed", [error, *failures]
                ) from None
            raise
        if packed is None:
            if queue is not None:
                await queue.release_selections(selected, disposition="discarded")
                for group in trajectory_groups:
                    group._distributed_lease = None
            return None
        if queue is not None:
            try:
                await queue.mark_packed(selected, generation_id)
            except BaseException as error:
                cleanup = await asyncio.gather(
                    self._runtime.release_batch(packed),
                    queue.release_selections(selected, disposition="discarded"),
                    *(
                        (
                            asyncio.to_thread(
                                Path(trajectory_log_path).unlink, missing_ok=True
                            ),
                        )
                        if trajectory_log_path is not None
                        else ()
                    ),
                    return_exceptions=True,
                )
                failures = [
                    result for result in cleanup if isinstance(result, BaseException)
                ]
                for group in trajectory_groups:
                    group._distributed_lease = None
                if failures:
                    raise BaseExceptionGroup(
                        "packing claim and cleanup failed", [error, *failures]
                    ) from None
                raise
            for group in trajectory_groups:
                group._distributed_lease = None
                group._prepared_log_path = packed.trajectory_log_path
        for group, shape in zip(
            trajectory_groups, packed.packed_group_shapes, strict=True
        ):
            if shape is not None:
                group._packed_group_shape = shape
        ref = packed.leases.ref
        stats = ref.prefix_tree_packing_stats
        if stats is None:
            raise RuntimeError("distributed packed batch has no prefix-tree statistics")
        return _PackedTrainingBatch(
            payload=(
                _DistributedBatchPayload(
                    packed=packed,
                    selections=selected,
                    generation_id=generation_id,
                )
                if selected
                else packed
            ),
            num_sequences=ref.num_sequences,
            sequence_length=ref.sequence_length,
            trainable_assistant_tokens=packed.trainable_assistant_tokens,
            loss_bearing_tokens=packed.loss_bearing_tokens,
            non_padding_tokens=packed.non_padding_tokens,
            logical_tokens=stats.logical_tokens,
            physical_tokens=stats.physical_tokens,
            include_moe_routing=include_moe_routing,
        )

    async def prepare_pipeline_batch(
        self, model: TrainableModel, trajectory_groups: list[TrajectoryGroup]
    ) -> dict[str, float] | None:
        include_moe_routing = self._model_uses_expert_replay(model)
        batch = await self._prepare_training_batch(
            model,
            trajectory_groups,
            {
                "packed_sequence_length": get_megatron_runtime_config().packed_sequence_length
            },
            include_moe_routing=include_moe_routing,
        )
        if batch is None:
            return None
        payload = batch.payload
        distributed = (
            payload.packed if isinstance(payload, _DistributedBatchPayload) else payload
        )
        metrics = {
            "time/step_trajectory_fetch_s": distributed.trajectory_fetch_s,
            "time/step_packing_core_s": distributed.packing_core_s,
            "time/step_trajectory_log_wait_s": distributed.trajectory_log_wait_s,
            "time/step_packed_batch_finalize_s": distributed.packed_batch_finalize_s,
            "time/step_packing_rpc_s": distributed.packing_rpc_s,
            "time/step_packed_batch_fanout_s": distributed.packed_batch_fanout_s,
        }
        prepared = _PipelinePreparedBatch(
            batch=batch,
            groups=tuple(trajectory_groups),
            include_moe_routing=include_moe_routing,
            metrics=metrics,
        )
        for group in trajectory_groups:
            group._prepared_training_batch = prepared
        return metrics

    async def discard_pipeline_batch(
        self, trajectory_groups: list[TrajectoryGroup]
    ) -> None:
        prepared = trajectory_groups[0]._prepared_training_batch
        if not isinstance(prepared, _PipelinePreparedBatch) or any(
            group._prepared_training_batch is not prepared
            for group in trajectory_groups
        ):
            raise RuntimeError("pipeline batch is not prepared")
        for group in trajectory_groups:
            group._prepared_training_batch = None
        paths = {
            group._prepared_log_path
            for group in trajectory_groups
            if group._prepared_log_path is not None
        }
        results = await asyncio.gather(
            self._release_distributed_batch(prepared.batch, disposition="discarded"),
            *(asyncio.to_thread(Path(path).unlink) for path in paths),
            return_exceptions=True,
        )
        for group in trajectory_groups:
            group._prepared_log_path = None
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("prepared batch discard failed", failures)

    async def _stream_prepared_training(
        self,
        model: TrainableModel,
        service: ModelService,
        batch: _PackedTrainingBatch,
        config: Any,
        service_dev_config: Any,
        grad_accumulation_sequences: int,
        verbose: bool,
    ) -> AsyncIterator[dict[str, float]]:
        if self._runtime is None:
            async for result in super()._stream_prepared_training(
                model,
                service,
                batch,
                config,
                service_dev_config,
                grad_accumulation_sequences,
                verbose,
            ):
                yield result
            return
        from ..distributed.art_runtime import DistributedPackedBatch
        from .distributed_service import DistributedMegatronService

        payload = batch.payload
        distributed_batch = cast(
            DistributedPackedBatch,
            payload.packed
            if isinstance(payload, _DistributedBatchPayload)
            else payload,
        )
        if isinstance(payload, _DistributedBatchPayload):
            ref = distributed_batch.leases.ref
            expected_groups = tuple(
                selection.lease.item.ref.result_id for selection in payload.selections
            )
            expected_records = tuple(
                record.record_id
                for selection in payload.selections
                for record in selection.lease.item.ref.records
            )
            versions = []
            for selection in payload.selections:
                item = selection.lease.item
                descriptor = item.ref.descriptor
                versions.extend(
                    version
                    for initial, final in zip(
                        descriptor.trajectory_initial_policy_versions,
                        descriptor.trajectory_final_policy_versions,
                        strict=True,
                    )
                    for version in (
                        initial
                        if initial is not None
                        else item.annotations.initial_policy_version,
                        final
                        if final is not None
                        else item.annotations.final_policy_version,
                    )
                )
            if (
                distributed_batch.packing_generation_id != payload.generation_id
                or ref.group_ids != expected_groups
                or ref.record_ids != expected_records
                or ref.min_source_version != min(versions)
                or ref.max_source_version != max(versions)
            ):
                raise RuntimeError("packed batch policy provenance does not match")
        distributed_service = cast(DistributedMegatronService, service)
        async for result in distributed_service.train_packed(
            distributed_batch, config, service_dev_config
        ):
            yield {**result, **distributed_service.drain_publication_metrics()}

    async def _release_training_batch(self, batch: _PackedTrainingBatch) -> None:
        if self._runtime is None:
            return await super()._release_training_batch(batch)
        await self._release_distributed_batch(batch, disposition="consumed")

    async def _release_distributed_batch(
        self,
        batch: _PackedTrainingBatch,
        *,
        disposition: Literal["consumed", "discarded"],
    ) -> None:
        from ..distributed.art_runtime import DistributedPackedBatch

        assert self._runtime is not None
        payload = batch.payload
        if not isinstance(payload, _DistributedBatchPayload):
            await self._runtime.release_batch(cast(DistributedPackedBatch, payload))
            return
        distributed_batch = cast(DistributedPackedBatch, payload.packed)
        queue = payload.selections[0].queue
        results = await asyncio.gather(
            self._runtime.release_batch(distributed_batch),
            queue.release_selections(
                payload.selections,
                disposition=disposition,
                generation_id=payload.generation_id,
            ),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup(
                "distributed training batch release failed", failures
            )

    async def _delete_checkpoint_files(
        self,
        model: AnyTrainableModel,
        steps_to_keep: list[int],
    ) -> None:
        if self._runtime is None:
            return await super()._delete_checkpoint_files(model, steps_to_keep)
        from ..local.checkpoints import delete_checkpoints
        from .distributed_service import DistributedMegatronService
        from .optimizer_state import optimizer_retention_lease

        service = cast(DistributedMegatronService, await self._get_service(model))
        output_dir = get_model_dir(model=model, art_path=self._path)
        async with service.checkpoint_retention_lease() as active_steps:

            def delete_retained() -> None:
                retained = set(steps_to_keep) | set(active_steps)
                with optimizer_retention_lease(output_dir, retained) as protected:
                    delete_checkpoints(output_dir, sorted(protected))

            await asyncio.to_thread(delete_retained)

    async def _advance_skipped_step(
        self,
        model: TrainableModel,
        service: ModelService,
        current_step: int,
        next_step: int,
    ) -> dict[str, float]:
        if self._runtime is None:
            return await super()._advance_skipped_step(
                model, service, current_step, next_step
            )
        from .distributed_service import DistributedMegatronService

        distributed = cast(DistributedMegatronService, service)
        return await distributed.advance_without_training(
            expected_step=current_step,
            learner_version=next_step,
        )

    async def _get_step(self, model: AnyTrainableModel) -> int:
        if not model.trainable:
            return 0
        if self._runtime is not None and model.name in self._services:
            from .distributed_service import DistributedMegatronService

            service = cast(DistributedMegatronService, self._services[model.name])
            return await service.prepare_for_packing()
        if model.name in self._resume_prepared_models:
            return await super()._get_step(model)
        output_dir = get_model_dir(model=model, art_path=self._path)
        info = prepare_megatron_resume_state(
            output_dir=output_dir,
            optimizer_state_path=f"{output_dir}/optimizer_states_rl",
        )
        print(format_megatron_resume_message(info))
        self._resume_prepared_models.add(model.name)
        return await super()._get_step(model)

    def _default_sft_batch_size(self) -> int:
        import torch

        num_gpus = max(int(torch.cuda.device_count()), 1)
        tensor_parallel_size = min(2, num_gpus)
        return max(num_gpus // tensor_parallel_size, 1)
