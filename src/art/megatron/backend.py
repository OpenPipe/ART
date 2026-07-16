from pathlib import Path
import secrets
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, cast

from mp_actors import move_to_child_process

from .. import dev
from ..backend import AnyTrainableModel
from ..local.backend import LocalBackend, _PackedTrainingBatch
from ..local.service import ModelService
from ..model import TrainableModel
from ..trajectories import TrajectoryGroup
from ..types import LocalTrainResult
from ..utils.output_dirs import get_model_dir
from ..vllm_runtime import get_external_vllm_runtime_config
from .optimizer_state import (
    format_megatron_resume_message,
    prepare_megatron_resume_state,
)
from .runtime_config import get_megatron_runtime_config

if TYPE_CHECKING:
    from ..distributed.art_runtime import ArtRuntime
    from .distributed_service import DistributedMegatronService


class MegatronBackend(LocalBackend):
    def __init__(
        self,
        *,
        in_process: bool = False,
        path: str | None = None,
        enable_expert_replay: bool = True,
        runtime: "ArtRuntime | None" = None,
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
        return await super().train(
            model,
            trajectory_groups,
            packed_sequence_length=get_megatron_runtime_config().packed_sequence_length,
            **kwargs,
        )

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
        if self._runtime is None:
            return await super()._prepare_training_batch(
                model,
                trajectory_groups,
                dev_config,
                include_moe_routing=include_moe_routing,
            )
        from ..distributed.packing import (
            PackingRequest,
            TrajectoryGroupPayload,
        )
        from ..distributed.rollout import RolloutModelSpec

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
        current_step = await self._get_step(model)
        group_ids = tuple(
            f"{group.metadata.get('scenario_id', 'group')}:{index}"
            for index, group in enumerate(trajectory_groups)
        )
        record_ids = tuple(
            f"{group_id}:{trajectory_index}"
            for group_id, group in zip(group_ids, trajectory_groups, strict=True)
            for trajectory_index, _ in enumerate(group.trajectories)
        )
        packed = await self._runtime.pack(
            PackingRequest(
                model=RolloutModelSpec.from_model(model),
                trajectory_groups=tuple(
                    TrajectoryGroupPayload.from_group(group)
                    for group in trajectory_groups
                ),
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
                group_ids=group_ids,
                record_ids=record_ids,
                min_source_version=min(versions, default=current_step),
                max_source_version=max(versions, default=current_step),
            )
        )
        if packed is None:
            return None
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
            payload=packed,
            num_sequences=ref.num_sequences,
            sequence_length=ref.sequence_length,
            trainable_assistant_tokens=packed.trainable_assistant_tokens,
            non_padding_tokens=packed.non_padding_tokens,
            logical_tokens=stats.logical_tokens,
            physical_tokens=stats.physical_tokens,
            include_moe_routing=include_moe_routing,
        )

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

        distributed_batch = cast(DistributedPackedBatch, batch.payload)
        distributed_service = cast(DistributedMegatronService, service)
        training_error: BaseException | None = None
        try:
            async for result in distributed_service.train_packed(
                distributed_batch, config, service_dev_config
            ):
                yield result
        except BaseException as error:
            training_error = error
            raise
        finally:
            try:
                await self._runtime.release_batch(distributed_batch)
            except BaseException as error:
                if training_error is None:
                    raise
                training_error.add_note(f"packed batch release also failed: {error}")

    async def _advance_skipped_step(
        self,
        model: TrainableModel,
        service: ModelService,
        current_step: int,
        next_step: int,
    ) -> None:
        if self._runtime is None:
            await super()._advance_skipped_step(model, service, current_step, next_step)
            return
        from .distributed_service import DistributedMegatronService

        distributed = cast(DistributedMegatronService, service)
        await distributed.advance_without_training(
            expected_step=current_step,
            learner_version=next_step,
        )

    async def _get_step(self, model: AnyTrainableModel) -> int:
        if not model.trainable:
            return 0
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
