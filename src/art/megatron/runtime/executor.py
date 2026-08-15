from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import math
from pathlib import Path
from threading import BoundedSemaphore, Event, Lock
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from art.distributed.object_store import (
    BinaryObjectTarget,
    S3BinaryObjectStore,
)
from art.megatron.model_support.lora_disk import (
    ART_LORA_FORMAT_CONFIG_KEY,
    ART_LORA_FORMAT_VLLM,
    encode_adapter_config,
)
from art.megatron.optimizer_state import (
    CheckpointFile,
    OptimizerAdapter,
    OptimizerShard,
    OptimizerTopology,
)
from art.utils.safetensors import PreparedSafetensors, SafetensorsLayout

from ..tensor_snapshot import PinnedCpuSnapshotStager
from ..training.gradient_accumulator import GradientAccumulator
from .data_plane import InMemoryPackedBatch, SFTBatchData, validate_packed_batch
from .publication import (
    TrainerPublicationFailed,
    TrainerPublicationSucceeded,
    TrainerRankPublication,
)
from .specs import (
    ForwardBackwardJobSpec,
    ForwardJobSpec,
    GenerationSnapshotJobSpec,
    LoadStateJobSpec,
    OptimizerJobSpec,
    SftForwardBackwardJobSpec,
    SftForwardJobSpec,
    TrainerGeneration,
    TrainerJobSpec,
    TrainJobSpec,
)
from .trainer_run import EventSink

if TYPE_CHECKING:
    from art.megatron.lora import LoRASlotRef
    from art.trainer_rank import TrainerRankOptimizerState


class MegatronTrainJobExecutor:
    """Thin adapter around the warm runtime's in-memory job entrypoint."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._publisher = _GenerationPublisher(
            runtime,
            stager=PinnedCpuSnapshotStager(),
            capacity=int(runtime.snapshot_pool_capacity),
        )
        self._gradients = GradientAccumulator(model_chunks=runtime.model)
        self._gradient_parent_version: int | None = None
        self._closed = False

    def execute(
        self,
        job: TrainJobSpec,
        batch: InMemoryPackedBatch,
        sink: EventSink,
        cancelled: Event,
    ) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._require_no_open_gradients()
        validate_packed_batch(batch)
        self._publisher.raise_if_failed()
        from art.megatron.train import execute_megatron_rl_job

        metrics = execute_megatron_rl_job(
            self.runtime,
            job,
            batch.tensors,
            progress_sink=lambda step_index, num_steps, metrics: sink.progress(
                step_index=step_index,
                num_steps=num_steps,
                metrics=metrics,
            ),
            adapter_ready_sink=lambda: sink.adapter_ready(
                learner_version=job.learner_version,
                adapter_path=job.output_adapter_path,
            ),
            snapshot_sink=lambda job, adapter_dtypes, adapter_config, save_optimizer: (
                self._publisher.stage_and_submit(
                    run_id=job.run_id,
                    generation=job.output.generation,
                    optimizer_state_path=job.output.optimizer_state_path,
                    staging_adapter_path=job.output.staging_adapter_path,
                    publication_targets=job.publication_targets,
                    adapter_dtypes=adapter_dtypes,
                    adapter_config=adapter_config,
                    save_optimizer=save_optimizer,
                    sink=sink,
                )
            ),
            cancelled=cancelled,
        )
        if job.merged_weight_transfer is not None:
            started = time.perf_counter()
            self._sync_merged(job.merged_weight_transfer)
            metrics["time/merged_weight_publish_s"] = time.perf_counter() - started
        return metrics

    def execute_forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        validate_packed_batch(batch)
        self._publisher.raise_if_failed()
        if self._gradient_parent_version not in {
            None,
            job.expected_learner_version,
        }:
            raise RuntimeError(
                "F/B parent does not match the open gradient accumulator"
            )
        from art.megatron.train import execute_megatron_rl_forward_backward_job

        result = execute_megatron_rl_forward_backward_job(
            self.runtime,
            job,
            batch.tensors,
            gradient_accumulator=self._gradients,
            cancelled=cancelled,
        )
        self._gradient_parent_version = job.expected_learner_version
        step = result.result
        return {
            "operation_id": job.operation_id,
            "metrics": result.metrics(),
            "token_count": int(result.token_count.item()),
            "token_logprobs": tuple(
                tuple(float(item) for item in values.flatten().tolist())
                for values in step.new_logprobs
            ),
        }

    def execute_forward(
        self,
        job: ForwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        validate_packed_batch(batch)
        self._publisher.raise_if_failed()
        from art.megatron.train import (
            _prepare_rl_training_state,
            execute_megatron_rl_forward_job,
        )

        _prepare_rl_training_state(self.runtime, job)
        result = execute_megatron_rl_forward_job(
            self.runtime,
            job,
            batch.tensors,
            cancelled=cancelled,
        )
        return {
            **result,
            "token_logprobs": tuple(
                tuple(float(item) for item in values.flatten().tolist())
                for values in result["token_logprobs"]
            ),
        }

    def execute_sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._publisher.raise_if_failed()
        if self._gradient_parent_version not in {
            None,
            job.expected_learner_version,
        }:
            raise RuntimeError(
                "SFT F/B parent does not match the open gradient accumulator"
            )
        from art.megatron.train import execute_megatron_sft_forward_backward_job

        result = execute_megatron_sft_forward_backward_job(
            self.runtime,
            job,
            batch,
            gradient_accumulator=self._gradients,
            cancelled=cancelled,
        )
        self._gradient_parent_version = job.expected_learner_version
        return result

    def execute_sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._publisher.raise_if_failed()
        from art.megatron.train import execute_megatron_sft_forward_job

        return execute_megatron_sft_forward_job(
            self.runtime,
            job,
            batch,
            cancelled=cancelled,
        )

    def execute_optimizer(self, job: OptimizerJobSpec) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        if self._gradient_parent_version != job.expected_learner_version:
            raise RuntimeError("optimizer parent does not match accumulated gradients")
        runtime = self.runtime
        if runtime.optimizer is None:
            raise RuntimeError("trainer has no resident optimizer")
        self._gradients.seal(job.contributing_forward_backward_operation_ids)
        self._gradients.prepare_optimizer()
        for group in runtime.optimizer.param_groups:
            group["betas"] = (job.optimizer.beta1, job.optimizer.beta2)
            group["eps"] = job.optimizer.eps
            group["weight_decay"] = job.optimizer.weight_decay
        runtime.optimizer.config.adam_beta1 = job.optimizer.beta1
        runtime.optimizer.config.adam_beta2 = job.optimizer.beta2
        runtime.optimizer.config.adam_eps = job.optimizer.eps
        runtime.optimizer.config.weight_decay = job.optimizer.weight_decay
        from art.megatron.train import run_megatron_optimizer_step

        started = time.perf_counter()
        result = run_megatron_optimizer_step(
            optimizer=runtime.optimizer,
            learning_rate=job.optimizer.learning_rate,
            model_support_handler=runtime.model_support_handler,
            model_chunks=runtime.model,
            before_step=runtime.optimizer_snapshot_barrier.wait_before_mutation,
        )
        if not result.update_successful or not math.isfinite(result.grad_norm):
            raise RuntimeError(
                "Megatron optimizer rejected the update: "
                f"update_successful={result.update_successful}, "
                f"grad_norm={result.grad_norm}"
            )
        optimizer_step_s = time.perf_counter() - started
        consumed = self._gradients.consume()
        if consumed != job.contributing_forward_backward_operation_ids:
            raise RuntimeError("optimizer consumed the wrong gradient contributions")
        runtime.resident_training_session_id = job.training_session_id
        runtime.resident_policy_step = job.learner_version
        runtime.optimizer_state_loaded = True
        self._gradient_parent_version = None
        if (
            runtime.adapter_export_dtypes is None
            or runtime.adapter_export_config is None
        ):
            raise RuntimeError("optimizer output has no adapter export layout")
        snapshot_metrics = self._publisher.stage(
            run_id=job.run_id,
            generation=job.generation,
            adapter_dtypes=runtime.adapter_export_dtypes,
            adapter_config=runtime.adapter_export_config,
        )
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "contributing_forward_backward_operation_ids": consumed,
            "metrics": {
                "loss/learning_rate": job.optimizer.learning_rate,
                "loss/grad_norm": float(result.grad_norm),
                "optimizer/update_successful": float(result.update_successful),
                "optimizer/num_zeros_in_grad": float(result.num_zeros_in_grad or 0),
                "time/optimizer_step_s": optimizer_step_s,
                **snapshot_metrics,
            },
        }

    def execute_load_state(self, job: LoadStateJobSpec) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._require_no_open_gradients()
        from art.megatron.train import execute_megatron_load_state_job

        execute_megatron_load_state_job(self.runtime, job)
        self._gradient_parent_version = None
        runtime = self.runtime
        if (
            runtime.adapter_export_dtypes is None
            or runtime.adapter_export_config is None
        ):
            raise RuntimeError("loaded state has no adapter export layout")
        snapshot_metrics = self._publisher.stage(
            run_id=job.run_id,
            generation=job.generation,
            adapter_dtypes=runtime.adapter_export_dtypes,
            adapter_config=runtime.adapter_export_config,
        )
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "optimizer_restored": job.restore_optimizer,
            "metrics": snapshot_metrics,
        }

    def execute_snapshot(
        self,
        job: GenerationSnapshotJobSpec,
        sink: EventSink,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._publisher.raise_if_failed()
        runtime = self.runtime
        if (
            runtime.resident_training_session_id != job.training_session_id
            or runtime.resident_policy_step != job.learner_version
            or runtime.adapter_export_dtypes is None
            or runtime.adapter_export_config is None
        ):
            raise RuntimeError("resident trainer state does not match snapshot learner")
        if job.save_optimizer and (
            not runtime.optimizer_state_loaded or runtime.optimizer is None
        ):
            raise RuntimeError("snapshot requested non-resident optimizer state")
        stage_metrics = {}
        if not self._publisher.has_generation(job.generation):
            stage_metrics = self._publisher.stage(
                run_id=job.run_id,
                generation=job.generation,
                adapter_dtypes=runtime.adapter_export_dtypes,
                adapter_config=runtime.adapter_export_config,
            )
        metrics = {
            **stage_metrics,
            **self._publisher.submit(
                generation=job.generation,
                optimizer_state_path=job.optimizer_state_path,
                staging_adapter_path=job.staging_adapter_path,
                existing_adapter=job.existing_adapter,
                publication_targets=job.publication_targets,
                adapter_object_target=job.adapter_object_target,
                save_optimizer=job.save_optimizer,
                sink=sink,
            ),
        }
        if job.merged_weight_transfer is not None:
            started = time.perf_counter()
            self._sync_merged(job.merged_weight_transfer)
            metrics["time/merged_weight_publish_s"] = time.perf_counter() - started
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "metrics": metrics,
        }

    def discard_open_gradients(self) -> None:
        self._gradients.discard()
        self._gradient_parent_version = None

    def sync_merged_source(
        self,
        generation: TrainerGeneration,
        transfer: Any,
    ) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._require_no_open_gradients()
        runtime = self.runtime
        if (
            runtime.resident_training_session_id != generation.training_session_id
            or runtime.resident_policy_step != generation.policy_step
        ):
            from art.megatron.model_support.lora_disk import load_adapter_config
            from art.megatron.train import _load_adapter_into_model

            _load_adapter_into_model(
                runtime.model,
                generation.adapter_path,
                runtime.rank,
                handler=runtime.model_support_handler,
            )
            runtime.adapter_export_config = load_adapter_config(generation.adapter_path)
            runtime.adapter_export_dtypes = {}
            runtime.resident_training_session_id = None
            runtime.resident_policy_step = None
            runtime.optimizer_state_loaded = False
        started = time.perf_counter()
        self._sync_merged(transfer)
        return {"time/merged_weight_publish_s": time.perf_counter() - started}

    def _sync_merged(self, transfer: Any) -> None:
        from art.megatron.weights.merged_weight_export import (
            sync_merged_weights_to_vllm,
        )

        runtime = self.runtime
        if runtime.adapter_export_config is None:
            raise RuntimeError("merged publication has no adapter export config")
        (
            runtime.merged_weight_transfer_group,
            runtime.merged_weight_transfer_init_info,
        ) = sync_merged_weights_to_vllm(
            bridge=runtime.bridge,
            model=runtime.model,
            model_support_handler=runtime.model_support_handler,
            adapter_model={},
            adapter_config=runtime.adapter_export_config,
            rank=runtime.rank,
            world_size=runtime.world_size,
            merged_weight_transfer_group=runtime.merged_weight_transfer_group,
            merged_weight_transfer_init_info=(runtime.merged_weight_transfer_init_info),
            spec=transfer,
            pause_generation=True,
        )

    def advance_without_training(
        self,
        *,
        training_session_id: str,
        expected_learner_version: int,
        learner_version: int,
        optimizer_state_path: str,
        adapter: "OptimizerAdapter | None",
    ) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("Megatron executor is closed")
        self._require_no_open_gradients()
        if learner_version != expected_learner_version + 1:
            raise ValueError("a no-op learner transition must advance exactly one step")
        runtime = self.runtime
        if (
            runtime.resident_training_session_id != training_session_id
            or runtime.resident_policy_step != expected_learner_version
            or not runtime.optimizer_state_loaded
            or runtime.optimizer is None
        ):
            raise RuntimeError("resident trainer state does not match no-op transition")
        del optimizer_state_path, adapter
        runtime.resident_policy_step = learner_version
        return {}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        try:
            self._gradients.discard()
        except BaseException as error:
            failures.append(error)
        try:
            self._publisher.close()
            self.runtime.optimizer_snapshot_barrier.synchronize()
        except BaseException as error:
            failures.append(error)
        controller = getattr(self.runtime, "moe_routing_replay_controller", None)
        if controller is not None:
            try:
                controller.remove_router_patches()
            except BaseException as error:
                failures.append(error)
            finally:
                self.runtime.moe_routing_replay_controller = None
        from art.megatron.train import _close_merged_weight_transfer_group

        try:
            _close_merged_weight_transfer_group(self.runtime)
        except BaseException as error:
            failures.append(error)
        try:
            import torch

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except BaseException as error:
            failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Megatron executor close failed", failures)

    def _require_no_open_gradients(self) -> None:
        if self._gradients.contribution_ids:
            raise RuntimeError("operation cannot discard open gradient contributions")


class _ResidentRunState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    training_session_id: str
    learner_version: int
    adapter_config: dict[str, Any]
    gradients: Any


class MCoreRunSlotExecutor:
    """Train independent exact-shape LoRAs on one warm MCore runtime."""

    def __init__(self, runtime: Any) -> None:
        from art.trainer_rank import TrainerRank

        self.runtime = runtime
        self._slot_trainer = TrainerRank(runtime)
        self._publisher = _GenerationPublisher(
            runtime,
            stager=PinnedCpuSnapshotStager(),
            capacity=int(runtime.snapshot_pool_capacity),
        )
        self._runs: dict[str, _ResidentRunState] = {}
        self._closed = False

    def register_run(
        self,
        *,
        run_id: str,
        training_session_id: str,
        learner_version: int,
        adapter_path: str,
    ) -> None:
        if self._closed:
            raise RuntimeError("Megatron run slot is closed")
        if run_id in self._runs:
            raise RuntimeError(f"training run is already resident: {run_id!r}")
        from art.megatron.model_support.lora_disk import (
            load_adapter_config,
            load_lora_tensors_for_megatron,
        )
        from art.megatron.training.gradient_accumulator import (
            ParameterGradientAccumulator,
        )

        adapter_config = load_adapter_config(adapter_path)
        adapter_model = load_lora_tensors_for_megatron(
            adapter_path, handler=self.runtime.model_support_handler
        )
        self._slot_trainer.load_checkpoint_slot(
            run_id,
            adapter_model,
            adapter_config=adapter_config,
        )
        self._runs[run_id] = _ResidentRunState(
            run_id=run_id,
            training_session_id=training_session_id,
            learner_version=learner_version,
            adapter_config=adapter_config,
            gradients=ParameterGradientAccumulator(
                parameters=self._slot_trainer.checkpoint_slot_parameters(run_id)
            ),
        )

    def optimizer_layout(self, run_id: str) -> Any:
        self._require_run(run_id)
        return self._slot_trainer.checkpoint_slot_optimizer_layout(run_id)

    def restore_optimizer_state(
        self, run_id: str, state: "TrainerRankOptimizerState"
    ) -> None:
        self._require_run(run_id)
        self._slot_trainer.restore_checkpoint_slot_optimizer_state(run_id, state)

    def execute_forward_backward(
        self,
        job: ForwardBackwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
    ) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        validate_packed_batch(batch)
        from art.megatron.train import (
            execute_megatron_dynamic_lora_forward_backward_job,
        )

        result = execute_megatron_dynamic_lora_forward_backward_job(
            self.runtime,
            job,
            batch.tensors,
            slot_trainer=self._slot_trainer,
            gradient_accumulator=state.gradients,
            cancelled=cancelled,
        )
        step = result.result
        return {
            "operation_id": job.operation_id,
            "metrics": result.metrics(),
            "token_count": int(result.token_count.item()),
            "token_logprobs": tuple(
                tuple(float(item) for item in values.flatten().tolist())
                for values in step.new_logprobs
            ),
        }

    def execute_forward(
        self,
        job: ForwardJobSpec,
        batch: InMemoryPackedBatch,
        cancelled: Event,
    ) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        validate_packed_batch(batch)
        self._publisher.raise_if_failed()
        from art.megatron.train import execute_megatron_dynamic_lora_forward_job

        result = execute_megatron_dynamic_lora_forward_job(
            self.runtime,
            job,
            batch.tensors,
            cancelled=cancelled,
        )
        return {
            **result,
            "token_logprobs": tuple(
                tuple(float(item) for item in values.flatten().tolist())
                for values in result["token_logprobs"]
            ),
        }

    def execute_sft_forward_backward(
        self,
        job: SftForwardBackwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        from art.megatron.train import (
            execute_megatron_dynamic_lora_sft_forward_backward_job,
        )

        return execute_megatron_dynamic_lora_sft_forward_backward_job(
            self.runtime,
            job,
            batch,
            slot_trainer=self._slot_trainer,
            gradient_accumulator=state.gradients,
            cancelled=cancelled,
        )

    def execute_sft_forward(
        self,
        job: SftForwardJobSpec,
        batch: SFTBatchData,
        cancelled: Event,
    ) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        self._publisher.raise_if_failed()
        from art.megatron.train import execute_megatron_dynamic_lora_sft_forward_job

        return execute_megatron_dynamic_lora_sft_forward_job(
            self.runtime,
            job,
            batch,
            cancelled=cancelled,
        )

    def execute_optimizer(self, job: OptimizerJobSpec) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        state.gradients.seal(job.contributing_forward_backward_operation_ids)
        gradients = state.gradients.prepare_optimizer()
        from art.trainer_rank import AdamParams

        self.runtime.optimizer_snapshot_barrier.wait_before_mutation()
        started = time.perf_counter()
        result = self._slot_trainer.optim_step_reduced(
            job.run_id,
            params=AdamParams(
                learning_rate=job.optimizer.learning_rate,
                beta1=job.optimizer.beta1,
                beta2=job.optimizer.beta2,
                eps=job.optimizer.eps,
                weight_decay=job.optimizer.weight_decay,
                grad_clip_norm=float(self.runtime.optimizer_config.clip_grad or 0.0),
            ),
            grads=gradients,
        )
        if not result["update_successful"] or not math.isfinite(result["grad_norm"]):
            raise RuntimeError("dynamic LoRA optimizer rejected the update")
        optimizer_step_s = time.perf_counter() - started
        consumed = state.gradients.consume()
        if consumed != job.contributing_forward_backward_operation_ids:
            raise RuntimeError("optimizer consumed the wrong gradient contributions")
        state.learner_version = job.learner_version
        from art.megatron.lora import LoRASlotRef

        snapshot_metrics = self._publisher.stage(
            run_id=job.run_id,
            generation=job.generation,
            adapter_dtypes={},
            adapter_config=state.adapter_config,
            slot_ref=LoRASlotRef("checkpoint", job.run_id),
            trainer_rank_optimizer_state=(
                self._slot_trainer.checkpoint_slot_optimizer_snapshot_sources(
                    job.run_id
                )
            ),
        )
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "contributing_forward_backward_operation_ids": consumed,
            "metrics": {
                "loss/learning_rate": job.optimizer.learning_rate,
                "loss/grad_norm": result["grad_norm"],
                "optimizer/update_successful": result["update_successful"],
                "optimizer/num_zeros_in_grad": result["num_zeros_in_grad"],
                "time/optimizer_step_s": optimizer_step_s,
                **snapshot_metrics,
            },
        }

    def optimizer_state(self, run_id: str) -> "TrainerRankOptimizerState | None":
        self._require_run(run_id)
        return self._slot_trainer.checkpoint_slot_optimizer_state(run_id)

    def execute_load_state(self, job: LoadStateJobSpec) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(
            state, job.training_session_id, job.expected_learner_version
        )
        if state.gradients.contribution_ids:
            raise RuntimeError("load_state cannot discard open gradient contributions")
        self.runtime.optimizer_snapshot_barrier.synchronize()
        from art.megatron.model_support.lora_disk import (
            load_adapter_config,
            load_lora_tensors_for_megatron,
        )
        from art.megatron.optimizer_state import load_trainer_rank_optimizer_state
        from art.megatron.training.gradient_accumulator import (
            ParameterGradientAccumulator,
        )

        config = load_adapter_config(job.adapter_path)
        self._validate_adapter_layout(state.adapter_config, config)
        optimizer_state = None
        if job.optimizer_state_path is not None:
            assert job.optimizer_generation_id is not None
            optimizer_state = load_trainer_rank_optimizer_state(
                self.runtime,
                optimizer_state_path=job.optimizer_state_path,
                adapter_path=job.adapter_path,
                adapter_step=job.adapter_step,
                optimizer_generation_id=job.optimizer_generation_id,
                layout=self._slot_trainer.checkpoint_slot_optimizer_layout(job.run_id),
            )
        adapter_model = load_lora_tensors_for_megatron(
            job.adapter_path, handler=self.runtime.model_support_handler
        )
        self._slot_trainer.load_checkpoint_slot(
            job.run_id,
            adapter_model,
            optimizer_state=optimizer_state,
            adapter_config=config,
        )
        state.learner_version = job.learner_version
        state.adapter_config = config
        state.gradients = ParameterGradientAccumulator(
            parameters=self._slot_trainer.checkpoint_slot_parameters(job.run_id)
        )
        from art.megatron.lora import LoRASlotRef

        snapshot_metrics = self._publisher.stage(
            run_id=job.run_id,
            generation=job.generation,
            adapter_dtypes={},
            adapter_config=state.adapter_config,
            slot_ref=LoRASlotRef("checkpoint", job.run_id),
            trainer_rank_optimizer_state=(
                self._slot_trainer.checkpoint_slot_optimizer_snapshot_sources(
                    job.run_id
                )
            ),
            snapshot_optimizer=job.restore_optimizer,
        )
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "optimizer_restored": job.restore_optimizer,
            "metrics": snapshot_metrics,
        }

    def execute_snapshot(
        self,
        job: GenerationSnapshotJobSpec,
        sink: EventSink,
    ) -> dict[str, Any]:
        state = self._require_run(job.run_id)
        self._validate_parent(state, job.training_session_id, job.learner_version)
        if job.merged_weight_transfer is not None:
            raise ValueError("multi-run slots publish LoRA weights, not merged weights")
        stage_metrics = {}
        if not self._publisher.has_generation(job.generation):
            from art.megatron.lora import LoRASlotRef

            stage_metrics = self._publisher.stage(
                run_id=job.run_id,
                generation=job.generation,
                adapter_dtypes={},
                adapter_config=state.adapter_config,
                slot_ref=LoRASlotRef("checkpoint", job.run_id),
                trainer_rank_optimizer_state=(
                    self._slot_trainer.checkpoint_slot_optimizer_snapshot_sources(
                        job.run_id
                    )
                ),
                snapshot_optimizer=job.save_optimizer,
            )
        metrics = {
            **stage_metrics,
            **self._publisher.submit(
                generation=job.generation,
                optimizer_state_path=job.optimizer_state_path,
                staging_adapter_path=job.staging_adapter_path,
                existing_adapter=job.existing_adapter,
                publication_targets=job.publication_targets,
                adapter_object_target=job.adapter_object_target,
                save_optimizer=job.save_optimizer,
                sink=sink,
            ),
        }
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "metrics": metrics,
        }

    def discard_run_gradients(self, run_id: str) -> None:
        self._require_run(run_id).gradients.discard()

    def unregister_run(self, run_id: str) -> None:
        state = self._require_run(run_id)
        state.gradients.discard()
        self._slot_trainer.unload_checkpoint_slot(run_id)
        self._runs.pop(run_id)
        self._publisher.retire_run(run_id)

    def close(self) -> None:
        if self._closed:
            return
        for state in self._runs.values():
            state.gradients.discard()
        self._publisher.close()
        self._closed = True

    def _require_run(self, run_id: str) -> _ResidentRunState:
        if self._closed:
            raise RuntimeError("Megatron run slot is closed")
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise RuntimeError(f"training run is not resident: {run_id!r}") from exc

    @staticmethod
    def _validate_parent(
        state: _ResidentRunState,
        training_session_id: str,
        learner_version: int,
    ) -> None:
        if (
            state.training_session_id != training_session_id
            or state.learner_version != learner_version
        ):
            raise RuntimeError("resident run state does not match command parent")

    @staticmethod
    def _validate_adapter_layout(
        current: dict[str, Any], replacement: dict[str, Any]
    ) -> None:
        for key in ("base_model_name_or_path", "r", "lora_alpha"):
            if replacement.get(key) != current.get(key):
                raise ValueError(f"loaded adapter changes immutable {key}")
        if set(replacement.get("target_modules", ())) != set(
            current.get("target_modules", ())
        ):
            raise ValueError("loaded adapter changes immutable target_modules")


class _ResolvedGeneration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    lora: Any | None
    optimizer: Any | None
    prepared_tensors: PreparedSafetensors | None


class _CachedGeneration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    generation: TrainerGeneration
    resolved: Future[_ResolvedGeneration]
    consumers: list[Future[TrainerRankPublication]] = Field(default_factory=list)
    object_ids: set[str] = Field(default_factory=set)
    retired: bool = False
    released: bool = False


class _SnapshotTransport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter: OptimizerAdapter | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class _RankSnapshotPersistence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generation: TrainerGeneration
    rank: int = Field(ge=0)
    adapter: OptimizerAdapter | None = None
    shard: OptimizerShard | None = None
    runtime_sha256: str | None = None
    topology: OptimizerTopology | None = None
    saves_optimizer: bool
    metrics: dict[str, float] = Field(default_factory=dict)


class _GenerationPublisher:
    def __init__(
        self,
        runtime: Any,
        *,
        stager: PinnedCpuSnapshotStager,
        capacity: int,
    ) -> None:
        if capacity < 1:
            raise ValueError("snapshot pool capacity must be positive")
        self.runtime = runtime
        self.stager = stager
        self.capacity = capacity
        self._slots = BoundedSemaphore(capacity)
        self._lock = Lock()
        self._resolution_pool = ThreadPoolExecutor(
            max_workers=capacity, thread_name_prefix="art-snapshot-resolve"
        )
        self._transport_pool = ThreadPoolExecutor(
            max_workers=capacity, thread_name_prefix="art-publish-transport"
        )
        self._durability_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="art-publish-durable"
        )
        self._completion_pool = ThreadPoolExecutor(
            max_workers=capacity, thread_name_prefix="art-publish-complete"
        )
        self._transport_sender: Any | None = None
        self._transport_sender_lock = Lock()
        self._object_store: S3BinaryObjectStore | None = None
        self._object_publications: dict[str, Future[OptimizerAdapter]] = {}
        self._lora_layouts: dict[tuple[Any, ...], SafetensorsLayout] = {}
        self._cache: dict[str, _CachedGeneration] = {}
        self._latest_by_run: dict[str, str] = {}
        self._failures: list[BaseException] = []
        self._in_flight = 0

    def stage(
        self,
        *,
        run_id: str,
        generation: TrainerGeneration,
        adapter_dtypes: dict[str, Any],
        adapter_config: dict[str, Any],
        slot_ref: "LoRASlotRef | None" = None,
        trainer_rank_optimizer_state: "TrainerRankOptimizerState | None" = None,
        snapshot_optimizer: bool = True,
    ) -> dict[str, float]:
        from art.megatron.optimizer_state import (
            stage_optimizer_state_snapshot,
            stage_trainer_rank_optimizer_state_snapshot,
        )
        from art.megatron.weights.lora_publish import (
            stage_vllm_lora_snapshot_from_model,
        )

        self._retire_previous(run_id)
        wait_s, in_flight = self._acquire_slot()
        prepare_started = time.perf_counter()
        try:
            lora = stage_vllm_lora_snapshot_from_model(
                model=self.runtime.model,
                adapter_dtypes=adapter_dtypes,
                handler=self.runtime.model_support_handler,
                adapter_config=adapter_config,
                rank=self.runtime.rank,
                world_size=self.runtime.world_size,
                stager=self.stager,
                slot_ref=slot_ref,
            )
            lora_launch_s = time.perf_counter() - prepare_started
            optimizer_started = time.perf_counter()
            if snapshot_optimizer:
                if slot_ref is not None:
                    if trainer_rank_optimizer_state is None:
                        raise RuntimeError(
                            "dynamic LoRA snapshot has no optimizer state"
                        )
                    optimizer = stage_trainer_rank_optimizer_state_snapshot(
                        self.runtime,
                        trainer_rank_optimizer_state,
                        generation_id=generation.generation_id,
                        step=generation.policy_step,
                        stager=self.stager,
                    )
                else:
                    optimizer = stage_optimizer_state_snapshot(
                        self.runtime,
                        generation_id=generation.generation_id,
                        step=generation.policy_step,
                        stager=self.stager,
                    )
            else:
                optimizer = None
            if lora is not None:
                self.runtime.optimizer_snapshot_barrier.register(lora)
            if optimizer is not None:
                self.runtime.optimizer_snapshot_barrier.register(optimizer)
            optimizer_launch_s = time.perf_counter() - optimizer_started
            resolved = self._resolution_pool.submit(
                self._resolve_generation, lora, optimizer
            )
            entry = _CachedGeneration(
                run_id=run_id,
                generation=generation,
                resolved=resolved,
            )
            with self._lock:
                if generation.generation_id in self._cache:
                    raise RuntimeError(
                        f"generation snapshot already exists: {generation.generation_id}"
                    )
                self._cache[generation.generation_id] = entry
                self._latest_by_run[run_id] = generation.generation_id
            resolved.add_done_callback(
                lambda done, cached=entry: self._snapshot_resolved(cached, done)
            )
        except BaseException as error:
            self._release_slot()
            raise
        return {
            "snapshot_pool_wait_s": wait_s,
            "snapshot_pool_in_use": float(in_flight),
            "snapshot_pool_pressure": in_flight / self.capacity,
            "snapshot_lora_launch_s": lora_launch_s,
            "snapshot_optimizer_launch_s": optimizer_launch_s,
            "snapshot_launch_s": time.perf_counter() - prepare_started,
        }

    def submit(
        self,
        *,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        staging_adapter_path: str | None,
        existing_adapter: OptimizerAdapter | None = None,
        publication_targets: tuple[Any, ...],
        adapter_object_target: BinaryObjectTarget | None,
        save_optimizer: bool,
        sink: EventSink,
    ) -> dict[str, float]:
        self.raise_if_failed()
        with self._lock:
            entry = self._cache.get(generation.generation_id)
        if entry is None or entry.retired or entry.generation != generation:
            raise RuntimeError(
                f"learner generation is not staged: {generation.generation_id}"
            )
        started = time.perf_counter()
        transport = self._transport_pool.submit(
            self._transfer_cached_snapshot,
            entry,
            publication_targets,
            adapter_object_target,
            generation,
        )
        durability = self._durability_pool.submit(
            self._persist_cached_snapshot,
            entry.resolved,
            generation,
            optimizer_state_path,
            staging_adapter_path,
            existing_adapter if int(self.runtime.rank) == 0 else None,
            save_optimizer,
        )
        persistence = self._completion_pool.submit(
            self._complete_publication, transport, durability
        )
        with self._lock:
            entry.consumers.append(persistence)
        persistence.add_done_callback(
            lambda done: self._completed(
                done,
                entry=entry,
                sink=sink,
                generation=generation,
            )
        )
        return {
            "snapshot_attach_s": time.perf_counter() - started,
        }

    def stage_and_submit(
        self,
        *,
        run_id: str,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        staging_adapter_path: str,
        publication_targets: tuple[Any, ...],
        adapter_dtypes: dict[str, Any],
        adapter_config: dict[str, Any],
        save_optimizer: bool,
        sink: EventSink,
    ) -> dict[str, float]:
        return {
            **self.stage(
                run_id=run_id,
                generation=generation,
                adapter_dtypes=adapter_dtypes,
                adapter_config=adapter_config,
                snapshot_optimizer=save_optimizer,
            ),
            **self.submit(
                generation=generation,
                optimizer_state_path=optimizer_state_path,
                staging_adapter_path=staging_adapter_path,
                publication_targets=publication_targets,
                adapter_object_target=None,
                save_optimizer=save_optimizer,
                sink=sink,
            ),
        }

    def has_generation(self, generation: TrainerGeneration) -> bool:
        with self._lock:
            entry = self._cache.get(generation.generation_id)
            return (
                entry is not None
                and not entry.retired
                and entry.generation == generation
            )

    def retire_run(self, run_id: str) -> None:
        self._retire_previous(run_id)

    def _acquire_slot(self) -> tuple[float, int]:
        self.raise_if_failed()
        started = time.perf_counter()
        if not self._slots.acquire(blocking=False):
            self._evict_for_capacity()
            self._slots.acquire()
        wait_s = time.perf_counter() - started
        with self._lock:
            self._in_flight += 1
            return wait_s, self._in_flight

    def _evict_for_capacity(self) -> None:
        with self._lock:
            entries = tuple(
                entry for entry in self._cache.values() if not entry.released
            )
            ready = tuple(
                entry
                for entry in entries
                if entry.resolved.done()
                and all(consumer.done() for consumer in entry.consumers)
            )
            if not entries:
                raise RuntimeError("snapshot pool is full without a cached generation")
            entry = (ready or entries)[0]
            entry.retired = True
            consumers = tuple(entry.consumers)
        try:
            entry.resolved.result()
            for consumer in consumers:
                consumer.result()
        finally:
            self._maybe_release(entry)

    def _retire_previous(self, run_id: str) -> None:
        with self._lock:
            generation_id = self._latest_by_run.pop(run_id, None)
            entry = None if generation_id is None else self._cache[generation_id]
            if entry is not None:
                entry.retired = True
        if entry is not None:
            self._maybe_release(entry)

    def _snapshot_resolved(
        self,
        entry: _CachedGeneration,
        resolved: Future[_ResolvedGeneration],
    ) -> None:
        if not resolved.cancelled() and (error := resolved.exception()) is not None:
            with self._lock:
                self._failures.append(error)
                entry.retired = True
        self._maybe_release(entry)

    def _maybe_release(self, entry: _CachedGeneration) -> None:
        with self._lock:
            if (
                not entry.retired
                or entry.released
                or not entry.resolved.done()
                or any(not consumer.done() for consumer in entry.consumers)
            ):
                return
            entry.released = True
            self._cache.pop(entry.generation.generation_id, None)
            if self._latest_by_run.get(entry.run_id) == entry.generation.generation_id:
                self._latest_by_run.pop(entry.run_id)
            for object_id in entry.object_ids:
                self._object_publications.pop(object_id, None)
            self._in_flight -= 1
        self._slots.release()

    def _resolve_generation(self, lora: Any, optimizer: Any) -> _ResolvedGeneration:
        lora = None if lora is None else lora.resolve()
        optimizer = None if optimizer is None else optimizer.resolve()
        prepared_tensors = None
        if lora is not None:
            layout_key = tuple(
                (key, tuple(tensor.shape), str(tensor.dtype))
                for key, tensor in sorted(lora.tensors.items())
            )
            with self._lock:
                layout = self._lora_layouts.get(layout_key)
                if layout is None:
                    layout = self._lora_layouts[layout_key] = SafetensorsLayout(
                        lora.tensors
                    )
            prepared_tensors = layout.bind(lora.tensors)
        return _ResolvedGeneration(
            lora=lora,
            optimizer=optimizer,
            prepared_tensors=prepared_tensors,
        )

    def _transfer_cached_snapshot(
        self,
        entry: _CachedGeneration,
        publication_targets: tuple[Any, ...],
        object_target: BinaryObjectTarget | None,
        generation: TrainerGeneration,
    ) -> _SnapshotTransport:
        started = time.perf_counter()
        if int(self.runtime.rank) != 0:
            return _SnapshotTransport()
        snapshot = entry.resolved.result()
        ready = time.perf_counter()
        if snapshot.lora is None or snapshot.prepared_tensors is None:
            raise RuntimeError("rank zero has no LoRA snapshot to transfer")
        adapter = None
        if object_target is not None:
            adapter = self._publish_lora_object_once(
                entry,
                object_target,
                generation,
                snapshot.lora,
                snapshot.prepared_tensors,
            )
        elif publication_targets:
            self._transfer_lora_snapshot(
                snapshot.lora,
                publication_targets,
                prepared_tensors=snapshot.prepared_tensors,
            )
        return _SnapshotTransport(
            adapter=adapter,
            metrics={
                "time/snapshot_transport_wait_s": ready - started,
                "time/snapshot_transport_s": time.perf_counter() - ready,
            },
        )

    def _persist_cached_snapshot(
        self,
        resolved: Future[_ResolvedGeneration],
        generation: TrainerGeneration,
        optimizer_state_path: str,
        staging_adapter_path: str | None,
        adapter: OptimizerAdapter | None,
        save_optimizer: bool,
    ) -> _RankSnapshotPersistence:
        started = time.perf_counter()
        snapshot = resolved.result()
        ready = time.perf_counter()
        if save_optimizer and snapshot.optimizer is None:
            raise RuntimeError("optimizer persistence requires an optimizer snapshot")
        result = self._persist_generation(
            generation=generation,
            optimizer_state_path=optimizer_state_path,
            staging_adapter_path=staging_adapter_path,
            lora=(
                snapshot.lora
                if adapter is None and staging_adapter_path is not None
                else None
            ),
            adapter=adapter,
            optimizer=snapshot.optimizer if save_optimizer else None,
            prepared_tensors=(
                snapshot.prepared_tensors
                if adapter is None and staging_adapter_path is not None
                else None
            ),
        )
        return result.model_copy(
            update={
                "metrics": {
                    "time/snapshot_persistence_wait_s": ready - started,
                    "time/snapshot_persistence_s": time.perf_counter() - ready,
                }
            }
        )

    @staticmethod
    def _complete_publication(
        transport: Future[_SnapshotTransport],
        durability: Future[_RankSnapshotPersistence],
    ) -> TrainerRankPublication:
        failures: list[BaseException] = []
        for future in (transport, durability):
            try:
                future.result()
            except BaseException as error:
                failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(
                "adapter persistence and transport failed", failures
            )
        transferred = transport.result()
        persisted = durability.result()
        adapter = persisted.adapter or transferred.adapter
        if persisted.rank == 0 and adapter is None:
            raise RuntimeError("rank zero snapshot requires an adapter output")
        if persisted.rank != 0 and (
            adapter is not None or transferred.adapter is not None
        ):
            raise RuntimeError("nonzero rank unexpectedly published an adapter")
        metrics = {**transferred.metrics, **persisted.metrics}
        metrics["time/snapshot_publication_s"] = max(
            metrics.get("time/snapshot_transport_wait_s", 0.0)
            + metrics.get("time/snapshot_transport_s", 0.0),
            metrics.get("time/snapshot_persistence_wait_s", 0.0)
            + metrics.get("time/snapshot_persistence_s", 0.0),
        )
        return TrainerRankPublication(
            generation=persisted.generation,
            rank=persisted.rank,
            adapter=adapter,
            transport_adapter=transferred.adapter,
            shard=persisted.shard,
            runtime_sha256=persisted.runtime_sha256,
            topology=persisted.topology,
            saves_optimizer=persisted.saves_optimizer,
            metrics=metrics,
        )

    def _transfer_lora_snapshot(
        self,
        lora: Any,
        targets: tuple[Any, ...],
        *,
        prepared_tensors: PreparedSafetensors,
    ) -> None:
        from art.distributed.adapter_transport import AdapterSnapshotSender

        with self._transport_sender_lock:
            if self._transport_sender is None:
                self._transport_sender = AdapterSnapshotSender()
            self._transport_sender.send(
                lora,
                targets,
                prepared_tensors=prepared_tensors,
            )

    def _publish_lora_object_once(
        self,
        entry: _CachedGeneration,
        target: BinaryObjectTarget,
        generation: TrainerGeneration,
        lora: Any,
        prepared_tensors: PreparedSafetensors,
    ) -> OptimizerAdapter:
        with self._lock:
            publication = self._object_publications.get(target.object_id)
            owner = publication is None
            if publication is None:
                publication = Future()
                self._object_publications[target.object_id] = publication
            entry.object_ids.add(target.object_id)
        if not owner:
            return publication.result()
        try:
            adapter = self._publish_lora_object(
                target, generation, lora, prepared_tensors
            )
        except BaseException as error:
            publication.set_exception(error)
            raise
        publication.set_result(adapter)
        return adapter

    def _publish_lora_object(
        self,
        target: BinaryObjectTarget,
        generation: TrainerGeneration,
        lora: Any,
        prepared_tensors: PreparedSafetensors,
    ) -> OptimizerAdapter:
        expected_metadata = {
            "training_session_id": generation.training_session_id,
            "generation_id": generation.generation_id,
            "policy_step": str(generation.policy_step),
        }
        if any(
            target.metadata.get(key) != value
            for key, value in expected_metadata.items()
        ):
            raise RuntimeError(
                "adapter object target identifies another learner generation"
            )
        with self._lock:
            if self._object_store is None:
                self._object_store = S3BinaryObjectStore(target.store)
            elif self._object_store.config != target.store:
                raise RuntimeError("generation publisher cannot mix object stores")
            store = self._object_store
        config = encode_adapter_config(
            {
                **lora.adapter_config,
                ART_LORA_FORMAT_CONFIG_KEY: ART_LORA_FORMAT_VLLM,
            }
        )
        ref = store.publish(
            target,
            {
                "adapter_model.safetensors": tuple(
                    memoryview(chunk.numpy()).cast("B")
                    for chunk in prepared_tensors.chunks
                ),
                "adapter_config.json": (memoryview(config),),
            },
        )
        files = {file.relative_path: file for file in ref.files}
        expected_files = ("adapter_config.json", "adapter_model.safetensors")
        if set(files) != set(expected_files):
            raise RuntimeError("adapter object manifest has unexpected files")
        return OptimizerAdapter(
            identity=ref.manifest_uri,
            training_session_id=generation.training_session_id,
            step=generation.policy_step,
            generation_id=generation.generation_id,
            files=tuple(
                CheckpointFile(name=name, size_bytes=files[name].byte_count)
                for name in expected_files
            ),
        )

    def _persist_generation(
        self,
        *,
        generation: TrainerGeneration,
        optimizer_state_path: str,
        staging_adapter_path: str | None,
        lora: Any,
        adapter: "OptimizerAdapter | None",
        optimizer: Any,
        prepared_tensors: PreparedSafetensors | None,
    ) -> _RankSnapshotPersistence:
        from art.megatron.optimizer_state import (
            adapter_publication_transaction,
            publish_adapter_checkpoint,
            write_optimizer_snapshot_shard,
        )
        from art.megatron.weights.lora_publish import save_vllm_lora_snapshot

        rank = int(self.runtime.rank)
        if rank == 0:
            if lora is not None:
                if staging_adapter_path is None or adapter is not None:
                    raise RuntimeError("new adapter publication is inconsistent")
                with adapter_publication_transaction(
                    staging_adapter_path,
                    step=generation.policy_step,
                    training_session_id=generation.training_session_id,
                    generation_id=generation.generation_id,
                ) as (staging, existing):
                    adapter = existing
                    if adapter is None:
                        save_vllm_lora_snapshot(
                            lora,
                            str(staging),
                            prepared_tensors=prepared_tensors,
                        )
                        adapter = publish_adapter_checkpoint(
                            staging,
                            step=generation.policy_step,
                            training_session_id=generation.training_session_id,
                            generation_id=generation.generation_id,
                        )
        shard = (
            write_optimizer_snapshot_shard(
                optimizer,
                optimizer_state_path=optimizer_state_path,
            )
            if optimizer is not None
            else None
        )
        return _RankSnapshotPersistence(
            generation=generation,
            rank=rank,
            adapter=adapter,
            shard=shard,
            runtime_sha256=None if optimizer is None else optimizer.runtime_sha256,
            topology=None if optimizer is None else optimizer.topology,
            saves_optimizer=optimizer is not None,
        )

    def _completed(
        self,
        future: Future[TrainerRankPublication],
        *,
        entry: _CachedGeneration,
        sink: EventSink,
        generation: TrainerGeneration,
    ) -> None:
        try:
            event = TrainerPublicationSucceeded(record=future.result())
        except BaseException as error:
            self._failed(error, entry=entry, sink=sink, generation=generation)
            return
        try:
            sink.publication(event)
        except BaseException as error:
            with self._lock:
                self._failures.append(error)
        finally:
            self._maybe_release(entry)

    def _failed(
        self,
        error: BaseException,
        *,
        entry: _CachedGeneration,
        sink: EventSink,
        generation: TrainerGeneration,
    ) -> None:
        self._report_failure(
            error,
            entry=entry,
            sink=sink,
            generation=generation,
            remember=True,
        )

    def _report_failure(
        self,
        error: BaseException,
        *,
        entry: _CachedGeneration,
        sink: EventSink,
        generation: TrainerGeneration,
        remember: bool,
    ) -> None:
        if remember:
            with self._lock:
                self._failures.append(error)
        event = TrainerPublicationFailed(
            generation_id=generation.generation_id,
            rank=int(self.runtime.rank),
            error_type=type(error).__name__,
            message=str(error) or type(error).__name__,
        )
        try:
            sink.publication(event)
        except BaseException as sink_error:
            with self._lock:
                self._failures.append(sink_error)
        finally:
            self._maybe_release(entry)

    def _release_slot(self) -> None:
        with self._lock:
            self._in_flight -= 1
        self._slots.release()

    def raise_if_failed(self) -> None:
        with self._lock:
            failures = tuple(self._failures)
        if failures:
            raise BaseExceptionGroup("trainer generation publication failed", failures)

    def close(self) -> None:
        with self._lock:
            entries = tuple(self._cache.values())
            for entry in entries:
                entry.retired = True
        for entry in entries:
            self._maybe_release(entry)
        self._resolution_pool.shutdown(wait=True)
        self._transport_pool.shutdown(wait=True)
        self._durability_pool.shutdown(wait=True)
        self._completion_pool.shutdown(wait=True)
        for entry in entries:
            self._maybe_release(entry)
        if self._transport_sender is not None:
            self._transport_sender.close()
            self._transport_sender = None
        if self._object_store is not None:
            self._object_store.close()
            self._object_store = None
        with self._lock:
            in_flight = self._in_flight
        if in_flight:
            raise RuntimeError(f"publication close retained {in_flight} snapshots")
        self.raise_if_failed()
