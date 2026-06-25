from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import time
from typing import Any, Iterator, cast

from megatron.bridge import AutoBridge
from pydantic import BaseModel, ConfigDict
import torch

from art.megatron.model_support.spec import ModelSupportHandler
from art.megatron.runtime.jobs import (
    MergedWeightTransferInitInfo,
    MergedWeightTransferSpec,
)
from art.megatron.training.model_chunks import ModelChunks, as_megatron_api_chunks
from art.megatron.weights.lora_publish import build_vllm_lora_tensors_from_model
from art.megatron.weights.param_name_canonicalization import (
    canonical_art_param_name,
    is_art_adapter_param_name,
)
from art.weight_transfer import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    TrainerNcclCommunicator,
    trainer_init,
    trainer_send_weights,
)


class MergedWeightExport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bridge: AutoBridge
    model: ModelChunks
    model_config_value: Any
    conversion_tasks: list[Any]
    adapter_weights_by_base: dict[str, list[Any]]


def _hf_param_names(hf_param: Any) -> list[str]:
    if isinstance(hf_param, str):
        return [hf_param]
    return list(hf_param.values())


def build_art_conversion_tasks(*, bridge: AutoBridge, model: ModelChunks) -> list[Any]:
    from megatron.bridge.models.conversion.model_bridge import (
        WeightConversionTask,
        _megatron_local_name_to_global,
    )
    from megatron.bridge.models.conversion.utils import (
        get_module_and_param_from_name,
        persistent_buffers,
    )

    mapping_registry = bridge._model_bridge.mapping_registry()
    hf_source = bridge.hf_pretrained.state.source
    hf_keys = set(hf_source.get_all_keys())
    megatron_models = as_megatron_api_chunks(model)
    model_config = getattr(model[0], "config")
    tasks: list[Any] = []
    for vp_stage, chunk in enumerate(model):
        for local_name, _ in chain(
            chunk.named_parameters(),
            persistent_buffers(chunk),
        ):
            if "_extra_state" in local_name or is_art_adapter_param_name(local_name):
                continue
            global_name = _megatron_local_name_to_global(
                megatron_models,
                model_config,
                canonical_art_param_name(local_name),
                vp_stage,
            )
            mapping = mapping_registry.megatron_to_hf_lookup(global_name)
            if mapping is None:
                raise RuntimeError(
                    f"Missing HF conversion mapping for Megatron param {global_name}"
                )
            hf_params = _hf_param_names(mapping.hf_param)
            missing_hf_params = sorted(set(hf_params) - hf_keys)
            if missing_hf_params and not getattr(
                mapping,
                "allow_hf_name_mismatch",
                False,
            ):
                raise RuntimeError(
                    f"Missing HF checkpoint weights for Megatron param {global_name}: "
                    f"{missing_hf_params}"
                )
            local_module, local_weights = cast(
                tuple[Any, torch.Tensor],
                get_module_and_param_from_name(
                    megatron_models,
                    local_name,
                    vp_stage,
                ),
            )
            if local_module is not None and not hasattr(local_module, "config"):
                setattr(local_module, "config", model_config)
            tasks.append(
                WeightConversionTask(
                    pp_rank=0,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    global_param_name=global_name,
                    megatron_module=local_module,
                    param_weight=local_weights,
                    mapping=mapping,
                )
            )
    return tasks


def build_merged_weight_export(
    *,
    bridge: AutoBridge,
    model: ModelChunks,
    model_support_handler: ModelSupportHandler,
) -> MergedWeightExport:
    return MergedWeightExport(
        bridge=bridge,
        model=model,
        model_config_value=getattr(model[0], "config"),
        conversion_tasks=build_art_conversion_tasks(
            bridge=bridge,
            model=model,
        ),
        adapter_weights_by_base=model_support_handler.build_adapter_weights_by_base(
            model
        ),
    )


def iter_merged_vllm_weights(
    weight_export: MergedWeightExport,
) -> Iterator[tuple[str, torch.Tensor]]:
    bridge = weight_export.bridge
    model_bridge = bridge._model_bridge
    hf_state_dict = bridge.hf_pretrained.state
    grouped_buffers: dict[str, dict[int, torch.Tensor]] = {}
    for task in weight_export.conversion_tasks:
        converted_weights_dict = task.mapping.megatron_to_hf(
            task.param_weight,
            task.megatron_module,
        )
        adapter_weights = weight_export.adapter_weights_by_base.get(
            task.global_param_name
        )
        if adapter_weights is not None:
            try:
                converted_weights_dict = model_bridge._merge_lora_adapter_weights(
                    weight_export.model,
                    converted_weights_dict,
                    adapter_weights,
                )
            except Exception as exc:
                converted_shapes = {
                    key: tuple(value.shape)
                    for key, value in converted_weights_dict.items()
                }
                adapter_summaries = [
                    {
                        "base_prefix": adapter_weight.global_base_prefix,
                        "adapter_key": adapter_weight.adapter_key,
                        "linear_in": tuple(
                            adapter_weight.linear_in_weight.weight.shape
                        ),
                        "linear_out": tuple(
                            adapter_weight.linear_out_weight.weight.shape
                        ),
                    }
                    for adapter_weight in adapter_weights
                ]
                raise RuntimeError(
                    "Failed merged LoRA export for "
                    f"{task.global_param_name}: converted={converted_shapes} "
                    f"adapter_weights={adapter_summaries}"
                ) from exc
        if getattr(task.mapping, "is_grouped_export", False):
            merged_result = model_bridge._accumulate_grouped_export(
                task,
                converted_weights_dict,
                weight_export.model_config_value,
                grouped_buffers,
                hf_state_dict,
            )
            if merged_result is None:
                continue
            converted_weights_dict = merged_result
        else:
            converted_weights_dict = model_bridge.maybe_modify_converted_hf_weight(
                task,
                converted_weights_dict,
                hf_state_dict,
            )
        yield from converted_weights_dict.items()


def _is_sender_rank(rank: int) -> bool:
    return rank == 0


def _maybe_distributed_barrier(world_size: int) -> None:
    if world_size <= 1:
        return
    dist = cast(Any, torch.distributed)
    if not dist.is_available() or not dist.is_initialized():
        return
    dist.barrier()


def _runtime_headers(spec: MergedWeightTransferSpec) -> dict[str, str]:
    if spec.api_key is None:
        return {}
    return {"Authorization": f"Bearer {spec.api_key}"}


def _post_with_retry(
    post: Any,
    url: str,
    *,
    phase: str,
    retry_seconds: float = 10.0,
    **kwargs: Any,
) -> Any:
    if kwargs.get("headers") == {}:
        kwargs = {key: value for key, value in kwargs.items() if key != "headers"}
    deadline = time.monotonic() + retry_seconds
    while True:
        try:
            response = post(url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{phase} failed after retrying for {retry_seconds:g}s"
                ) from exc
            time.sleep(0.5)


def _sync_rank_zero_status(
    *,
    rank: int,
    world_size: int,
    phase: str,
    error: BaseException | None,
) -> None:
    dist = cast(Any, torch.distributed)
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        if error is not None:
            raise RuntimeError(f"{phase} failed on rank 0") from error
        return
    payload = [
        f"{type(error).__name__}: {error}"
        if _is_sender_rank(rank) and error is not None
        else None
    ]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        return
    if _is_sender_rank(rank):
        raise RuntimeError(f"{phase} failed on rank 0: {payload[0]}") from error
    raise RuntimeError(f"{phase} failed on rank 0: {payload[0]}")


def _drain_merged_vllm_weights(
    weight_export: MergedWeightExport,
    *,
    names: list[str] | None = None,
    dtype_names: list[str] | None = None,
    shapes: list[list[int]] | None = None,
) -> None:
    for name, tensor in iter_merged_vllm_weights(weight_export):
        if names is not None:
            assert dtype_names is not None
            assert shapes is not None
            names.append(name)
            dtype_names.append(str(tensor.dtype).removeprefix("torch."))
            shapes.append(list(tensor.shape))


def ensure_merged_weight_transfer_group(
    *,
    rank: int,
    world_size: int,
    merged_weight_transfer_group: TrainerNcclCommunicator | None,
    merged_weight_transfer_init_info: MergedWeightTransferInitInfo | None,
    spec: MergedWeightTransferSpec,
) -> tuple[TrainerNcclCommunicator | None, MergedWeightTransferInitInfo]:
    if merged_weight_transfer_init_info == spec.init_info:
        if _is_sender_rank(rank):
            assert merged_weight_transfer_group is not None
        assert merged_weight_transfer_init_info is not None
        _maybe_distributed_barrier(world_size)
        return merged_weight_transfer_group, merged_weight_transfer_init_info

    import httpx

    error: BaseException | None = None
    if _is_sender_rank(rank):
        init_kwargs = {
            "master_address": spec.init_info.master_address,
            "master_port": spec.init_info.master_port,
            "world_size": spec.init_info.world_size,
            "nccl_so_path": spec.nccl_so_path,
        }
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            trainer_future = executor.submit(trainer_init, init_kwargs)
            _post_with_retry(
                httpx.post,
                f"{spec.vllm_base_url}/init_weight_transfer_engine",
                phase="initialize merged weight transfer",
                json={"init_info": spec.init_info.model_dump()},
                headers=_runtime_headers(spec),
                timeout=300.0,
            )
            merged_weight_transfer_group = trainer_future.result()
        except BaseException as exc:
            error = exc
        finally:
            executor.shutdown(wait=error is None, cancel_futures=error is not None)
    _sync_rank_zero_status(
        rank=rank,
        world_size=world_size,
        phase="initialize merged weight transfer",
        error=error,
    )
    return merged_weight_transfer_group, spec.init_info


def sync_merged_weights_to_vllm(
    *,
    bridge: Any,
    model: ModelChunks,
    model_support_handler: Any,
    adapter_model: dict[str, torch.Tensor],
    adapter_config: dict[str, Any],
    rank: int,
    world_size: int,
    merged_weight_transfer_group: TrainerNcclCommunicator | None,
    merged_weight_transfer_init_info: MergedWeightTransferInitInfo | None,
    spec: MergedWeightTransferSpec,
    pause_generation: bool,
) -> tuple[TrainerNcclCommunicator | None, MergedWeightTransferInitInfo]:
    import httpx

    (
        merged_weight_transfer_group,
        merged_weight_transfer_init_info,
    ) = ensure_merged_weight_transfer_group(
        rank=rank,
        world_size=world_size,
        merged_weight_transfer_group=merged_weight_transfer_group,
        merged_weight_transfer_init_info=merged_weight_transfer_init_info,
        spec=spec,
    )
    _ = bridge
    lora_result = build_vllm_lora_tensors_from_model(
        model=model,
        adapter_model=adapter_model,
        handler=model_support_handler,
        adapter_config=adapter_config,
        rank=rank,
        world_size=world_size,
    )
    lora_weights: list[tuple[str, torch.Tensor]] = []
    published_config: dict[str, Any] = {}
    if _is_sender_rank(rank):
        assert lora_result is not None
        vllm_lora_tensors, published_config = lora_result
        lora_weights = sorted(vllm_lora_tensors.items())

    def _send_weights() -> None:
        assert merged_weight_transfer_group is not None
        trainer_send_weights(
            iter(lora_weights),
            {
                "group": merged_weight_transfer_group,
                "packed": True,
                "packed_buffer_size_bytes": DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                "packed_num_buffers": DEFAULT_PACKED_NUM_BUFFERS,
            },
        )

    torch.cuda.synchronize()
    names = [name for name, _tensor in lora_weights]
    dtype_names = [
        str(tensor.dtype).removeprefix("torch.") for _name, tensor in lora_weights
    ]
    shapes = [list(tensor.shape) for _name, tensor in lora_weights]
    _maybe_distributed_barrier(world_size)

    pause_error: BaseException | None = None
    update_error: BaseException | None = None
    resume_error: BaseException | None = None

    if _is_sender_rank(rank):
        with httpx.Client() as client:
            if pause_generation:
                try:
                    _post_with_retry(
                        client.post,
                        f"{spec.vllm_base_url}/pause",
                        phase="pause generation",
                        params={"mode": "wait"},
                        headers=_runtime_headers(spec),
                        timeout=300.0,
                    )
                except BaseException as exc:
                    pause_error = exc

            _sync_rank_zero_status(
                rank=rank,
                world_size=world_size,
                phase="pause generation",
                error=pause_error,
            )
            try:
                _post_with_retry(
                    client.post,
                    f"{spec.vllm_base_url}/start_weight_update",
                    phase="start merged weight update",
                    json={"is_checkpoint_format": False},
                    headers=_runtime_headers(spec),
                    timeout=300.0,
                )
                with ThreadPoolExecutor(max_workers=1) as executor:
                    send_future = executor.submit(_send_weights)
                    _post_with_retry(
                        client.post,
                        f"{spec.vllm_base_url}/update_weights",
                        phase="update merged weights",
                        json={
                            "update_info": {
                                "art_weight_update_kind": "lora_delta",
                                "art_lora_config": published_config,
                                "names": names,
                                "dtype_names": dtype_names,
                                "shapes": shapes,
                                "packed": True,
                                "packed_buffer_size_bytes": DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                                "packed_num_buffers": DEFAULT_PACKED_NUM_BUFFERS,
                            }
                        },
                        headers=_runtime_headers(spec),
                        timeout=600.0,
                    )
                    send_future.result()
                _post_with_retry(
                    client.post,
                    f"{spec.vllm_base_url}/finish_weight_update",
                    phase="finish merged weight update",
                    headers=_runtime_headers(spec),
                    timeout=600.0,
                )
                _post_with_retry(
                    client.post,
                    f"{spec.vllm_base_url}/art/set_served_model_name",
                    phase="set served model name",
                    json={"name": spec.served_model_name},
                    headers=_runtime_headers(spec),
                    timeout=30.0,
                )
                torch.cuda.synchronize()
            except BaseException as exc:
                update_error = exc
            finally:
                if pause_generation:
                    try:
                        _post_with_retry(
                            client.post,
                            f"{spec.vllm_base_url}/resume",
                            phase="resume generation",
                            headers=_runtime_headers(spec),
                            timeout=30.0,
                        )
                    except BaseException as exc:
                        resume_error = exc
                _sync_rank_zero_status(
                    rank=rank,
                    world_size=world_size,
                    phase="update merged weights",
                    error=update_error,
                )
                _sync_rank_zero_status(
                    rank=rank,
                    world_size=world_size,
                    phase="resume generation",
                    error=resume_error,
                )
    else:
        _sync_rank_zero_status(
            rank=rank,
            world_size=world_size,
            phase="pause generation",
            error=None,
        )
        _sync_rank_zero_status(
            rank=rank,
            world_size=world_size,
            phase="update merged weights",
            error=None,
        )
        _sync_rank_zero_status(
            rank=rank,
            world_size=world_size,
            phase="resume generation",
            error=None,
        )
    return merged_weight_transfer_group, merged_weight_transfer_init_info


__all__ = [
    "MergedWeightExport",
    "build_art_conversion_tasks",
    "build_merged_weight_export",
    "ensure_merged_weight_transfer_group",
    "iter_merged_vllm_weights",
    "sync_merged_weights_to_vllm",
]
