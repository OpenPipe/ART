"""Fast parallel direct-to-GPU + NVLink broadcast HF->Megatron base weight loader.

Accelerates pretrained weight loading in ``art.megatron.runtime.bridge_runtime``:
  1. Eliminates redundant checkpoint reads across DP/CP and expert-DP replicas:
     only ``dp_cp_rank == 0`` reads non-expert weights and only ``edp_rank == 0``
     reads each disjoint expert-parallel slice, then coalesced 256 MiB-bucketed
     NCCL broadcasts replicate GPU weights across NVLink.
  2. Eliminates host ``pin_memory()`` RAM pressure by reading shard-grouped
     safetensors tensors directly onto the target CUDA device in bounded parallel
     shard waves (``ThreadPoolExecutor`` + ``safe_open(..., device=...)``),
     immediately converting ready tasks and freeing temporary HF tensors between
     waves so peak extra VRAM stays bounded.
  3. Separates non-expert (``tp_group``) and expert (``etp_group``) wave phases
     whenever ``tp_size > 1`` or ``etp_size > 1`` so TP/ETP collectives always
     execute in deterministic lockstep even for unfused per-expert checkpoints or
     asymmetric ``ETP != TP`` / ``PP > 1`` topologies.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import contextlib
import fnmatch
import logging
import os
from pathlib import Path
import time
from typing import Any

logger = logging.getLogger(__name__)

FAST_WEIGHT_LOAD_ENV = "ART_MEGATRON_FAST_WEIGHT_LOAD"
FAST_WEIGHT_LOAD_WAVE_SIZE_ENV = "ART_MEGATRON_FAST_WEIGHT_LOAD_WAVE_SIZE"
FAST_WEIGHT_LOAD_WORKERS_ENV = "ART_MEGATRON_FAST_WEIGHT_LOAD_WORKERS"

_DEFAULT_SHARD_WORKERS = 8
_DEFAULT_BROADCAST_BUCKET_BYTES = 256 * 1024 * 1024

_TP_RANK0_ONLY_MAPPING_NAMES = frozenset(
    {
        "ColumnParallelMapping",
        "RowParallelMapping",
        "QKVMapping",
        "GatedMLPMapping",
        "GDNConv1dMapping",
        "GDNLinearMappingSeparate",
    }
)


def _tp_non_zero_skips_hf_read(task: Any) -> bool:
    """Return True when a non-zero TP rank's ``hf_to_megatron`` ignores ``hf_weights``."""
    mapping = task.mapping
    if (
        int(getattr(mapping, "tp_size", 1)) <= 1
        or int(getattr(mapping, "tp_rank", 0)) == 0
    ):
        return False
    cls = type(mapping)
    if cls.__module__ != "megatron.bridge.models.conversion.param_mapping":
        return False
    if cls.__name__ in _TP_RANK0_ONLY_MAPPING_NAMES:
        return True
    if cls.__name__ == "AutoMapping":
        detect = getattr(mapping, "_detect_parallelism_type", None)
        if detect is not None and task.megatron_module is not None:
            with contextlib.suppress(Exception):
                return detect(task.megatron_module) in {"column", "row"}
    return False


def fast_weight_load_enabled() -> bool:
    """Return True unless explicitly disabled via ``ART_MEGATRON_FAST_WEIGHT_LOAD=0``."""
    return os.environ.get(FAST_WEIGHT_LOAD_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def resolve_shard_wave_size(free_gpu_bytes: int | None, num_shards: int) -> int:
    """Choose a bounded parallel shard wave size based on available GPU memory."""
    env_raw = os.environ.get(FAST_WEIGHT_LOAD_WAVE_SIZE_ENV, "").strip()
    if env_raw:
        with contextlib.suppress(ValueError):
            return max(1, min(max(1, num_shards), int(env_raw)))
    if free_gpu_bytes is None:
        base = 8
    elif free_gpu_bytes >= 80 * 1024**3:
        base = 8
    elif free_gpu_bytes >= 40 * 1024**3:
        base = 4
    else:
        base = 2
    return max(1, min(max(1, num_shards), base))


def resolve_shard_workers(num_shards: int) -> int:
    """Choose bounded reader thread count for parallel safetensors shard reads."""
    env_raw = os.environ.get(FAST_WEIGHT_LOAD_WORKERS_ENV, "").strip()
    target = _DEFAULT_SHARD_WORKERS
    if env_raw:
        with contextlib.suppress(ValueError):
            target = max(1, min(32, int(env_raw)))
    return max(1, min(target, max(1, num_shards)))


def _is_contiguous(tensor: Any) -> bool:
    fn = getattr(tensor, "is_contiguous", None)
    return True if fn is None else bool(fn())


def _coalesced_broadcast_tensors(
    params: Sequence[Any],
    *,
    group: Any,
    src_global_rank: int,
    my_global_rank: int,
    torch_mod: Any,
    dist_mod: Any,
    bucket_bytes: int = _DEFAULT_BROADCAST_BUCKET_BYTES,
) -> None:
    """Broadcast parameter tensors in coalesced ``(device, dtype)``-grouped flat buckets."""
    if not params:
        return
    by_key: dict[tuple[Any, Any], list[Any]] = defaultdict(list)
    for tensor in params:
        if int(tensor.numel()) == 0:
            continue
        by_key[(tensor.device, tensor.dtype)].append(tensor)

    for (cur_device, cur_dtype), tensors in by_key.items():
        elem_size = max(1, int(tensors[0].element_size()))
        max_elems = max(1, int(bucket_bytes) // elem_size)
        batch: list[Any] = []
        batch_elems = 0

        def _flush(
            cur: list[Any],
            device: Any = cur_device,
            dtype: Any = cur_dtype,
        ) -> None:
            if len(cur) == 1:
                only = cur[0]
                if _is_contiguous(only):
                    dist_mod.broadcast(only, src=src_global_rank, group=group)
                    return
                flat_only = torch_mod.empty(
                    int(only.numel()), dtype=dtype, device=device
                )
                if my_global_rank == src_global_rank:
                    flat_only.copy_(only.reshape(-1), non_blocking=True)
                dist_mod.broadcast(flat_only, src=src_global_rank, group=group)
                if my_global_rank != src_global_rank:
                    only.copy_(flat_only.reshape(only.shape), non_blocking=True)
                return
            total_elems = sum(int(t.numel()) for t in cur)
            flat = torch_mod.empty(total_elems, dtype=dtype, device=device)
            if my_global_rank == src_global_rank:
                offset = 0
                for t in cur:
                    n = int(t.numel())
                    flat[offset : offset + n].copy_(t.reshape(-1), non_blocking=True)
                    offset += n
            dist_mod.broadcast(flat, src=src_global_rank, group=group)
            if my_global_rank != src_global_rank:
                offset = 0
                for t in cur:
                    n = int(t.numel())
                    t.copy_(
                        flat[offset : offset + n].reshape(t.shape),
                        non_blocking=True,
                    )
                    offset += n

        for tensor in tensors:
            n = int(tensor.numel())
            if batch and batch_elems + n > max_elems:
                _flush(batch)
                batch = []
                batch_elems = 0
            batch.append(tensor)
            batch_elems += n
        if batch:
            _flush(batch)


def partition_load_tasks(
    tasks: Iterable[Any],
    *,
    dp_cp_rank: int,
    dp_cp_size: int,
    edp_rank: int,
    edp_size: int,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Split conversion tasks into rank-local I/O tasks and broadcast parameter lists."""
    local_tasks: list[Any] = []
    dp_cp_bcast_params: list[Any] = []
    dp_cp_seen: set[int] = set()
    edp_bcast_params: list[Any] = []
    edp_seen: set[int] = set()

    for task in tasks:
        if task is None or task.megatron_module is None:
            continue
        is_expert = bool(getattr(task.mapping, "is_expert", False))
        pw = task.param_weight
        if is_expert:
            if edp_rank == 0:
                local_tasks.append(task)
            if edp_size > 1 and pw is not None:
                pid = id(pw)
                if pid not in edp_seen:
                    edp_seen.add(pid)
                    edp_bcast_params.append(pw.data)
        else:
            if dp_cp_rank == 0:
                local_tasks.append(task)
            if dp_cp_size > 1 and pw is not None:
                pid = id(pw)
                if pid not in dp_cp_seen:
                    dp_cp_seen.add(pid)
                    dp_cp_bcast_params.append(pw.data)

    return local_tasks, dp_cp_bcast_params, edp_bcast_params


def _sync_tp_min_free_bytes(
    free_gpu_bytes: int | None,
    *,
    tp_group: Any,
    device_str: str,
    torch_mod: Any,
    dist_mod: Any,
) -> int | None:
    """Reduce ``free_gpu_bytes`` to the group minimum so peer ranks share one wave size."""
    if tp_group is None or dist_mod.get_world_size(tp_group) <= 1:
        return free_gpu_bytes
    sentinel = 1 << 62
    buf = torch_mod.tensor(
        [free_gpu_bytes if free_gpu_bytes is not None else sentinel],
        dtype=torch_mod.int64,
        device=device_str,
    )
    dist_mod.all_reduce(buf, op=dist_mod.ReduceOp.MIN, group=tp_group)
    reduced = int(buf.item())
    return None if reduced >= sentinel else reduced


def _get_optional_group(fn: Callable[..., Any] | None) -> Any:
    if fn is None:
        return None
    try:
        return fn(check_initialized=False)
    except TypeError:
        with contextlib.suppress(Exception):
            return fn()
        return None
    except Exception:
        return None


def _resolve_target_device(
    tasks: Sequence[Any],
    *,
    torch_mod: Any,
) -> tuple[bool, str, Any]:
    cuda_available = bool(torch_mod.cuda.is_available())
    for task in tasks:
        if task is None or task.megatron_module is None or task.param_weight is None:
            continue
        param_device = task.param_weight.device
        if param_device.type == "cuda" and cuda_available:
            idx = (
                torch_mod.cuda.current_device()
                if param_device.index is None
                else int(param_device.index)
            )
            return True, f"cuda:{idx}", torch_mod.device("cuda", idx)
        return False, "cpu", param_device
    if cuda_available:
        idx = int(torch_mod.cuda.current_device())
        return True, f"cuda:{idx}", torch_mod.device("cuda", idx)
    return False, "cpu", torch_mod.device("cpu")


def _load_task_group_in_waves(
    bridge: Any,
    group_tasks: list[Any],
    *,
    hf_state_dict: Mapping[str, Any],
    key_to_filename: Mapping[str, str],
    source_path: str | Path,
    allowed_mismatched_params: list[str] | None,
    use_cuda: bool,
    device_str: str,
    target_device: Any,
    sync_group: Any,
    dist_ready: bool,
    torch_mod: Any,
    dist_mod: Any,
    br: Any,
    safe_open_fn: Any,
) -> None:
    """Load and convert one collective-homogeneous group of tasks in bounded shard waves."""
    free_gpu_bytes: int | None = None
    if use_cuda:
        with contextlib.suppress(Exception):
            free_gpu_bytes = int(torch_mod.cuda.mem_get_info(target_device)[0])
        if dist_ready:
            with contextlib.suppress(Exception):
                free_gpu_bytes = _sync_tp_min_free_bytes(
                    free_gpu_bytes,
                    tp_group=sync_group,
                    device_str=device_str,
                    torch_mod=torch_mod,
                    dist_mod=dist_mod,
                )

    # Compute a deterministic shard wave schedule identical across every rank in
    # `sync_group` so TP/ETP scatters execute in lockstep without deadlock.
    all_phys_shards: set[str] = set()
    task_shards: list[set[str]] = []
    all_group_keys: set[str] = set()
    for task in group_tasks:
        tshards: set[str] = set()
        for key in br._iter_hf_param_names(task.mapping.hf_param):
            all_group_keys.add(key)
            fn = key_to_filename.get(key)
            if fn is not None:
                tshards.add(fn)
                all_phys_shards.add(fn)
                continue
            src = br._planned_hf_weight_source(bridge, key, task=task)
            opt = br._select_physical_key_option(src, hf_state_dict)
            for phys in opt:
                phys_fn = key_to_filename.get(phys)
                if phys_fn is not None:
                    tshards.add(phys_fn)
                    all_phys_shards.add(phys_fn)
        task_shards.append(tshards)

    extra_keys_fn = getattr(bridge, "art_extra_hf_prefetch_keys", None)
    if extra_keys_fn is not None and all_group_keys:
        for key in extra_keys_fn(tuple(sorted(all_group_keys)), hf_state_dict):
            fn = key_to_filename.get(key)
            if fn is not None:
                all_phys_shards.add(fn)
                continue
            src = br._planned_hf_weight_source(bridge, key, task=None)
            opt = br._select_physical_key_option(src, hf_state_dict)
            for phys in opt:
                phys_fn = key_to_filename.get(phys)
                if phys_fn is not None:
                    all_phys_shards.add(phys_fn)

    prefetch_task_by_key: dict[str, Any] = {}
    for task in group_tasks:
        if br._needs_local_hf_prefetch(task):
            for key in br._iter_hf_param_names(task.mapping.hf_param):
                prefetch_task_by_key.setdefault(key, task)
    if extra_keys_fn is not None and prefetch_task_by_key:
        for key in extra_keys_fn(tuple(sorted(prefetch_task_by_key)), hf_state_dict):
            prefetch_task_by_key.setdefault(key, None)

    direct_phys_by_logical: dict[str, str] = {}
    materialized_sources: dict[str, tuple[Any, tuple[str, ...]]] = {}
    for key in sorted(prefetch_task_by_key):
        src = br._planned_hf_weight_source(
            bridge, key, task=prefetch_task_by_key.get(key)
        )
        opt = br._select_physical_key_option(src, hf_state_dict)
        if src.kind == "direct":
            if len(opt) != 1:
                raise RuntimeError(
                    "direct HF source must select exactly one physical key for "
                    f"{src.logical_key!r}; got {opt!r}"
                )
            direct_phys_by_logical[key] = opt[0]
        else:
            materialized_sources[key] = (src, opt)

    expert_slices_needed: dict[str, tuple[int, int, Any]] = {}
    for task in group_tasks:
        if not br._needs_expert_slice_prefetch(task):
            continue
        start, stop = br._expert_slice_range(task)
        for key in br._iter_hf_param_names(task.mapping.hf_param):
            prev = expert_slices_needed.get(key)
            if prev is None:
                expert_slices_needed[key] = (start, stop, task)
            else:
                expert_slices_needed[key] = (
                    min(prev[0], start),
                    max(prev[1], stop),
                    prev[2],
                )

    shard_direct: dict[str, list[str]] = defaultdict(list)
    seen_phys: set[str] = set()
    for phys in direct_phys_by_logical.values():
        if phys not in seen_phys and phys in key_to_filename:
            seen_phys.add(phys)
            fn = key_to_filename[phys]
            shard_direct[fn].append(phys)

    shard_expert: dict[str, list[tuple[str, str, int, int, Any]]] = defaultdict(list)
    non_direct_expert: list[tuple[str, int, int, Any]] = []
    for key, (start, stop, task) in expert_slices_needed.items():
        src = br._planned_hf_weight_source(bridge, key, task=task)
        opt = br._select_physical_key_option(src, hf_state_dict)
        if src.kind == "direct" and len(opt) == 1 and opt[0] in key_to_filename:
            fn = key_to_filename[opt[0]]
            shard_expert[fn].append((key, opt[0], start, stop, task))
        else:
            non_direct_expert.append((key, start, stop, task))

    all_shards = sorted(all_phys_shards)
    task_needed_keys: list[set[str]] = []
    key_use_count: dict[str, int] = defaultdict(int)
    for task in group_tasks:
        tkeys = set(br._iter_hf_param_names(task.mapping.hf_param))
        task_needed_keys.append(tkeys)
        for k in tkeys:
            key_use_count[k] += 1

    def _load_one_shard(fname: str) -> tuple[dict[str, Any], dict[str, Any]]:
        d_keys = shard_direct.get(fname, ())
        e_keys = shard_expert.get(fname, ())
        if not d_keys and not e_keys:
            return {}, {}
        fpath = Path(source_path) / fname
        d_out: dict[str, Any] = {}
        e_out: dict[str, Any] = {}
        with safe_open_fn(fpath, framework="pt", device=device_str) as handle:
            for phys in d_keys:
                d_out[phys] = handle.get_tensor(phys)
            for lkey, phys, start, stop, task in e_keys:
                sl = handle.get_slice(phys)
                sh = tuple(int(d) for d in sl.get_shape())
                if not sh or start < 0 or stop > sh[0] or start >= stop:
                    raise RuntimeError(
                        f"invalid expert slice [{start}, {stop}) for {phys!r} with shape {sh}"
                    )
                idx = (slice(start, stop),) + (slice(None),) * (len(sh) - 1)
                layout = br.get_expert_parallel_layout(
                    getattr(getattr(task, "megatron_module", None), "config", None)
                )
                e_out[lkey] = br.ExpertTensorSlice(
                    sl[idx],
                    global_start=start,
                    global_stop=stop,
                    physical_to_logical=(
                        None if layout is None else layout.physical_to_logical
                    ),
                )
        return d_out, e_out

    phys_to_logicals: dict[str, list[str]] = defaultdict(list)
    for lkey, phys in direct_phys_by_logical.items():
        phys_to_logicals[phys].append(lkey)

    raw_cache: dict[str, Any] = {}
    has_local_gpu_cache = use_cuda and bool(
        shard_direct or shard_expert or materialized_sources or non_direct_expert
    )
    try:
        cached_state = br._CachedStateLookup(
            cache=raw_cache,
            source=hf_state_dict,
            device=device_str if has_local_gpu_cache else None,
        )
    except TypeError:
        cached_state = br._CachedStateLookup(cache=raw_cache, source=hf_state_dict)

    for key, (src, opt) in materialized_sources.items():
        raw_cache[key] = br._materialize_hf_weight_source(
            bridge, src, hf_state_dict, selected_option=opt
        ).to(device_str, non_blocking=True)
    for key, start, stop, task in non_direct_expert:
        src = br._planned_hf_weight_source(bridge, key, task=task)
        opt = br._select_physical_key_option(src, hf_state_dict)
        ten = br._materialize_hf_weight_source(
            bridge, src, hf_state_dict, selected_option=opt
        )
        if not ten.ndim or start < 0 or stop > ten.shape[0] or start >= stop:
            raise RuntimeError(
                f"invalid expert slice [{start}, {stop}) for {key!r} with shape {tuple(ten.shape)}"
            )
        layout = br.get_expert_parallel_layout(
            getattr(getattr(task, "megatron_module", None), "config", None)
        )
        raw_cache[key] = br.ExpertTensorSlice(
            ten[start:stop].to(device_str, non_blocking=True),
            global_start=start,
            global_stop=stop,
            physical_to_logical=(
                None if layout is None else layout.physical_to_logical
            ),
        )
        del ten

    loaded_shards: set[str] = set()
    pending_task_indices = list(range(len(group_tasks)))

    def _run_ready_tasks(drain_all: bool = False) -> None:
        nonlocal pending_task_indices
        still_pending: list[int] = []
        for ti in pending_task_indices:
            if not drain_all and not task_shards[ti].issubset(loaded_shards):
                still_pending.append(ti)
                continue
            task = group_tasks[ti]
            hf_param = task.mapping.hf_param
            if _tp_non_zero_skips_hf_read(task):
                hf_weights = None
            elif (
                isinstance(hf_param, str)
                and hf_param in raw_cache
                and hf_param not in hf_state_dict
            ):
                hf_weights = raw_cache[hf_param]
            else:
                hf_weights = bridge.maybe_modify_loaded_hf_weight(
                    hf_param, cached_state
                )
            converted = task.mapping.hf_to_megatron(hf_weights, task.megatron_module)
            if converted is not None:
                assert task.param_weight is not None, (
                    "param_weight is required for HF->Megatron conversion"
                )
                if converted.shape != task.param_weight.shape:
                    is_whitelisted = bool(allowed_mismatched_params) and any(
                        fnmatch.fnmatch(task.mapping.megatron_param, pat)
                        or fnmatch.fnmatch(task.param_name, pat)
                        for pat in (allowed_mismatched_params or ())
                    )
                    if not is_whitelisted:
                        raise ValueError(
                            f"Shape mismatch for megatron param {task.mapping.megatron_param}:\n"
                            f"  Expected shape: {task.param_weight.shape}\n"
                            f"  Got shape: {converted.shape}\n"
                            f"  Bridge type: {type(task.mapping).__name__}\n"
                            f"  HF mapping: {task.mapping.hf_param}"
                        )
                else:
                    if converted.data_ptr() != task.param_weight.data.data_ptr():
                        task.param_weight.data.copy_(converted, non_blocking=True)
            del hf_weights, converted
            for k in task_needed_keys[ti]:
                key_use_count[k] -= 1
                if key_use_count[k] <= 0:
                    raw_cache.pop(k, None)
        pending_task_indices = still_pending

    wave_size = resolve_shard_wave_size(free_gpu_bytes, len(all_shards))
    has_shard_reads = bool(shard_direct or shard_expert)
    if all_shards and has_shard_reads:
        workers = resolve_shard_workers(min(wave_size, max(1, len(all_shards))))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="art-wt-load"
        ) as ex:
            for off in range(0, len(all_shards), wave_size):
                wave = all_shards[off : off + wave_size]
                for d_out, e_out in ex.map(_load_one_shard, wave):
                    while d_out:
                        phys, ten = d_out.popitem()
                        for lkey in phys_to_logicals.get(phys, ()):
                            raw_cache[lkey] = ten
                        del ten
                    raw_cache.update(e_out)
                    e_out.clear()
                    del d_out, e_out
                loaded_shards.update(wave)
                _run_ready_tasks(drain_all=False)
            _run_ready_tasks(drain_all=True)
    else:
        for off in range(0, len(all_shards), wave_size):
            loaded_shards.update(all_shards[off : off + wave_size])
            _run_ready_tasks(drain_all=False)
        _run_ready_tasks(drain_all=True)

    raw_cache.clear()


def fast_load_prepared_weights_hf_to_megatron(
    bridge: Any,
    hf_pretrained: Any,
    megatron_model: list[Any],
    tasks: list[Any],
    *,
    allowed_mismatched_params: list[str] | None = None,
    key_to_filename: Mapping[str, str],
    source_path: str | Path,
) -> list[Any]:
    """Load prepared HF->Megatron conversion tasks via parallel waves and NVLink broadcast."""
    from megatron.core import parallel_state as ps
    from safetensors import safe_open
    import torch
    import torch.distributed as dist

    from art.megatron.runtime import bridge_runtime as br

    t0 = time.perf_counter()
    hf_state_dict = hf_pretrained.state

    dist_ready = dist.is_available() and dist.is_initialized()
    my_rank = dist.get_rank() if dist_ready else 0

    dp_cp_group = None
    if dist_ready:
        with contextlib.suppress(Exception):
            dp_cp_group = ps.get_data_parallel_group(with_context_parallel=True)
    dp_cp_size = dist.get_world_size(dp_cp_group) if dp_cp_group is not None else 1
    dp_cp_rank = dist.get_rank(dp_cp_group) if dp_cp_group is not None else 0
    dp_cp_src = (
        dist.get_global_rank(dp_cp_group, 0)
        if dp_cp_size > 1 and dp_cp_group is not None
        else my_rank
    )

    edp_group = (
        _get_optional_group(getattr(ps, "get_expert_data_parallel_group", None))
        if dist_ready
        else None
    )
    edp_size = dist.get_world_size(edp_group) if edp_group is not None else 1
    edp_rank = dist.get_rank(edp_group) if edp_group is not None else 0
    edp_src = (
        dist.get_global_rank(edp_group, 0)
        if edp_size > 1 and edp_group is not None
        else my_rank
    )

    tp_group = (
        _get_optional_group(getattr(ps, "get_tensor_model_parallel_group", None))
        if dist_ready
        else None
    )
    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1

    etp_group = (
        _get_optional_group(getattr(ps, "get_expert_tensor_parallel_group", None))
        if dist_ready
        else None
    )
    etp_size = dist.get_world_size(etp_group) if etp_group is not None else 1

    local_tasks, dp_cp_bcast_params, edp_bcast_params = partition_load_tasks(
        tasks,
        dp_cp_rank=dp_cp_rank,
        dp_cp_size=dp_cp_size,
        edp_rank=edp_rank,
        edp_size=edp_size,
    )

    use_cuda, device_str, target_device = _resolve_target_device(tasks, torch_mod=torch)
    t_plan = time.perf_counter()

    if local_tasks:
        dense_local_tasks = [
            task
            for task in local_tasks
            if not bool(getattr(task.mapping, "is_expert", False))
        ]
        expert_local_tasks = [
            task
            for task in local_tasks
            if bool(getattr(task.mapping, "is_expert", False))
        ]
        if tp_size <= 1 and etp_size <= 1:
            _load_task_group_in_waves(
                bridge,
                local_tasks,
                hf_state_dict=hf_state_dict,
                key_to_filename=key_to_filename,
                source_path=source_path,
                allowed_mismatched_params=allowed_mismatched_params,
                use_cuda=use_cuda,
                device_str=device_str,
                target_device=target_device,
                sync_group=None,
                dist_ready=dist_ready,
                torch_mod=torch,
                dist_mod=dist,
                br=br,
                safe_open_fn=safe_open,
            )
        else:
            if dense_local_tasks:
                _load_task_group_in_waves(
                    bridge,
                    dense_local_tasks,
                    hf_state_dict=hf_state_dict,
                    key_to_filename=key_to_filename,
                    source_path=source_path,
                    allowed_mismatched_params=allowed_mismatched_params,
                    use_cuda=use_cuda,
                    device_str=device_str,
                    target_device=target_device,
                    sync_group=tp_group,
                    dist_ready=dist_ready,
                    torch_mod=torch,
                    dist_mod=dist,
                    br=br,
                    safe_open_fn=safe_open,
                )
            if expert_local_tasks:
                _load_task_group_in_waves(
                    bridge,
                    expert_local_tasks,
                    hf_state_dict=hf_state_dict,
                    key_to_filename=key_to_filename,
                    source_path=source_path,
                    allowed_mismatched_params=allowed_mismatched_params,
                    use_cuda=use_cuda,
                    device_str=device_str,
                    target_device=target_device,
                    sync_group=etp_group,
                    dist_ready=dist_ready,
                    torch_mod=torch,
                    dist_mod=dist,
                    br=br,
                    safe_open_fn=safe_open,
                )
        if use_cuda:
            torch.cuda.synchronize(target_device)

    t_read = time.perf_counter()

    if dist_ready and dp_cp_size > 1 and dp_cp_bcast_params:
        _coalesced_broadcast_tensors(
            dp_cp_bcast_params,
            group=dp_cp_group,
            src_global_rank=dp_cp_src,
            my_global_rank=my_rank,
            torch_mod=torch,
            dist_mod=dist,
        )
    if dist_ready and edp_size > 1 and edp_bcast_params:
        _coalesced_broadcast_tensors(
            edp_bcast_params,
            group=edp_group,
            src_global_rank=edp_src,
            my_global_rank=my_rank,
            torch_mod=torch,
            dist_mod=dist,
        )

    if use_cuda:
        torch.cuda.synchronize(target_device)
        torch.cuda.empty_cache()
    t_bcast = time.perf_counter()

    bridge._broadcast_shared_embeddings(
        br._shared_embedding_broadcast_model(megatron_model)
    )
    if my_rank == 0:
        logger.info(
            "Fast HF->Megatron weight load for %s completed in %.2fs "
            "(plan=%.2fs, read_conv=%.2fs, nvlink_bcast=%.2fs, dp_cp=%d, edp=%d)",
            hf_pretrained.model_name_or_path,
            time.perf_counter() - t0,
            t_plan - t0,
            t_read - t_plan,
            t_bcast - t_read,
            dp_cp_size,
            edp_size,
        )
    return megatron_model
