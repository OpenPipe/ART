"""Unit and multi-rank parity tests for ART's native fast HF->Megatron weight loader."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import fnmatch
import gc
from pathlib import Path
import queue
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any
import weakref

import numpy as np
import pytest

from art.megatron.runtime._fast_weight_loader import (
    _coalesced_broadcast_tensors,
    _sync_tp_min_free_bytes,
    _tp_non_zero_skips_hf_read,
    fast_load_prepared_weights_hf_to_megatron,
    fast_weight_load_enabled,
    partition_load_tasks,
    resolve_shard_wave_size,
    resolve_shard_workers,
)


def _import_torch() -> Any:
    return pytest.importorskip("torch")


def _import_safetensors() -> Any:
    return pytest.importorskip("safetensors")


# ---------------------------------------------------------------------------
# 1. Policy knobs, task partitioning, TP free-memory sync & coalesced bcast
# ---------------------------------------------------------------------------


def test_fast_weight_loader_policy_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ART_MEGATRON_FAST_WEIGHT_LOAD", raising=False)
    assert fast_weight_load_enabled() is True
    for off_val in ("0", "false", "FALSE", "no", "off"):
        monkeypatch.setenv("ART_MEGATRON_FAST_WEIGHT_LOAD", off_val)
        assert fast_weight_load_enabled() is False
    monkeypatch.setenv("ART_MEGATRON_FAST_WEIGHT_LOAD", "1")
    assert fast_weight_load_enabled() is True

    monkeypatch.delenv("ART_MEGATRON_FAST_WEIGHT_LOAD_WAVE_SIZE", raising=False)
    assert resolve_shard_wave_size(120 * 1024**3, 18) == 8
    assert resolve_shard_wave_size(50 * 1024**3, 18) == 4
    assert resolve_shard_wave_size(20 * 1024**3, 18) == 2
    assert resolve_shard_wave_size(None, 18) == 8
    assert resolve_shard_wave_size(120 * 1024**3, 3) == 3
    assert resolve_shard_wave_size(120 * 1024**3, 0) == 1

    monkeypatch.setenv("ART_MEGATRON_FAST_WEIGHT_LOAD_WAVE_SIZE", "5")
    assert resolve_shard_wave_size(120 * 1024**3, 18) == 5

    monkeypatch.delenv("ART_MEGATRON_FAST_WEIGHT_LOAD_WORKERS", raising=False)
    assert resolve_shard_workers(18) == 8
    assert resolve_shard_workers(3) == 3
    monkeypatch.setenv("ART_MEGATRON_FAST_WEIGHT_LOAD_WORKERS", "64")
    assert resolve_shard_workers(40) == 32


def test_tp_non_zero_skips_hf_read_gates_strictly_on_builtin_tp_mappings() -> None:
    def _mk_mapping(
        cls_name: str,
        module_name: str,
        *,
        tp_size: int,
        tp_rank: int,
        detected_type: str | None = None,
    ) -> Any:
        attrs: dict[str, Any] = {"__module__": module_name}
        if detected_type is not None:
            attrs["_detect_parallelism_type"] = lambda self, _mod: detected_type
        cls = type(cls_name, (), attrs)
        inst = cls()
        inst.tp_size = tp_size
        inst.tp_rank = tp_rank
        return SimpleNamespace(mapping=inst, megatron_module=object())

    mb_mod = "megatron.bridge.models.conversion.param_mapping"
    for name in (
        "ColumnParallelMapping",
        "RowParallelMapping",
        "QKVMapping",
        "GatedMLPMapping",
        "GDNConv1dMapping",
        "GDNLinearMappingSeparate",
    ):
        assert _tp_non_zero_skips_hf_read(
            _mk_mapping(name, mb_mod, tp_size=2, tp_rank=1)
        )
        assert not _tp_non_zero_skips_hf_read(
            _mk_mapping(name, mb_mod, tp_size=2, tp_rank=0)
        )
        assert not _tp_non_zero_skips_hf_read(
            _mk_mapping(name, mb_mod, tp_size=1, tp_rank=0)
        )
        # Subclasses outside megatron.bridge.models.conversion.param_mapping never skip.
        assert not _tp_non_zero_skips_hf_read(
            _mk_mapping(name, "art.megatron.custom", tp_size=2, tp_rank=1)
        )

    assert _tp_non_zero_skips_hf_read(
        _mk_mapping("AutoMapping", mb_mod, tp_size=2, tp_rank=1, detected_type="column")
    )
    assert _tp_non_zero_skips_hf_read(
        _mk_mapping("AutoMapping", mb_mod, tp_size=2, tp_rank=1, detected_type="row")
    )
    assert not _tp_non_zero_skips_hf_read(
        _mk_mapping(
            "AutoMapping", mb_mod, tp_size=2, tp_rank=1, detected_type="replicated"
        )
    )
    assert not _tp_non_zero_skips_hf_read(
        _mk_mapping("ReplicatedMapping", mb_mod, tp_size=2, tp_rank=1)
    )
    assert not _tp_non_zero_skips_hf_read(
        _mk_mapping("RMSNorm2ZeroCenteredRMSNormMapping", mb_mod, tp_size=2, tp_rank=1)
    )


def test_partition_load_tasks_splits_dp_cp_and_edp_and_deduplicates() -> None:
    class _FakeParam:
        """Emulates ``torch.nn.Parameter.data`` returning a fresh wrapper each access."""

        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        @property
        def data(self) -> Any:
            return SimpleNamespace(arr=self._arr)

    shared_w = _FakeParam(np.zeros((4, 4), dtype=np.float32))
    dense_w = _FakeParam(np.zeros((2, 2), dtype=np.float32))
    expert_w = _FakeParam(np.zeros((8, 4), dtype=np.float32))
    d1, d2 = shared_w.data, shared_w.data
    assert d1 is not d2

    tasks = [
        SimpleNamespace(
            megatron_module=object(),
            mapping=SimpleNamespace(is_expert=False),
            param_weight=shared_w,
        ),
        # Tied weight referencing the exact same Parameter object
        SimpleNamespace(
            megatron_module=object(),
            mapping=SimpleNamespace(is_expert=False),
            param_weight=shared_w,
        ),
        SimpleNamespace(
            megatron_module=object(),
            mapping=SimpleNamespace(is_expert=False),
            param_weight=dense_w,
        ),
        SimpleNamespace(
            megatron_module=object(),
            mapping=SimpleNamespace(is_expert=True),
            param_weight=expert_w,
        ),
        # Non-local PP placeholder task
        SimpleNamespace(
            megatron_module=None,
            mapping=SimpleNamespace(is_expert=False),
            param_weight=None,
        ),
        None,
    ]

    local0, dp0, edp0 = partition_load_tasks(
        tasks, dp_cp_rank=0, dp_cp_size=4, edp_rank=0, edp_size=1
    )
    assert len(local0) == 4
    assert len(dp0) == 2  # shared_w deduplicated on id(pw) even with fresh .data
    assert edp0 == []

    local2, dp2, edp2 = partition_load_tasks(
        tasks, dp_cp_rank=2, dp_cp_size=4, edp_rank=0, edp_size=1
    )
    assert len(local2) == 1
    assert local2[0].param_weight is expert_w
    assert len(dp2) == 2
    assert edp2 == []

    local_e1, _, edp_e1 = partition_load_tasks(
        tasks, dp_cp_rank=1, dp_cp_size=4, edp_rank=1, edp_size=2
    )
    assert local_e1 == []
    assert len(edp_e1) == 1


def test_sync_tp_min_free_bytes_uses_tp_group_minimum() -> None:
    class _FakeScalarTensor:
        def __init__(self, val: int) -> None:
            self.val = int(val)

        def item(self) -> int:
            return self.val

    class _FakeDist:
        ReduceOp = SimpleNamespace(MIN="MIN")

        def __init__(self, world_size: int, peer_free: int) -> None:
            self._world_size = world_size
            self._peer_free = peer_free
            self.calls = 0

        def get_world_size(self, _group: Any) -> int:
            return self._world_size

        def all_reduce(self, buf: _FakeScalarTensor, *, op: Any, group: Any) -> None:
            assert op == "MIN"
            assert group == "tp_group"
            self.calls += 1
            buf.val = min(buf.val, self._peer_free)

    fake_torch = SimpleNamespace(
        int64="int64",
        tensor=lambda vals, **_kw: _FakeScalarTensor(vals[0]),
    )

    dist_tp1 = _FakeDist(world_size=1, peer_free=10 * 1024**3)
    assert (
        _sync_tp_min_free_bytes(
            90 * 1024**3,
            tp_group="tp_group",
            device_str="cuda:0",
            torch_mod=fake_torch,
            dist_mod=dist_tp1,
        )
        == 90 * 1024**3
    )
    assert dist_tp1.calls == 0

    dist_tp2 = _FakeDist(world_size=2, peer_free=50 * 1024**3)
    synced = _sync_tp_min_free_bytes(
        90 * 1024**3,
        tp_group="tp_group",
        device_str="cuda:0",
        torch_mod=fake_torch,
        dist_mod=dist_tp2,
    )
    assert dist_tp2.calls == 1
    assert synced == 50 * 1024**3
    assert resolve_shard_wave_size(synced, 18) == 4

    dist_none = _FakeDist(world_size=2, peer_free=1 << 62)
    assert (
        _sync_tp_min_free_bytes(
            None,
            tp_group="tp_group",
            device_str="cuda:0",
            torch_mod=fake_torch,
            dist_mod=dist_none,
        )
        is None
    )


def test_coalesced_broadcast_tensors_packs_by_dtype_and_handles_noncontiguous() -> None:
    torch = _import_torch()

    src_f32_a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    src_f32_noncontig = (
        torch.arange(6, dtype=torch.float32).reshape(3, 2).t() + 100.0
    ).t()
    # Build genuinely non-contiguous views (transposed) on both src and dst!
    src_nc = torch.arange(12, dtype=torch.float32).reshape(4, 3).t()
    assert not src_nc.is_contiguous()
    src_i16 = torch.arange(8, dtype=torch.int16).reshape(4, 2) - 5
    src_empty = torch.empty((0, 4), dtype=torch.float32)

    dst_f32_a = torch.zeros((3, 4), dtype=torch.float32)
    dst_nc_base = torch.zeros((4, 3), dtype=torch.float32)
    dst_nc = dst_nc_base.t()
    assert not dst_nc.is_contiguous()
    dst_i16 = torch.zeros((4, 2), dtype=torch.int16)
    dst_empty = torch.empty((0, 4), dtype=torch.float32)
    del src_f32_noncontig

    src_params = [src_f32_a, src_nc, src_i16, src_empty]
    dst_params = [dst_f32_a, dst_nc, dst_i16, dst_empty]

    wire_payloads: list[Any] = []

    class _FakeDistSrc:
        def broadcast(self, tensor: Any, *, src: int, group: Any) -> None:
            del src, group
            assert tensor.is_contiguous()
            wire_payloads.append(tensor.detach().clone())

    class _FakeDistDst:
        def __init__(self) -> None:
            self._idx = 0

        def broadcast(self, tensor: Any, *, src: int, group: Any) -> None:
            del src, group
            assert tensor.is_contiguous()
            tensor.copy_(wire_payloads[self._idx])
            self._idx += 1

    _coalesced_broadcast_tensors(
        src_params,
        group="g",
        src_global_rank=0,
        my_global_rank=0,
        torch_mod=torch,
        dist_mod=_FakeDistSrc(),
        bucket_bytes=1024,
    )
    # 2 float32 tensors coalesced into 1 bucket + 1 int16 tensor -> 2 broadcasts total (empty skipped).
    assert len(wire_payloads) == 2

    _coalesced_broadcast_tensors(
        dst_params,
        group="g",
        src_global_rank=0,
        my_global_rank=1,
        torch_mod=torch,
        dist_mod=_FakeDistDst(),
        bucket_bytes=1024,
    )
    for sp, dp in zip(src_params, dst_params, strict=True):
        assert torch.equal(dp, sp)

    # Single-tensor non-contiguous bucket path also writes back in-place.
    wire_single: list[Any] = []
    single_src = torch.arange(6, dtype=torch.bfloat16).reshape(3, 2).t()
    single_dst = torch.zeros((3, 2), dtype=torch.bfloat16).t()
    assert not single_src.is_contiguous() and not single_dst.is_contiguous()

    class _SingleSrc:
        def broadcast(self, tensor: Any, *, src: int, group: Any) -> None:
            del src, group
            assert tensor.is_contiguous()
            wire_single.append(tensor.detach().clone())

    class _SingleDst:
        def broadcast(self, tensor: Any, *, src: int, group: Any) -> None:
            del src, group
            tensor.copy_(wire_single[0])

    _coalesced_broadcast_tensors(
        [single_src],
        group="g",
        src_global_rank=0,
        my_global_rank=0,
        torch_mod=torch,
        dist_mod=_SingleSrc(),
    )
    _coalesced_broadcast_tensors(
        [single_dst],
        group="g",
        src_global_rank=0,
        my_global_rank=1,
        torch_mod=torch,
        dist_mod=_SingleDst(),
    )
    assert torch.equal(single_dst, single_src)


# ---------------------------------------------------------------------------
# 2. Shared synthetic multi-shard safetensors & in-process distributed harness
# ---------------------------------------------------------------------------


class _SafetensorsTestStateDict(Mapping[str, Any]):
    """Minimal safetensors-backed Mapping matching HF / ART bridge state dict contract."""

    def __init__(
        self,
        ckpt_dir: Path,
        key_to_filename: dict[str, str],
        torch_mod: Any,
    ) -> None:
        self._ckpt_dir = ckpt_dir
        self._key_to_filename = dict(key_to_filename)
        self._torch = torch_mod
        self.source = SimpleNamespace(
            path=ckpt_dir,
            key_to_filename_map=self._key_to_filename,
        )

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._key_to_filename

    def __iter__(self) -> Any:
        return iter(self._key_to_filename)

    def __len__(self) -> int:
        return len(self._key_to_filename)

    def _read_one(self, key: str) -> Any:
        from safetensors import safe_open

        if key not in self._key_to_filename:
            raise KeyError(key)
        fpath = self._ckpt_dir / self._key_to_filename[key]
        with safe_open(fpath, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, list):
            return {k: self._read_one(k) for k in key}
        return self._read_one(str(key))


class _InProcessDist:
    """Thread-safe in-process ``torch.distributed`` simulator for multi-rank tests."""

    ReduceOp = SimpleNamespace(MIN="MIN", SUM="SUM")

    def __init__(self, n_ranks: int, torch_mod: Any) -> None:
        self._world_size = n_ranks
        self._torch = torch_mod
        self._tls = threading.local()
        self._lock = threading.Lock()
        self._groups: dict[tuple[int, ...], SimpleNamespace] = {}
        self._p2p: dict[tuple[int, int], queue.Queue[Any]] = {
            (s, d): queue.Queue() for s in range(n_ranks) for d in range(n_ranks)
        }

    def set_thread_rank(self, rank: int) -> None:
        self._tls.rank = rank

    def _group_state(self, group: Any) -> tuple[tuple[int, ...], SimpleNamespace]:
        key = (
            tuple(range(self._world_size))
            if group is None
            else tuple(int(x) for x in group)
        )
        with self._lock:
            st = self._groups.get(key)
            if st is None:
                st = SimpleNamespace(
                    barrier=threading.Barrier(len(key)),
                    bcast_buf=None,
                    reduce_vals={},
                )
                self._groups[key] = st
            return key, st

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def get_rank(self, group: Any = None) -> int:
        r = int(self._tls.rank)
        if group is None:
            return r
        return tuple(group).index(r)

    def get_world_size(self, group: Any = None) -> int:
        if group is None:
            return self._world_size
        return len(group)

    def get_global_rank(self, group: Any, group_rank: int) -> int:
        return int(tuple(group)[int(group_rank)])

    def broadcast(self, tensor: Any, src: int, group: Any = None) -> None:
        _, st = self._group_state(group)
        if self.get_rank() == int(src):
            st.bcast_buf = tensor.detach().clone()
        st.barrier.wait(timeout=15.0)
        if self.get_rank() != int(src):
            tensor.copy_(st.bcast_buf)
        st.barrier.wait(timeout=15.0)

    def all_reduce(self, tensor: Any, op: Any = None, group: Any = None) -> None:
        key, st = self._group_state(group)
        r = self.get_rank()
        st.reduce_vals[r] = tensor.detach().clone()
        st.barrier.wait(timeout=15.0)
        stacked = self._torch.stack([st.reduce_vals[k] for k in key], dim=0)
        reduced = stacked.sum(dim=0) if op == self.ReduceOp.SUM else stacked.amin(dim=0)
        tensor.copy_(reduced)
        st.barrier.wait(timeout=15.0)

    def send(self, tensor: Any, dst: int, group: Any = None) -> None:
        del group
        self._p2p[(self.get_rank(), int(dst))].put(tensor.detach().clone())

    def recv(self, tensor: Any, src: int, group: Any = None) -> None:
        del group
        payload = self._p2p[(int(src), self.get_rank())].get(timeout=15.0)
        tensor.copy_(payload)


@dataclass(frozen=True)
class _SyntheticHfWeightSource:
    logical_key: str
    physical_key_options: tuple[tuple[str, ...], ...]
    kind: str = "direct"


class _SyntheticExpertTensorSlice:
    def __init__(
        self,
        tensor: Any,
        *,
        global_start: int,
        global_stop: int,
        physical_to_logical: tuple[int | None, ...] | None = None,
    ) -> None:
        self.tensor = tensor
        self.global_start = int(global_start)
        self.global_stop = int(global_stop)
        self.physical_to_logical = physical_to_logical

    def __getitem__(self, idx: int) -> Any:
        assert self.global_start <= idx < self.global_stop
        return self.tensor[idx - self.global_start]


class _SyntheticCachedStateLookup(Mapping[str, Any]):
    def __init__(
        self,
        *,
        cache: Mapping[str, Any],
        source: Mapping[str, Any],
        device: Any = None,
    ) -> None:
        self._cache = cache
        self._source = source
        self._device = device

    def __getitem__(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        ten = self._source[key]
        if self._device is not None:
            return ten.to(device=self._device, non_blocking=True)
        return ten

    def __iter__(self) -> Any:
        seen = set(self._cache)
        yield from self._cache
        for k in self._source:
            if k not in seen:
                yield k

    def __len__(self) -> int:
        return len(set(self._cache).union(self._source))


def _make_fake_bridge_runtime_module() -> ModuleType:
    def _iter_hf_param_names(hf_param: Any) -> Iterable[str]:
        if isinstance(hf_param, str):
            yield hf_param
        elif isinstance(hf_param, Mapping):
            for v in hf_param.values():
                yield from _iter_hf_param_names(v)

    def _needs_expert_slice_prefetch(task: Any) -> bool:
        m = task.mapping
        return (
            int(getattr(m, "ep_size", 1)) > 1
            and bool(getattr(m, "is_expert", False))
            and bool(getattr(m, "is_grouped_export", False))
            and isinstance(getattr(m, "hf_param", None), (str, Mapping))
        )

    def _needs_local_hf_prefetch(task: Any) -> bool:
        if task is None or task.megatron_module is None:
            return False
        if _needs_expert_slice_prefetch(task):
            return False
        m = task.mapping
        if int(getattr(m, "tp_size", 1)) <= 1:
            return True
        return int(getattr(m, "tp_rank", 0)) == 0

    def _expert_slice_range(task: Any) -> tuple[int, int]:
        m = task.mapping
        n_exp = int(task.megatron_module.config.num_moe_experts)
        per = n_exp // int(m.ep_size)
        start = int(m.ep_rank) * per
        return start, start + per

    def _planned_hf_weight_source(bridge: Any, key: str, *, task: Any = None) -> Any:
        fn = getattr(bridge, "_art_hf_weight_source", None)
        src = fn(key, task=task) if fn is not None else None
        if src is None:
            return _SyntheticHfWeightSource(
                logical_key=key, physical_key_options=((key,),)
            )
        return src

    def _select_physical_key_option(
        src: Any, state: Mapping[str, Any]
    ) -> tuple[str, ...]:
        for opt in src.physical_key_options:
            if all(k in state for k in opt):
                return opt
        raise KeyError(src.logical_key)

    def _materialize_hf_weight_source(
        bridge: Any,
        src: Any,
        state: Mapping[str, Any],
        *,
        selected_option: tuple[str, ...],
    ) -> Any:
        if src.kind == "direct":
            return state[selected_option[0]]
        if src.kind == "bridge_materialized":
            return bridge.maybe_modify_loaded_hf_weight(src.logical_key, state)
        raise RuntimeError(src.kind)

    fake_br: Any = ModuleType("art.megatron.runtime.bridge_runtime")
    fake_br.ExpertTensorSlice = _SyntheticExpertTensorSlice
    fake_br._CachedStateLookup = _SyntheticCachedStateLookup
    fake_br._iter_hf_param_names = _iter_hf_param_names
    fake_br._needs_local_hf_prefetch = _needs_local_hf_prefetch
    fake_br._needs_expert_slice_prefetch = _needs_expert_slice_prefetch
    fake_br._expert_slice_range = _expert_slice_range
    fake_br._planned_hf_weight_source = _planned_hf_weight_source
    fake_br._select_physical_key_option = _select_physical_key_option
    fake_br._materialize_hf_weight_source = _materialize_hf_weight_source
    fake_br.get_expert_parallel_layout = lambda _cfg: None
    fake_br._shared_embedding_broadcast_model = lambda m: m
    return fake_br


def _reference_load_prepared(
    bridge: Any,
    hf_pretrained: Any,
    megatron_model: list[Any],
    tasks: list[Any],
    *,
    br: Any,
    allowed_mismatched_params: list[str] | None = None,
) -> list[Any]:
    """Faithful sequential baseline matching ``bridge_runtime._sequential_load_weights_hf_to_megatron``."""
    hf_state = hf_pretrained.state
    raw_cache: dict[str, Any] = {}
    for task in tasks:
        if task is None or task.megatron_module is None:
            continue
        if br._needs_local_hf_prefetch(task):
            for key in br._iter_hf_param_names(task.mapping.hf_param):
                if key in raw_cache:
                    continue
                src = br._planned_hf_weight_source(bridge, key, task=task)
                opt = br._select_physical_key_option(src, hf_state)
                raw_cache[key] = br._materialize_hf_weight_source(
                    bridge, src, hf_state, selected_option=opt
                )
        elif br._needs_expert_slice_prefetch(task):
            start, stop = br._expert_slice_range(task)
            for key in br._iter_hf_param_names(task.mapping.hf_param):
                if key in raw_cache:
                    continue
                src = br._planned_hf_weight_source(bridge, key, task=task)
                opt = br._select_physical_key_option(src, hf_state)
                full = br._materialize_hf_weight_source(
                    bridge, src, hf_state, selected_option=opt
                )
                raw_cache[key] = br.ExpertTensorSlice(
                    full[start:stop], global_start=start, global_stop=stop
                )
    cached_state = br._CachedStateLookup(cache=raw_cache, source=hf_state)
    for task in tasks:
        if task is None or task.megatron_module is None:
            continue
        hf_param = task.mapping.hf_param
        if (
            isinstance(hf_param, str)
            and hf_param in raw_cache
            and hf_param not in hf_state
        ):
            hf_weights = raw_cache[hf_param]
        else:
            hf_weights = bridge.maybe_modify_loaded_hf_weight(hf_param, cached_state)
        converted = task.mapping.hf_to_megatron(hf_weights, task.megatron_module)
        if converted is None:
            continue
        if converted.shape != task.param_weight.shape:
            ok = bool(allowed_mismatched_params) and any(
                fnmatch.fnmatch(task.mapping.megatron_param, pat)
                or fnmatch.fnmatch(task.param_name, pat)
                for pat in (allowed_mismatched_params or ())
            )
            if not ok:
                raise ValueError(f"mismatch {task.param_name}")
            continue
        task.param_weight.data.copy_(converted)
    bridge._broadcast_shared_embeddings(megatron_model)
    return megatron_model


def _write_synthetic_multishard_checkpoint(
    ckpt_dir: Path,
    torch_mod: Any,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Write a deterministic 4-shard safetensors checkpoint exercising all loader paths."""
    from safetensors.torch import save_file

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gen = torch_mod.Generator().manual_seed(20261127)

    def _rand(shape: tuple[int, ...], dtype: Any, scale: float = 1.0) -> Any:
        return (
            torch_mod.randn(shape, generator=gen, dtype=torch_mod.float32) * scale
        ).to(dtype)

    shards: dict[str, dict[str, Any]] = {
        "shard-0.safetensors": {
            "model.norm.weight": _rand((8,), torch_mod.bfloat16, 0.5),
            "model.layers.0.self_attn.q_proj.weight": _rand(
                (12, 8), torch_mod.bfloat16, 0.25
            ),
            "model.layers.0.mlp.gate_proj.weight": _rand(
                (8, 6), torch_mod.float32, 0.3
            ),
        },
        "shard-1.safetensors": {
            "phys.shared_src.weight": _rand((8, 6), torch_mod.bfloat16, 0.4),
            "phys.mat_a.weight": _rand((6, 4), torch_mod.float32, 0.5),
            "phys.mat_b.weight": _rand((6, 4), torch_mod.float32, 0.25),
        },
        "shard-2.safetensors": {
            "model.layers.0.mlp.up_proj.weight": _rand((8, 6), torch_mod.float32, 0.35),
            # 4 global experts, each (8, 6) -> sliced across EP=1 or EP=2
            "model.layers.0.mlp.experts.gate_up_proj": _rand(
                (4, 8, 6), torch_mod.float32, 0.2
            ),
        },
        "shard-3.safetensors": {
            "model.embed_tokens.weight": _rand((16, 8), torch_mod.bfloat16, 0.6),
            # 4 global experts, each (6, 8)
            "model.layers.0.mlp.experts.down_proj": _rand(
                (4, 6, 8), torch_mod.bfloat16, 0.2
            ),
            "model.ignored_mismatch.weight": _rand((5, 5), torch_mod.float32, 1.0),
        },
    }

    key_to_filename: dict[str, str] = {}
    all_tensors: dict[str, Any] = {}
    for fname, tensors in shards.items():
        save_file(tensors, str(ckpt_dir / fname))
        for k, t in tensors.items():
            key_to_filename[k] = fname
            all_tensors[k] = t
    return key_to_filename, all_tensors


def _build_synthetic_bridge_and_model(
    *,
    torch_mod: Any,
    dist_mod: Any | None,
    hf_weight_source_cls: Any,
    tp_size: int,
    tp_rank: int,
    tp_group: Any,
    ep_size: int,
    ep_rank: int,
    etp_size: int,
    etp_rank: int,
    etp_group: Any,
    on_modify_hook: Any = None,
) -> tuple[Any, Any]:
    """Build a ``torch.nn.Module`` + synthetic bridge covering dense, alias, materialized, tied & expert tasks."""
    num_experts = 4
    local_experts = num_experts // ep_size

    class _SyntheticMegatronModel(torch_mod.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(num_moe_experts=num_experts)
            # Non-zero sentinels ensure any unwritten parameter immediately fails parity.
            self.norm_w = torch_mod.nn.Parameter(
                torch_mod.full((8,), -99.0, dtype=torch_mod.bfloat16)
            )
            self.q_proj_w = torch_mod.nn.Parameter(
                torch_mod.full((12 // tp_size, 8), -99.0, dtype=torch_mod.bfloat16)
            )
            self.gate_up_w = torch_mod.nn.Parameter(
                torch_mod.full((16 // tp_size, 6), -99.0, dtype=torch_mod.float32)
            )
            self.alias_a_w = torch_mod.nn.Parameter(
                torch_mod.full((8 // tp_size, 6), -99.0, dtype=torch_mod.bfloat16)
            )
            self.alias_b_w = torch_mod.nn.Parameter(
                torch_mod.full((8, 6), -99.0, dtype=torch_mod.bfloat16)
            )
            self.mat_w = torch_mod.nn.Parameter(
                torch_mod.full((6, 4), -99.0, dtype=torch_mod.float32)
            )
            self.embed_w = torch_mod.nn.Parameter(
                torch_mod.full((16 // tp_size, 8), -99.0, dtype=torch_mod.bfloat16)
            )
            self.exp_gate_up_w = torch_mod.nn.Parameter(
                torch_mod.full(
                    (local_experts, 8 // etp_size, 6), -99.0, dtype=torch_mod.float32
                )
            )
            self.exp_down_w = torch_mod.nn.Parameter(
                torch_mod.full(
                    (local_experts, 6 // etp_size, 8), -99.0, dtype=torch_mod.bfloat16
                )
            )
            self.ignored_mismatch_w = torch_mod.nn.Parameter(
                torch_mod.full((3, 3), 7.25, dtype=torch_mod.float32)
            )

    def _scatter_dim(
        full_tensor: Any,
        *,
        dim: int,
        out_shape: tuple[int, ...],
        out_dtype: Any,
        t_size: int,
        t_rank: int,
        t_group: Any,
    ) -> Any:
        if t_size == 1:
            return full_tensor.to(dtype=out_dtype).contiguous()
        assert dist_mod is not None
        out = torch_mod.empty(out_shape, dtype=out_dtype)
        src_global = dist_mod.get_global_rank(group=t_group, group_rank=0)
        if t_rank == 0:
            chunks = list(
                torch_mod.chunk(full_tensor.to(dtype=out_dtype), t_size, dim=dim)
            )
            out.copy_(chunks[0])
            for peer in range(1, t_size):
                dist_mod.send(
                    chunks[peer].contiguous(),
                    dst=dist_mod.get_global_rank(group=t_group, group_rank=peer),
                    group=t_group,
                )
        else:
            dist_mod.recv(out, src=src_global, group=t_group)
        return out

    class _ReplicatedMapping:
        is_expert = False
        is_grouped_export = False

        def __init__(
            self,
            hf_param: str,
            megatron_param: str,
            *,
            scale: float = 1.0,
        ) -> None:
            self.hf_param = hf_param
            self.megatron_param = megatron_param
            self.tp_size = tp_size
            self.tp_rank = tp_rank
            self.scale = scale

        def hf_to_megatron(self, hf_weights: Any, _module: Any) -> Any:
            val = (
                (hf_weights * self.scale).to(hf_weights.dtype)
                if self.scale != 1.0
                else hf_weights
            )
            if self.tp_size == 1:
                return val
            assert dist_mod is not None
            buf = val.clone() if self.tp_rank == 0 else torch_mod.empty_like(val)
            dist_mod.broadcast(
                buf,
                src=dist_mod.get_global_rank(group=tp_group, group_rank=0),
                group=tp_group,
            )
            return buf

    class _ColParallelMapping:
        is_expert = False
        is_grouped_export = False

        def __init__(
            self,
            hf_param: Any,
            megatron_param: str,
            out_shape: tuple[int, ...],
            out_dtype: Any,
        ) -> None:
            self.hf_param = hf_param
            self.megatron_param = megatron_param
            self.tp_size = tp_size
            self.tp_rank = tp_rank
            self.out_shape = out_shape
            self.out_dtype = out_dtype

        def hf_to_megatron(self, hf_weights: Any, _module: Any) -> Any:
            if isinstance(hf_weights, Mapping):
                full = torch_mod.cat([hf_weights["gate"], hf_weights["up"]], dim=0)
            else:
                full = hf_weights
            return _scatter_dim(
                full,
                dim=0,
                out_shape=self.out_shape,
                out_dtype=self.out_dtype,
                t_size=self.tp_size,
                t_rank=self.tp_rank,
                t_group=tp_group,
            )

    class _GroupedExpertMapping:
        is_expert = True
        is_grouped_export = True

        def __init__(
            self,
            hf_param: str,
            megatron_param: str,
            out_shape: tuple[int, ...],
            out_dtype: Any,
        ) -> None:
            self.hf_param = hf_param
            self.megatron_param = megatron_param
            self.ep_size = ep_size
            self.ep_rank = ep_rank
            self.tp_size = etp_size
            self.tp_rank = etp_rank
            self.out_shape = out_shape
            self.out_dtype = out_dtype

        def hf_to_megatron(self, hf_weights: Any, _module: Any) -> Any:
            start = ep_rank * local_experts
            sliced = torch_mod.stack(
                [hf_weights[start + i] for i in range(local_experts)],
                dim=0,
            )
            return _scatter_dim(
                sliced,
                dim=1,
                out_shape=self.out_shape,
                out_dtype=self.out_dtype,
                t_size=self.tp_size,
                t_rank=self.tp_rank,
                t_group=etp_group,
            )

    class _SyntheticBridge:
        def __init__(self) -> None:
            self.shared_embed_bcast_calls = 0

        def _share_embeddings_and_output_weights(self, _config: Any) -> bool:
            return False

        def _is_adapter_param_name(self, _name: str) -> bool:
            return False

        def _with_progress_tracking(
            self, tasks: Iterable[Any], _desc: str
        ) -> Iterable[Any]:
            return tasks

        def _broadcast_shared_embeddings(self, _model: Any) -> None:
            self.shared_embed_bcast_calls += 1

        def _art_hf_weight_source(self, key: str, *, task: Any = None) -> Any:
            del task
            if key in {"logical.alias_a.weight", "logical.alias_b.weight"}:
                return hf_weight_source_cls(
                    logical_key=key,
                    physical_key_options=(("phys.shared_src.weight",),),
                    kind="direct",
                )
            if key == "logical.materialized.weight":
                return hf_weight_source_cls(
                    logical_key=key,
                    physical_key_options=(("phys.mat_a.weight", "phys.mat_b.weight"),),
                    kind="bridge_materialized",
                )
            return None

        def maybe_modify_loaded_hf_weight(
            self,
            hf_param: Any,
            hf_state_dict: Mapping[str, Any],
        ) -> Any:
            if on_modify_hook is not None:
                on_modify_hook(hf_param, hf_state_dict)
            if isinstance(hf_param, Mapping):
                return {k: hf_state_dict[v] for k, v in hf_param.items()}
            if hf_param == "logical.materialized.weight":
                return (
                    hf_state_dict["phys.mat_a.weight"]
                    + 2.0 * hf_state_dict["phys.mat_b.weight"]
                )
            if hf_param in {"logical.alias_a.weight", "logical.alias_b.weight"}:
                if hf_param in hf_state_dict:
                    return hf_state_dict[hf_param]
                return hf_state_dict["phys.shared_src.weight"]
            return hf_state_dict[hf_param]

        def build_conversion_tasks(
            self,
            _hf_pretrained: Any,
            megatron_model: list[Any],
        ) -> list[Any]:
            m = megatron_model[0]
            return [
                SimpleNamespace(
                    megatron_module=m,
                    param_name="norm_w",
                    param_weight=m.norm_w,
                    mapping=_ReplicatedMapping("model.norm.weight", "norm_w"),
                ),
                SimpleNamespace(
                    megatron_module=m,
                    param_name="q_proj_w",
                    param_weight=m.q_proj_w,
                    mapping=_ColParallelMapping(
                        "model.layers.0.self_attn.q_proj.weight",
                        "q_proj_w",
                        tuple(m.q_proj_w.shape),
                        m.q_proj_w.dtype,
                    ),
                ),
                # Multi-key dict mapping spanning shard-0 and shard-2 (cross-wave retention!)
                SimpleNamespace(
                    megatron_module=m,
                    param_name="gate_up_w",
                    param_weight=m.gate_up_w,
                    mapping=_ColParallelMapping(
                        {
                            "gate": "model.layers.0.mlp.gate_proj.weight",
                            "up": "model.layers.0.mlp.up_proj.weight",
                        },
                        "gate_up_w",
                        tuple(m.gate_up_w.shape),
                        m.gate_up_w.dtype,
                    ),
                ),
                # phys_to_logicals fan-out: TWO logical keys backed by ONE physical key in shard-1
                SimpleNamespace(
                    megatron_module=m,
                    param_name="alias_a_w",
                    param_weight=m.alias_a_w,
                    mapping=_ColParallelMapping(
                        "logical.alias_a.weight",
                        "alias_a_w",
                        tuple(m.alias_a_w.shape),
                        m.alias_a_w.dtype,
                    ),
                ),
                SimpleNamespace(
                    megatron_module=m,
                    param_name="alias_b_w",
                    param_weight=m.alias_b_w,
                    mapping=_ReplicatedMapping(
                        "logical.alias_b.weight", "alias_b_w", scale=0.5
                    ),
                ),
                # bridge_materialized source combining two physical tensors
                SimpleNamespace(
                    megatron_module=m,
                    param_name="mat_w",
                    param_weight=m.mat_w,
                    mapping=_ReplicatedMapping("logical.materialized.weight", "mat_w"),
                ),
                # Tied weight: TWO tasks sharing the exact same Parameter & HF key
                SimpleNamespace(
                    megatron_module=m,
                    param_name="embed_w",
                    param_weight=m.embed_w,
                    mapping=_ColParallelMapping(
                        "model.embed_tokens.weight",
                        "embed_w",
                        tuple(m.embed_w.shape),
                        m.embed_w.dtype,
                    ),
                ),
                SimpleNamespace(
                    megatron_module=m,
                    param_name="lm_head_tied_w",
                    param_weight=m.embed_w,
                    mapping=_ColParallelMapping(
                        "model.embed_tokens.weight",
                        "lm_head_tied_w",
                        tuple(m.embed_w.shape),
                        m.embed_w.dtype,
                    ),
                ),
                # Grouped MoE expert slices (EP=1 or EP=2, ETP=1 or ETP=2)
                SimpleNamespace(
                    megatron_module=m,
                    param_name="exp_gate_up_w",
                    param_weight=m.exp_gate_up_w,
                    mapping=_GroupedExpertMapping(
                        "model.layers.0.mlp.experts.gate_up_proj",
                        "exp_gate_up_w",
                        tuple(m.exp_gate_up_w.shape),
                        m.exp_gate_up_w.dtype,
                    ),
                ),
                SimpleNamespace(
                    megatron_module=m,
                    param_name="exp_down_w",
                    param_weight=m.exp_down_w,
                    mapping=_GroupedExpertMapping(
                        "model.layers.0.mlp.experts.down_proj",
                        "exp_down_w",
                        tuple(m.exp_down_w.shape),
                        m.exp_down_w.dtype,
                    ),
                ),
                # Whitelisted mismatched shape param
                SimpleNamespace(
                    megatron_module=m,
                    param_name="ignored_mismatch_w",
                    param_weight=m.ignored_mismatch_w,
                    mapping=_ReplicatedMapping(
                        "model.ignored_mismatch.weight", "ignored_mismatch_w"
                    ),
                ),
            ]

    return _SyntheticBridge(), _SyntheticMegatronModel()


# ---------------------------------------------------------------------------
# 3. Full multi-rank orchestrator bitwise parity & wave eviction tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dp_cp_size", "tp_size", "ep_size", "etp_size"),
    [
        # Single rank: world_size=1
        (1, 1, 1, 1),
        # 4 ranks: dp_cp=2, tp=2, ep=2, etp=1, edp=2
        (2, 2, 2, 1),
        # 4 ranks asymmetric ETP > TP: dp_cp=4, tp=1, ep=2, etp=2, edp=1
        (4, 1, 2, 2),
        # 8 ranks: dp_cp=4, tp=2, ep=2, etp=2, edp=2 (includes dp_cp_rank!=0 & edp_rank==0!)
        (4, 2, 2, 2),
    ],
)
def test_fast_weight_loader_full_orchestrator_multirank_parity_and_wave_eviction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dp_cp_size: int,
    tp_size: int,
    ep_size: int,
    etp_size: int,
) -> None:
    """Full CPU orchestrator parity test across world_size=1, TP/DP-CP/EP/ETP topologies."""
    torch = _import_torch()
    safetensors_mod = _import_safetensors()

    world_size = dp_cp_size * tp_size
    assert world_size % (ep_size * etp_size) == 0

    ckpt_dir = tmp_path / f"ckpt_{dp_cp_size}_{tp_size}_{ep_size}_{etp_size}"
    key_to_filename, _ = _write_synthetic_multishard_checkpoint(ckpt_dir, torch)

    rank_ckpt_dirs: list[Path] = []
    for r in range(world_size):
        rdir = tmp_path / f"rank_{r}_ckpt_{dp_cp_size}_{tp_size}_{ep_size}_{etp_size}"
        rdir.symlink_to(ckpt_dir)
        rank_ckpt_dirs.append(rdir)

    rank_topo: list[dict[str, Any]] = []
    for r in range(world_size):
        tp_rank = r % tp_size
        dp_cp_rank = r // tp_size
        tp_group = tuple(range(dp_cp_rank * tp_size, (dp_cp_rank + 1) * tp_size))
        dp_cp_group = tuple(range(tp_rank, world_size, tp_size))
        if etp_size == 1:
            ep_rank = r % ep_size
            etp_rank = 0
            etp_group = (r,)
            edp_rank = r // ep_size
            edp_group = tuple(range(ep_rank, world_size, ep_size))
        else:
            etp_rank = r % etp_size
            ep_rank = (r // etp_size) % ep_size
            etp_base = (r // etp_size) * etp_size
            etp_group = tuple(range(etp_base, etp_base + etp_size))
            edp_stride = ep_size * etp_size
            edp_rank = r // edp_stride
            edp_group = tuple(range(r % edp_stride, world_size, edp_stride))
        rank_topo.append(
            {
                "tp_rank": tp_rank,
                "tp_group": tp_group,
                "dp_cp_rank": dp_cp_rank,
                "dp_cp_group": dp_cp_group,
                "ep_rank": ep_rank,
                "etp_rank": etp_rank,
                "etp_group": etp_group,
                "edp_rank": edp_rank,
                "edp_group": edp_group,
            }
        )

    fake_br = _make_fake_bridge_runtime_module()

    def _run_all_ranks(use_fast: bool) -> tuple[list[Any], dict[str, Any]]:
        dist_sim = _InProcessDist(world_size, torch)
        fake_ps: Any = ModuleType("megatron.core.parallel_state")
        fake_ps.get_data_parallel_group = lambda with_context_parallel=True: rank_topo[
            dist_sim.get_rank()
        ]["dp_cp_group"]
        fake_ps.get_expert_data_parallel_group = lambda check_initialized=False: (
            rank_topo[dist_sim.get_rank()]["edp_group"]
        )
        fake_ps.get_tensor_model_parallel_group = lambda check_initialized=False: (
            rank_topo[dist_sim.get_rank()]["tp_group"]
        )
        fake_ps.get_expert_tensor_parallel_group = lambda check_initialized=False: (
            rank_topo[dist_sim.get_rank()]["etp_group"]
        )

        opens_by_rank: dict[int, list[str]] = {r: [] for r in range(world_size)}
        opens_lock = threading.Lock()
        orig_safe_open = safetensors_mod.safe_open

        def _counting_safe_open(fpath: Any, *args: Any, **kwargs: Any) -> Any:
            p = Path(fpath)
            parent_name = p.parent.name
            if parent_name.startswith("rank_"):
                rank_idx = int(parent_name.split("_")[1])
                with opens_lock:
                    opens_by_rank[rank_idx].append(p.name)
            return orig_safe_open(fpath, *args, **kwargs)

        rank0_obs: dict[str, Any] = {
            "wave1_norm_ref": None,
            "wave1_norm_alive_in_wave3_dense": None,
            "wave3_dense_cache_keys": None,
            "wave3_expert_cache_keys": None,
            "raw_cache_refs": [],
        }

        def _rank0_modify_hook(hf_param: Any, cached_state: Any) -> None:
            raw_cache = getattr(cached_state, "_cache", None)
            if raw_cache is None:
                return
            if not any(rc is raw_cache for rc in rank0_obs["raw_cache_refs"]):
                rank0_obs["raw_cache_refs"].append(raw_cache)
            if hf_param == "model.norm.weight":
                ten = raw_cache.get("model.norm.weight")
                if ten is not None:
                    rank0_obs["wave1_norm_ref"] = weakref.ref(ten)
            elif isinstance(hf_param, Mapping) and "gate" in hf_param:
                # Executes in wave 3 (shard-2) of the dense phase:
                # wave-1/wave-2 single-shard keys must ALREADY be popped & GC'd,
                # while cross-shard gate_proj (wave 1) stays cached alongside up_proj (wave 3)!
                rank0_obs["wave3_dense_cache_keys"] = set(raw_cache.keys())
                gc.collect()
                if rank0_obs["wave1_norm_ref"] is not None:
                    rank0_obs["wave1_norm_alive_in_wave3_dense"] = (
                        rank0_obs["wave1_norm_ref"]() is not None
                    )
            elif hf_param == "model.layers.0.mlp.experts.gate_up_proj":
                rank0_obs["wave3_expert_cache_keys"] = set(raw_cache.keys())

        with monkeypatch.context() as mp:
            mp.setenv("ART_MEGATRON_FAST_WEIGHT_LOAD_WAVE_SIZE", "1")
            mp.setenv("ART_MEGATRON_FAST_WEIGHT_LOAD_WORKERS", "2")
            mp.setattr(torch.cuda, "is_available", lambda: False)
            mp.setattr(torch, "distributed", dist_sim)
            mp.setitem(sys.modules, "torch.distributed", dist_sim)
            mp.setitem(sys.modules, "art.megatron.runtime.bridge_runtime", fake_br)
            mp.setitem(sys.modules, "megatron", ModuleType("megatron"))
            mp.setitem(sys.modules, "megatron.core", ModuleType("megatron.core"))
            mp.setitem(sys.modules, "megatron.core.parallel_state", fake_ps)
            mp.setattr(safetensors_mod, "safe_open", _counting_safe_open)

            models: list[Any] = [None] * world_size
            bridges: list[Any] = [None] * world_size
            errors: list[BaseException] = []

            def _worker(r: int) -> None:
                try:
                    dist_sim.set_thread_rank(r)
                    topo = rank_topo[r]
                    bridge, model = _build_synthetic_bridge_and_model(
                        torch_mod=torch,
                        dist_mod=dist_sim if world_size > 1 else None,
                        hf_weight_source_cls=_SyntheticHfWeightSource,
                        tp_size=tp_size,
                        tp_rank=topo["tp_rank"],
                        tp_group=topo["tp_group"],
                        ep_size=ep_size,
                        ep_rank=topo["ep_rank"],
                        etp_size=etp_size,
                        etp_rank=topo["etp_rank"],
                        etp_group=topo["etp_group"],
                        on_modify_hook=(
                            _rank0_modify_hook if (use_fast and r == 0) else None
                        ),
                    )
                    state = _SafetensorsTestStateDict(
                        rank_ckpt_dirs[r], key_to_filename, torch
                    )
                    hf_pretrained = SimpleNamespace(
                        model_name_or_path="synthetic/qwen-moe",
                        state=state,
                    )
                    tasks = bridge.build_conversion_tasks(hf_pretrained, [model])
                    if use_fast:
                        fast_load_prepared_weights_hf_to_megatron(
                            bridge,
                            hf_pretrained,
                            [model],
                            tasks,
                            allowed_mismatched_params=["*ignored_mismatch*"],
                            key_to_filename=key_to_filename,
                            source_path=rank_ckpt_dirs[r],
                        )
                    else:
                        _reference_load_prepared(
                            bridge,
                            hf_pretrained,
                            [model],
                            tasks,
                            br=fake_br,
                            allowed_mismatched_params=["*ignored_mismatch*"],
                        )
                    models[r] = model
                    bridges[r] = bridge
                except BaseException as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=_worker, args=(r,), daemon=True)
                for r in range(world_size)
            ]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=20.0)
                assert not th.is_alive(), (
                    "Simulated multi-rank load timed out (deadlock!)"
                )
            if errors:
                raise errors[0]

        return models, {
            "opens_by_rank": opens_by_rank,
            "rank0_obs": rank0_obs,
            "bridges": bridges,
        }

    ref_models, _ = _run_all_ranks(use_fast=False)
    fast_models, meta = _run_all_ranks(use_fast=True)

    # 1. Bitwise `torch.equal` parity on EVERY parameter across EVERY rank.
    for r in range(world_size):
        ref_params = dict(ref_models[r].named_parameters())
        fast_params = dict(fast_models[r].named_parameters())
        assert set(fast_params) == set(ref_params)
        for pname, ref_p in ref_params.items():
            assert torch.equal(fast_params[pname].data, ref_p.data), (
                f"Rank {r} param {pname!r} diverged between fast and reference loader"
            )
        assert meta["bridges"][r].shared_embed_bcast_calls == 1

    # 2. Structural sanity: replicas match, TP splits differ, disjoint EP slices differ.
    for r in range(world_size):
        topo = rank_topo[r]
        src_dp = topo["dp_cp_group"][0]
        assert torch.equal(fast_models[r].norm_w.data, fast_models[src_dp].norm_w.data)
        assert torch.equal(
            fast_models[r].q_proj_w.data, fast_models[src_dp].q_proj_w.data
        )
        assert torch.equal(
            fast_models[r].alias_a_w.data, fast_models[src_dp].alias_a_w.data
        )
        assert torch.equal(
            fast_models[r].alias_b_w.data, fast_models[src_dp].alias_b_w.data
        )
        assert torch.equal(fast_models[r].mat_w.data, fast_models[src_dp].mat_w.data)
        assert torch.equal(
            fast_models[r].embed_w.data, fast_models[src_dp].embed_w.data
        )
        src_edp = topo["edp_group"][0]
        assert torch.equal(
            fast_models[r].exp_gate_up_w.data, fast_models[src_edp].exp_gate_up_w.data
        )
        assert torch.equal(
            fast_models[r].exp_down_w.data, fast_models[src_edp].exp_down_w.data
        )
    if tp_size > 1:
        assert not torch.equal(
            fast_models[0].q_proj_w.data, fast_models[1].q_proj_w.data
        )
    if ep_size > 1:
        ep0_rank = next(
            r
            for r in range(world_size)
            if rank_topo[r]["ep_rank"] == 0 and rank_topo[r]["etp_rank"] == 0
        )
        ep1_rank = next(
            r
            for r in range(world_size)
            if rank_topo[r]["ep_rank"] == 1 and rank_topo[r]["etp_rank"] == 0
        )
        assert not torch.equal(
            fast_models[ep0_rank].exp_gate_up_w.data,
            fast_models[ep1_rank].exp_gate_up_w.data,
        )
        assert not torch.equal(
            fast_models[ep0_rank].exp_down_w.data,
            fast_models[ep1_rank].exp_down_w.data,
        )

    # 3. Intra-phase multi-wave eviction & immediate weakref tensor release!
    obs = meta["rank0_obs"]
    assert obs["wave3_dense_cache_keys"] is not None
    # Wave-1 single-shard keys (norm, q_proj) and wave-2 alias keys were popped before wave 3!
    assert "model.norm.weight" not in obs["wave3_dense_cache_keys"]
    assert "model.layers.0.self_attn.q_proj.weight" not in obs["wave3_dense_cache_keys"]
    assert "logical.alias_a.weight" not in obs["wave3_dense_cache_keys"]
    assert "logical.alias_b.weight" not in obs["wave3_dense_cache_keys"]
    # Cross-shard gate_proj (wave 1) stayed cached until wave 3's up_proj arrived!
    assert "model.layers.0.mlp.gate_proj.weight" in obs["wave3_dense_cache_keys"]
    assert "model.layers.0.mlp.up_proj.weight" in obs["wave3_dense_cache_keys"]
    # Wave-1 norm tensor had zero lingering references by wave 3 (del d_out, e_out, ten)!
    assert obs["wave1_norm_alive_in_wave3_dense"] is False
    assert obs["wave3_expert_cache_keys"] is not None
    assert "model.norm.weight" not in obs["wave3_expert_cache_keys"]
    # All raw_cache dicts cleared at completion.
    assert all(rc == {} for rc in obs["raw_cache_refs"])

    # 4. Zero redundant disk I/O on non-source replica ranks!
    opens = meta["opens_by_rank"]
    for r in range(world_size):
        topo = rank_topo[r]
        if topo["dp_cp_rank"] > 0 and topo["edp_rank"] > 0:
            assert opens[r] == [], (
                f"Replica rank {r} should do zero disk I/O, got {opens[r]}"
            )
        elif topo["dp_cp_rank"] > 0 and topo["edp_rank"] == 0:
            assert set(opens[r]) == {"shard-2.safetensors", "shard-3.safetensors"}
        elif topo["dp_cp_rank"] == 0 and topo["tp_rank"] == 0:
            assert set(opens[r]) == {
                "shard-0.safetensors",
                "shard-1.safetensors",
                "shard-2.safetensors",
                "shard-3.safetensors",
            }


# ---------------------------------------------------------------------------
# 4. Unfused per-expert shards + TP>1/EP>1 lockstep safety (two-phase waves)
# ---------------------------------------------------------------------------


def test_fast_weight_loader_unfused_per_expert_shards_tp_ep_no_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify TP=2, EP=2, ETP=1 with disjoint per-expert shards executes in lockstep without deadlock or corruption."""
    torch = _import_torch()
    _import_safetensors()
    from safetensors.torch import save_file

    ckpt_dir = tmp_path / "unfused_moe"
    ckpt_dir.mkdir(parents=True)
    gen = torch.Generator().manual_seed(999)

    def _r(shape: tuple[int, ...]) -> Any:
        return torch.randn(shape, generator=gen, dtype=torch.float32)

    # Deliberately place Expert 0 in shard-0 (before dense-a/dense-b) and Expert 1 in shard-3
    # so rank 0 (ep=0) and rank 1 (ep=1) would diverge in wave indices if dense & expert were mixed!
    shards = {
        "shard-0-exp0.safetensors": {"model.experts.0.w": _r((4, 4))},
        "shard-1-dense-a.safetensors": {"model.dense_a.w": _r((8, 4))},
        "shard-2-dense-b.safetensors": {"model.dense_b.w": _r((8, 4))},
        "shard-3-exp1.safetensors": {"model.experts.1.w": _r((4, 4))},
    }
    key_to_filename: dict[str, str] = {}
    for fname, tensors in shards.items():
        save_file(tensors, str(ckpt_dir / fname))
        for k in tensors:
            key_to_filename[k] = fname

    world_size = 4  # dp_cp=2, tp=2, ep=2, etp=1, edp=2
    rank_topo: list[dict[str, Any]] = [
        {
            "tp_rank": r % 2,
            "tp_group": (0, 1) if r < 2 else (2, 3),
            "dp_cp_rank": r // 2,
            "dp_cp_group": (0, 2) if r % 2 == 0 else (1, 3),
            "ep_rank": r % 2,
            "etp_group": (r,),
            "edp_rank": r // 2,
            "edp_group": (0, 2) if r % 2 == 0 else (1, 3),
        }
        for r in range(world_size)
    ]
    fake_br = _make_fake_bridge_runtime_module()

    def _run(use_fast: bool) -> list[Any]:
        dist_sim = _InProcessDist(world_size, torch)
        fake_ps: Any = ModuleType("megatron.core.parallel_state")
        fake_ps.get_data_parallel_group = lambda with_context_parallel=True: rank_topo[
            dist_sim.get_rank()
        ]["dp_cp_group"]
        fake_ps.get_expert_data_parallel_group = lambda check_initialized=False: (
            rank_topo[dist_sim.get_rank()]["edp_group"]
        )
        fake_ps.get_tensor_model_parallel_group = lambda check_initialized=False: (
            rank_topo[dist_sim.get_rank()]["tp_group"]
        )
        fake_ps.get_expert_tensor_parallel_group = lambda check_initialized=False: (
            rank_topo[dist_sim.get_rank()]["etp_group"]
        )

        class _DenseTPMap:
            is_expert = False
            is_grouped_export = False

            def __init__(self, hf_key: str, mname: str, tp_rank: int, tp_group: Any):
                self.hf_param = hf_key
                self.megatron_param = mname
                self.tp_size = 2
                self.tp_rank = tp_rank
                self._tp_group = tp_group

            def hf_to_megatron(self, hf_w: Any, _mod: Any) -> Any:
                out = torch.empty((4, 4), dtype=torch.float32)
                src = dist_sim.get_global_rank(self._tp_group, 0)
                if self.tp_rank == 0:
                    c0, c1 = torch.chunk(hf_w, 2, dim=0)
                    out.copy_(c0)
                    dist_sim.send(
                        c1.contiguous(),
                        dst=dist_sim.get_global_rank(self._tp_group, 1),
                        group=self._tp_group,
                    )
                else:
                    dist_sim.recv(out, src=src, group=self._tp_group)
                return out

        class _UnfusedExpMap:
            is_expert = True
            is_grouped_export = False

            def __init__(self, hf_key: str, mname: str):
                self.hf_param = hf_key
                self.megatron_param = mname
                self.tp_size = 1
                self.tp_rank = 0
                self.ep_size = 2

            def hf_to_megatron(self, hf_w: Any, _mod: Any) -> Any:
                return hf_w

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dense_a = torch.nn.Parameter(torch.full((4, 4), -1.0))
                self.dense_b = torch.nn.Parameter(torch.full((4, 4), -1.0))
                self.exp_local = torch.nn.Parameter(torch.full((4, 4), -1.0))

        class _Bridge:
            def maybe_modify_loaded_hf_weight(self, hf_param: str, state: Any) -> Any:
                return state[hf_param]

            def _broadcast_shared_embeddings(self, _model: Any) -> None:
                pass

        with monkeypatch.context() as mp:
            mp.setenv("ART_MEGATRON_FAST_WEIGHT_LOAD_WAVE_SIZE", "1")
            mp.setattr(torch.cuda, "is_available", lambda: False)
            mp.setattr(torch, "distributed", dist_sim)
            mp.setitem(sys.modules, "torch.distributed", dist_sim)
            mp.setitem(sys.modules, "art.megatron.runtime.bridge_runtime", fake_br)
            mp.setitem(sys.modules, "megatron", ModuleType("megatron"))
            mp.setitem(sys.modules, "megatron.core", ModuleType("megatron.core"))
            mp.setitem(sys.modules, "megatron.core.parallel_state", fake_ps)

            models: list[Any] = [None] * world_size
            errors: list[BaseException] = []

            def _worker(r: int) -> None:
                try:
                    dist_sim.set_thread_rank(r)
                    topo = rank_topo[r]
                    m = _Model()
                    b = _Bridge()
                    ep_idx = topo["ep_rank"]
                    tasks = [
                        SimpleNamespace(
                            megatron_module=m,
                            param_name="dense_a",
                            param_weight=m.dense_a,
                            mapping=_DenseTPMap(
                                "model.dense_a.w",
                                "dense_a",
                                topo["tp_rank"],
                                topo["tp_group"],
                            ),
                        ),
                        SimpleNamespace(
                            megatron_module=m,
                            param_name="dense_b",
                            param_weight=m.dense_b,
                            mapping=_DenseTPMap(
                                "model.dense_b.w",
                                "dense_b",
                                topo["tp_rank"],
                                topo["tp_group"],
                            ),
                        ),
                        SimpleNamespace(
                            megatron_module=m,
                            param_name="exp_local",
                            param_weight=m.exp_local,
                            mapping=_UnfusedExpMap(
                                f"model.experts.{ep_idx}.w", "exp_local"
                            ),
                        ),
                    ]
                    hf_pretrained = SimpleNamespace(
                        model_name_or_path="synthetic/unfused-moe",
                        state=_SafetensorsTestStateDict(
                            ckpt_dir, key_to_filename, torch
                        ),
                    )
                    if use_fast:
                        fast_load_prepared_weights_hf_to_megatron(
                            b,
                            hf_pretrained,
                            [m],
                            tasks,
                            key_to_filename=key_to_filename,
                            source_path=ckpt_dir,
                        )
                    else:
                        _reference_load_prepared(
                            b, hf_pretrained, [m], tasks, br=fake_br
                        )
                    models[r] = m
                except BaseException as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=_worker, args=(r,), daemon=True)
                for r in range(world_size)
            ]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=15.0)
                assert not th.is_alive(), "Unfused MoE TP+EP load deadlocked!"
            if errors:
                raise errors[0]
            return models

    refs = _run(use_fast=False)
    fasts = _run(use_fast=True)
    for r in range(world_size):
        for name, ref_p in refs[r].named_parameters():
            assert torch.equal(dict(fasts[r].named_parameters())[name].data, ref_p.data)


# ---------------------------------------------------------------------------
# 5. Pipeline Parallel (PP=2) safety: disjoint stage shards + tied embeddings
# ---------------------------------------------------------------------------


def test_fast_weight_loader_pipeline_parallel_multistage_parity_and_shared_embeddings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PP=2, dp_cp=2, tp=2 (8 ranks) with stage-disjoint shards & cross-PP tied embeddings."""
    torch = _import_torch()
    safetensors_mod = _import_safetensors()
    from safetensors.torch import save_file

    ckpt_dir = tmp_path / "pp2_ckpt"
    ckpt_dir.mkdir(parents=True)
    gen = torch.Generator().manual_seed(2026)

    def _r(shape: tuple[int, ...]) -> Any:
        return torch.randn(shape, generator=gen, dtype=torch.float32)

    shards = {
        "stage0.safetensors": {
            "model.embed_tokens.weight": _r((12, 4)),
            "model.layers.0.w": _r((8, 4)),
        },
        "stage1.safetensors": {
            "model.layers.1.w": _r((8, 4)),
            "model.layers.1.experts.w": _r((4, 4)),
        },
    }
    key_to_filename: dict[str, str] = {}
    for fname, tensors in shards.items():
        save_file(tensors, str(ckpt_dir / fname))
        for k in tensors:
            key_to_filename[k] = fname

    world_size = 8  # pp=2, dp_cp=2, tp=2
    rank_ckpt_dirs: list[Path] = []
    for r in range(world_size):
        rdir = tmp_path / f"pprank_{r}"
        rdir.symlink_to(ckpt_dir)
        rank_ckpt_dirs.append(rdir)

    # Ranks 0..3 -> pp=0; ranks 4..7 -> pp=1.
    # Within each stage (base = pp * 4): tp_groups=(base+0,base+1),(base+2,base+3);
    # dp_cp_groups=(base+0,base+2),(base+1,base+3); embedding_group=(r%4, 4+r%4).
    rank_topo: list[dict[str, Any]] = []
    for r in range(world_size):
        pp_rank = r // 4
        local_r = r % 4
        base = pp_rank * 4
        tp_rank = local_r % 2
        dp_cp_rank = local_r // 2
        tp_group = (base + dp_cp_rank * 2, base + dp_cp_rank * 2 + 1)
        dp_cp_group = (base + tp_rank, base + tp_rank + 2)
        embed_group = (local_r, 4 + local_r)
        rank_topo.append(
            {
                "pp_rank": pp_rank,
                "tp_rank": tp_rank,
                "tp_group": tp_group,
                "dp_cp_rank": dp_cp_rank,
                "dp_cp_group": dp_cp_group,
                "edp_group": dp_cp_group,
                "etp_group": (r,),
                "embed_group": embed_group,
            }
        )

    fake_br = _make_fake_bridge_runtime_module()

    def _run(use_fast: bool) -> tuple[list[Any], dict[int, list[str]]]:
        dist_sim = _InProcessDist(world_size, torch)
        fake_ps: Any = ModuleType("megatron.core.parallel_state")
        fake_ps.get_data_parallel_group = lambda with_context_parallel=True: rank_topo[
            dist_sim.get_rank()
        ]["dp_cp_group"]
        fake_ps.get_expert_data_parallel_group = lambda check_initialized=False: (
            rank_topo[dist_sim.get_rank()]["edp_group"]
        )
        fake_ps.get_tensor_model_parallel_group = lambda check_initialized=False: (
            rank_topo[dist_sim.get_rank()]["tp_group"]
        )
        fake_ps.get_expert_tensor_parallel_group = lambda check_initialized=False: (
            rank_topo[dist_sim.get_rank()]["etp_group"]
        )

        opens_by_rank: dict[int, list[str]] = {r: [] for r in range(world_size)}
        opens_lock = threading.Lock()
        orig_safe_open = safetensors_mod.safe_open

        def _counting_safe_open(fpath: Any, *args: Any, **kwargs: Any) -> Any:
            p = Path(fpath)
            if p.parent.name.startswith("pprank_"):
                idx = int(p.parent.name.split("_")[1])
                with opens_lock:
                    opens_by_rank[idx].append(p.name)
            return orig_safe_open(fpath, *args, **kwargs)

        class _ColMap:
            is_expert = False
            is_grouped_export = False

            def __init__(
                self,
                hf_key: str,
                mname: str,
                out_shape: tuple[int, ...],
                tp_rank: int,
                tp_group: Any,
            ) -> None:
                self.hf_param = hf_key
                self.megatron_param = mname
                self.tp_size = 2
                self.tp_rank = tp_rank
                self._tp_group = tp_group
                self._out_shape = out_shape

            def hf_to_megatron(self, hf_w: Any, _mod: Any) -> Any:
                out = torch.empty(self._out_shape, dtype=torch.float32)
                src = dist_sim.get_global_rank(self._tp_group, 0)
                if self.tp_rank == 0:
                    c0, c1 = torch.chunk(hf_w, 2, dim=0)
                    out.copy_(c0)
                    dist_sim.send(
                        c1.contiguous(),
                        dst=dist_sim.get_global_rank(self._tp_group, 1),
                        group=self._tp_group,
                    )
                else:
                    dist_sim.recv(out, src=src, group=self._tp_group)
                return out

        class _ExpMap:
            is_expert = True
            is_grouped_export = False

            def __init__(self, hf_key: str, mname: str) -> None:
                self.hf_param = hf_key
                self.megatron_param = mname
                self.tp_size = 1
                self.tp_rank = 0
                self.ep_size = 1

            def hf_to_megatron(self, hf_w: Any, _mod: Any) -> Any:
                return hf_w

        class _Stage0Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_w = torch.nn.Parameter(torch.full((6, 4), -5.0))
                self.layer0_w = torch.nn.Parameter(torch.full((4, 4), -5.0))

        class _Stage1Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer1_w = torch.nn.Parameter(torch.full((4, 4), -5.0))
                self.exp_w = torch.nn.Parameter(torch.full((4, 4), -5.0))
                # Tied to Stage 0 embed_w via _broadcast_shared_embeddings (no direct load task!)
                self.output_w = torch.nn.Parameter(torch.full((6, 4), -5.0))

        class _PPBridge:
            def __init__(self, pp_rank: int, embed_group: tuple[int, int]) -> None:
                self._pp_rank = pp_rank
                self._embed_group = embed_group

            def maybe_modify_loaded_hf_weight(self, hf_param: str, state: Any) -> Any:
                return state[hf_param]

            def _broadcast_shared_embeddings(self, megatron_model: list[Any]) -> None:
                m = megatron_model[0]
                tensor = m.embed_w.data if self._pp_rank == 0 else m.output_w.data
                dist_sim.broadcast(
                    tensor,
                    src=self._embed_group[0],
                    group=self._embed_group,
                )

        with monkeypatch.context() as mp:
            mp.setenv("ART_MEGATRON_FAST_WEIGHT_LOAD_WAVE_SIZE", "1")
            mp.setattr(torch.cuda, "is_available", lambda: False)
            mp.setattr(torch, "distributed", dist_sim)
            mp.setitem(sys.modules, "torch.distributed", dist_sim)
            mp.setitem(sys.modules, "art.megatron.runtime.bridge_runtime", fake_br)
            mp.setitem(sys.modules, "megatron", ModuleType("megatron"))
            mp.setitem(sys.modules, "megatron.core", ModuleType("megatron.core"))
            mp.setitem(sys.modules, "megatron.core.parallel_state", fake_ps)
            mp.setattr(safetensors_mod, "safe_open", _counting_safe_open)

            models: list[Any] = [None] * world_size
            errors: list[BaseException] = []

            def _worker(r: int) -> None:
                try:
                    dist_sim.set_thread_rank(r)
                    topo = rank_topo[r]
                    b = _PPBridge(topo["pp_rank"], topo["embed_group"])
                    if topo["pp_rank"] == 0:
                        m: Any = _Stage0Model()
                        tasks = [
                            SimpleNamespace(
                                megatron_module=m,
                                param_name="embed_w",
                                param_weight=m.embed_w,
                                mapping=_ColMap(
                                    "model.embed_tokens.weight",
                                    "embed_w",
                                    (6, 4),
                                    topo["tp_rank"],
                                    topo["tp_group"],
                                ),
                            ),
                            SimpleNamespace(
                                megatron_module=m,
                                param_name="layer0_w",
                                param_weight=m.layer0_w,
                                mapping=_ColMap(
                                    "model.layers.0.w",
                                    "layer0_w",
                                    (4, 4),
                                    topo["tp_rank"],
                                    topo["tp_group"],
                                ),
                            ),
                            # Remote Stage-1 placeholder task (megatron_module=None)
                            SimpleNamespace(
                                megatron_module=None,
                                param_name="layer1_w",
                                param_weight=None,
                                mapping=_ColMap(
                                    "model.layers.1.w",
                                    "layer1_w",
                                    (4, 4),
                                    topo["tp_rank"],
                                    topo["tp_group"],
                                ),
                            ),
                        ]
                    else:
                        m = _Stage1Model()
                        tasks = [
                            # Remote Stage-0 placeholder task (megatron_module=None)
                            SimpleNamespace(
                                megatron_module=None,
                                param_name="layer0_w",
                                param_weight=None,
                                mapping=_ColMap(
                                    "model.layers.0.w",
                                    "layer0_w",
                                    (4, 4),
                                    topo["tp_rank"],
                                    topo["tp_group"],
                                ),
                            ),
                            SimpleNamespace(
                                megatron_module=m,
                                param_name="layer1_w",
                                param_weight=m.layer1_w,
                                mapping=_ColMap(
                                    "model.layers.1.w",
                                    "layer1_w",
                                    (4, 4),
                                    topo["tp_rank"],
                                    topo["tp_group"],
                                ),
                            ),
                            SimpleNamespace(
                                megatron_module=m,
                                param_name="exp_w",
                                param_weight=m.exp_w,
                                mapping=_ExpMap("model.layers.1.experts.w", "exp_w"),
                            ),
                        ]
                    hf_pretrained = SimpleNamespace(
                        model_name_or_path="synthetic/pp2",
                        state=_SafetensorsTestStateDict(
                            rank_ckpt_dirs[r], key_to_filename, torch
                        ),
                    )
                    if use_fast:
                        fast_load_prepared_weights_hf_to_megatron(
                            b,
                            hf_pretrained,
                            [m],
                            tasks,
                            key_to_filename=key_to_filename,
                            source_path=rank_ckpt_dirs[r],
                        )
                    else:
                        _reference_load_prepared(
                            b, hf_pretrained, [m], tasks, br=fake_br
                        )
                    models[r] = m
                except BaseException as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=_worker, args=(r,), daemon=True)
                for r in range(world_size)
            ]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=15.0)
                assert not th.is_alive(), "PP=2 load deadlocked!"
            if errors:
                raise errors[0]
            return models, opens_by_rank

    refs, _ = _run(use_fast=False)
    fasts, opens = _run(use_fast=True)

    # Bitwise parity on EVERY rank across both PP stages, including cross-PP tied output_w!
    for r in range(world_size):
        for name, ref_p in refs[r].named_parameters():
            assert torch.equal(dict(fasts[r].named_parameters())[name].data, ref_p.data)
    for local_r in range(4):
        assert torch.equal(
            fasts[local_r].embed_w.data, fasts[4 + local_r].output_w.data
        )

    # Stage 0 never opens stage1.safetensors; Stage 1 never opens stage0.safetensors;
    # dp_cp_rank > 0 replicas do zero disk I/O!
    assert set(opens[0]) == {"stage0.safetensors"}
    assert set(opens[4]) == {"stage1.safetensors"}
    for replica in (2, 3, 6, 7):
        assert opens[replica] == [], f"PP replica {replica} read disk: {opens[replica]}"


# ---------------------------------------------------------------------------
# 6. Real ART bridge_runtime._optimized_load_weights_hf_to_megatron parity
# ---------------------------------------------------------------------------


def test_fast_weight_loader_real_art_bridge_parity_and_opt_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare real ``bridge_runtime._optimized_load_weights_hf_to_megatron`` with fast on vs off."""
    torch = _import_torch()
    _import_safetensors()
    pytest.importorskip("megatron.bridge")

    from art.megatron.model_support.spec import HfWeightSource
    from art.megatron.runtime import _fast_weight_loader
    from art.megatron.runtime import bridge_runtime as br

    ckpt_dir = tmp_path / "art_parity_ckpt"
    key_to_filename, _ = _write_synthetic_multishard_checkpoint(ckpt_dir, torch)
    monkeypatch.setenv("ART_MEGATRON_FAST_WEIGHT_LOAD_WAVE_SIZE", "1")
    monkeypatch.setenv("ART_MEGATRON_FAST_WEIGHT_LOAD_WORKERS", "2")
    target_device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    fast_calls = 0
    orig_fast_fn = _fast_weight_loader.fast_load_prepared_weights_hf_to_megatron

    def _spy_fast(*args: Any, **kwargs: Any) -> Any:
        nonlocal fast_calls
        fast_calls += 1
        return orig_fast_fn(*args, **kwargs)

    monkeypatch.setattr(
        _fast_weight_loader, "fast_load_prepared_weights_hf_to_megatron", _spy_fast
    )

    for ep_rank in (0, 1):
        bridge_ref, model_ref = _build_synthetic_bridge_and_model(
            torch_mod=torch,
            dist_mod=None,
            hf_weight_source_cls=HfWeightSource,
            tp_size=1,
            tp_rank=0,
            tp_group=None,
            ep_size=2,
            ep_rank=ep_rank,
            etp_size=1,
            etp_rank=0,
            etp_group=None,
        )
        bridge_fast, model_fast = _build_synthetic_bridge_and_model(
            torch_mod=torch,
            dist_mod=None,
            hf_weight_source_cls=HfWeightSource,
            tp_size=1,
            tp_rank=0,
            tp_group=None,
            ep_size=2,
            ep_rank=ep_rank,
            etp_size=1,
            etp_rank=0,
            etp_group=None,
        )
        model_ref.to(target_device)
        model_fast.to(target_device)
        hf_pretrained = SimpleNamespace(
            model_name_or_path="synthetic/qwen-moe-art-parity",
            state=_SafetensorsTestStateDict(ckpt_dir, key_to_filename, torch),
        )

        # Opt-out runs sequential fallback without entering fast loader.
        monkeypatch.setenv("ART_MEGATRON_FAST_WEIGHT_LOAD", "0")
        before_calls = fast_calls
        br._optimized_load_weights_hf_to_megatron(
            bridge_ref,
            hf_pretrained,
            [model_ref],
            allowed_mismatched_params=["*ignored_mismatch*"],
        )
        assert fast_calls == before_calls

        # Default (enabled) runs native fast loader path.
        monkeypatch.setenv("ART_MEGATRON_FAST_WEIGHT_LOAD", "1")
        br._optimized_load_weights_hf_to_megatron(
            bridge_fast,
            hf_pretrained,
            [model_fast],
            allowed_mismatched_params=["*ignored_mismatch*"],
        )
        assert fast_calls == before_calls + 1

        ref_params = dict(model_ref.named_parameters())
        fast_params = dict(model_fast.named_parameters())
        assert set(fast_params) == set(ref_params)
        for pname, ref_p in ref_params.items():
            assert torch.equal(fast_params[pname].data.cpu(), ref_p.data.cpu()), (
                f"ART bridge parity mismatch on ep_rank={ep_rank} param={pname!r}"
            )
        assert (
            bridge_fast.shared_embed_bcast_calls
            == bridge_ref.shared_embed_bcast_calls
            == 1
        )
