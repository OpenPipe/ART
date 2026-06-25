from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
import torch.distributed as dist

from .range_ops import (
    range_gather,
    range_gather_head_major,
    range_reduce_sum_,
    range_reduce_sum_head_major_,
)
from .types import DkvReducePlan, KvFetchPlan, TokenRange

_DIST = cast(Any, dist)


class _Waitable(Protocol):
    def wait(self) -> Any: ...


def _launch_peer_exchange(
    *,
    recv_buffer: torch.Tensor,
    send_buffer: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group: Any,
    async_op: bool,
) -> _Waitable | None:
    # CP exchange waves are globally scheduled: every rank in the CP group must
    # enter the wave's collective in the same order, even when this rank's local
    # split sizes are all zero.
    return cast(
        _Waitable | None,
        _DIST.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        ),
    )


@dataclass
class KvFetchWork:
    packed_buffer: torch.Tensor
    recv_splits: tuple[int, ...]
    handle: _Waitable | None
    send_buffer: torch.Tensor | None = None
    stream: torch.cuda.Stream | None = None
    label: str = "kv_fetch"
    output_layout: str = "head_major"
    _wait_complete: bool = False

    def is_completed(self) -> bool:
        if self._wait_complete:
            return True
        handle_complete = True
        if self.handle is not None:
            is_completed = getattr(self.handle, "is_completed", None)
            if callable(is_completed):
                handle_complete = bool(is_completed())
        if self.stream is not None:
            return handle_complete and bool(self.stream.query())
        return handle_complete

    def wait(self) -> None:
        if self._wait_complete:
            return
        if self.handle is not None:
            self.handle.wait()
        if self.stream is not None:
            current_stream = torch.cuda.current_stream(self.packed_buffer.device)
            current_stream.wait_stream(self.stream)
        self._wait_complete = True

    def wait_post_process(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.wait()
        return _unpack_packed_tensor_per_peer(
            self.packed_buffer,
            self.recv_splits,
            output_layout=self.output_layout,
        )


@dataclass
class DkvReduceWork:
    packed_buffer: torch.Tensor | None
    handle: _Waitable | None
    send_buffer: torch.Tensor | None
    stream: torch.cuda.Stream | None
    plan: DkvReducePlan
    dk_local: torch.Tensor
    dv_local: torch.Tensor
    range_meta_cache: dict[Any, Any] | None = None
    label: str = "dkv_reduce"
    input_layout: str = "token_major"
    _wait_complete: bool = False

    def is_completed(self) -> bool:
        if self._wait_complete:
            return True
        handle_complete = True
        if self.handle is not None:
            is_completed = getattr(self.handle, "is_completed", None)
            if callable(is_completed):
                handle_complete = bool(is_completed())
        if self.stream is not None:
            return handle_complete and bool(self.stream.query())
        return handle_complete

    def wait(self) -> None:
        if self._wait_complete:
            return
        if self.handle is not None:
            self.handle.wait()
        if self.stream is not None and self.packed_buffer is not None:
            current_stream = torch.cuda.current_stream(self.packed_buffer.device)
            current_stream.wait_stream(self.stream)
        self._wait_complete = True

    def wait_post_process(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.wait()
        if self.packed_buffer is not None and int(self.packed_buffer.shape[0]) > 0:
            dk_remote, dv_remote = _unpack_packed_tensor_per_peer(
                self.packed_buffer,
                self.plan.recv_splits,
                output_layout=(
                    "head_major" if self.input_layout == "head_major" else "token_major"
                ),
            )
            flattened_ranges = tuple(
                range_
                for peer_ranges in self.plan.recv_ranges_by_peer
                for range_ in peer_ranges
                if range_.size() > 0
            )

            reduce_fn = (
                range_reduce_sum_head_major_
                if self.input_layout == "head_major"
                else range_reduce_sum_
            )
            reduce_fn(
                dk_remote
                if dk_remote.dtype == self.dk_local.dtype
                else dk_remote.to(dtype=self.dk_local.dtype),
                output_tensor=self.dk_local,
                ranges=flattened_ranges,
                range_meta_cache=self.range_meta_cache,
            )
            reduce_fn(
                dv_remote
                if dv_remote.dtype == self.dv_local.dtype
                else dv_remote.to(dtype=self.dv_local.dtype),
                output_tensor=self.dv_local,
                ranges=flattened_ranges,
                range_meta_cache=self.range_meta_cache,
            )
        return self.dk_local, self.dv_local


class A2AVCommunicator:
    def __init__(self) -> None:
        self._streams: dict[int, torch.cuda.Stream] = {}

    def _get_stream(self, tensor: torch.Tensor) -> torch.cuda.Stream | None:
        if not tensor.is_cuda:
            return None
        device_index = tensor.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        stream = self._streams.get(device_index)
        if stream is None:
            stream = torch.cuda.Stream(device=tensor.device)
            self._streams[device_index] = stream
        return stream

    def _launch_exchange(
        self,
        *,
        tensor: torch.Tensor,
        recv_buffer: torch.Tensor,
        total_send_rows: int,
        make_send_buffer: Callable[[], torch.Tensor],
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        group: Any,
        async_op: bool,
        input_layout: str,
    ) -> tuple[_Waitable | None, torch.Tensor, torch.cuda.Stream | None]:
        stream = self._get_stream(tensor) if async_op else None
        send_buffer = (
            tensor.new_empty(
                _packed_peer_tensor_shape(
                    tensor=tensor,
                    total_rows=0,
                    input_layout=input_layout,
                )
            )
            if total_send_rows <= 0
            else make_send_buffer()
        )
        if stream is None:
            return (
                _launch_peer_exchange(
                    recv_buffer=recv_buffer,
                    send_buffer=send_buffer,
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group,
                    async_op=async_op,
                ),
                send_buffer,
                None,
            )

        current_stream = torch.cuda.current_stream(tensor.device)
        stream.wait_stream(current_stream)
        send_buffer.record_stream(stream)
        recv_buffer.record_stream(stream)
        with torch.cuda.stream(stream):
            handle = _launch_peer_exchange(
                recv_buffer=recv_buffer,
                send_buffer=send_buffer,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
        return handle, send_buffer, stream

    def launch_kv_fetch(
        self,
        *,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        plan: KvFetchPlan,
        group: Any,
        async_op: bool,
        range_meta_cache: dict[Any, Any] | None = None,
        label: str = "kv_fetch",
        input_layout: str = "token_major",
        output_layout: str = "head_major",
    ) -> KvFetchWork:
        if group is None or _DIST.get_world_size(group) == 1:
            return KvFetchWork(
                packed_buffer=k_local.new_empty(
                    _packed_peer_tensor_shape(
                        tensor=k_local,
                        total_rows=0,
                        input_layout=input_layout,
                    )
                ),
                recv_splits=plan.recv_splits,
                handle=None,
                label=label,
                output_layout=output_layout,
            )

        total_send_rows = int(sum(plan.send_splits))
        total_recv_rows = int(sum(plan.recv_splits))
        recv_packed = k_local.new_empty(
            _packed_peer_tensor_shape(
                tensor=k_local,
                total_rows=total_recv_rows,
                input_layout=input_layout,
            )
        )
        input_split_sizes = [split * 2 for split in plan.send_splits]
        output_split_sizes = [split * 2 for split in plan.recv_splits]
        handle, send_buffer, stream = self._launch_exchange(
            tensor=k_local,
            recv_buffer=recv_packed,
            total_send_rows=total_send_rows,
            make_send_buffer=lambda: _pack_gathered_tensors_per_peer(
                left_tensor=k_local,
                right_tensor=v_local,
                ranges_by_peer=plan.send_ranges_by_peer,
                range_meta_cache=range_meta_cache,
                input_layout=input_layout,
            ),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
            input_layout=input_layout,
        )
        return KvFetchWork(
            packed_buffer=recv_packed,
            recv_splits=plan.recv_splits,
            handle=handle,
            send_buffer=send_buffer,
            stream=stream,
            label=label,
            output_layout=output_layout,
        )

    def launch_dkv_reduce(
        self,
        *,
        dk_remote: torch.Tensor,
        dv_remote: torch.Tensor,
        plan: DkvReducePlan,
        group: Any,
        async_op: bool,
        dk_local: torch.Tensor,
        dv_local: torch.Tensor,
        range_meta_cache: dict[Any, Any] | None = None,
        label: str = "dkv_reduce",
        input_layout: str = "token_major",
    ) -> DkvReduceWork:
        if group is None or _DIST.get_world_size(group) == 1:
            return DkvReduceWork(
                packed_buffer=None,
                handle=None,
                send_buffer=None,
                stream=None,
                plan=plan,
                dk_local=dk_local,
                dv_local=dv_local,
                range_meta_cache=range_meta_cache,
                label=label,
            )

        total_send_rows = int(sum(plan.send_splits))
        recv_total = int(sum(plan.recv_splits))
        recv_packed = dk_remote.new_empty(
            _packed_peer_tensor_shape(
                tensor=dk_remote,
                total_rows=recv_total,
                input_layout=input_layout,
            )
        )
        input_split_sizes = [split * 2 for split in plan.send_splits]
        output_split_sizes = [split * 2 for split in plan.recv_splits]
        handle, send_buffer, stream = self._launch_exchange(
            tensor=dk_remote,
            recv_buffer=recv_packed,
            total_send_rows=total_send_rows,
            make_send_buffer=lambda: _pack_split_tensors_by_peer(
                left_tensor=dk_remote,
                right_tensor=dv_remote,
                splits=plan.send_splits,
                input_layout=input_layout,
            ),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
            input_layout=input_layout,
        )
        return DkvReduceWork(
            packed_buffer=recv_packed,
            handle=handle,
            send_buffer=send_buffer,
            stream=stream,
            plan=plan,
            dk_local=dk_local,
            dv_local=dv_local,
            range_meta_cache=range_meta_cache,
            label=label,
            input_layout=input_layout,
        )


def range_gather_per_peer(
    input_tensor: torch.Tensor,
    ranges_by_peer: tuple[tuple[TokenRange, ...], ...],
    range_meta_cache: dict[Any, Any] | None = None,
) -> torch.Tensor:
    chunks = [
        range_gather(
            input_tensor,
            peer_ranges,
            range_meta_cache=range_meta_cache,
        )
        for peer_ranges in ranges_by_peer
    ]
    if not chunks:
        return input_tensor.new_empty((0, *input_tensor.shape[1:]))
    nonempty = [chunk for chunk in chunks if int(chunk.shape[0]) > 0]
    if not nonempty:
        return input_tensor.new_empty((0, *input_tensor.shape[1:]))
    return torch.cat(chunks, dim=0).contiguous()


def _pack_gathered_tensors_per_peer(
    *,
    left_tensor: torch.Tensor,
    right_tensor: torch.Tensor,
    ranges_by_peer: tuple[tuple[TokenRange, ...], ...],
    range_meta_cache: dict[Any, Any] | None = None,
    input_layout: str = "token_major",
) -> torch.Tensor:
    _validate_peer_layout(input_layout, context="gathered-pack input")
    total_rows = sum(
        range_.size() for peer_ranges in ranges_by_peer for range_ in peer_ranges
    )
    packed = left_tensor.new_empty(
        _packed_peer_tensor_shape(
            tensor=left_tensor,
            total_rows=total_rows,
            input_layout=input_layout,
        )
    )
    cursor = 0
    for peer_ranges in ranges_by_peer:
        split = sum(range_.size() for range_ in peer_ranges)
        if split <= 0:
            continue
        packed[cursor : cursor + split].copy_(
            _gather_peer_rows(
                left_tensor,
                peer_ranges,
                input_layout=input_layout,
                range_meta_cache=range_meta_cache,
            )
        )
        packed[cursor + split : cursor + split * 2].copy_(
            _gather_peer_rows(
                right_tensor,
                peer_ranges,
                input_layout=input_layout,
                range_meta_cache=range_meta_cache,
            )
        )
        cursor += split * 2
    return packed


def _pack_split_tensors_by_peer(
    *,
    left_tensor: torch.Tensor,
    right_tensor: torch.Tensor,
    splits: tuple[int, ...],
    input_layout: str = "token_major",
) -> torch.Tensor:
    _validate_peer_layout(input_layout, context="split-pack input")
    total_rows = int(sum(splits))
    packed = left_tensor.new_empty(
        _packed_peer_tensor_shape(
            tensor=left_tensor,
            total_rows=total_rows,
            input_layout=input_layout,
        )
    )
    cursor = 0
    for split in splits:
        if split <= 0:
            continue
        packed[cursor * 2 : cursor * 2 + split].copy_(
            _slice_peer_rows(left_tensor, cursor, cursor + split, layout=input_layout)
        )
        packed[cursor * 2 + split : cursor * 2 + split * 2].copy_(
            _slice_peer_rows(right_tensor, cursor, cursor + split, layout=input_layout)
        )
        cursor += split
    left_rows = _peer_row_count(left_tensor, layout=input_layout)
    right_rows = _peer_row_count(right_tensor, layout=input_layout)
    if cursor != left_rows or cursor != right_rows:
        raise RuntimeError(
            "Packed split consumed the wrong number of rows: "
            f"consumed={cursor}, left={left_rows}, right={right_rows}"
        )
    return packed


def _validate_peer_layout(layout: str, *, context: str) -> None:
    if layout not in {"token_major", "head_major"}:
        raise ValueError(f"Unsupported {context} layout: {layout}")


def _packed_peer_tensor_shape(
    *,
    tensor: torch.Tensor,
    total_rows: int,
    input_layout: str,
) -> tuple[int, ...]:
    _validate_peer_layout(input_layout, context="peer tensor input")
    if input_layout == "head_major":
        return (total_rows * 2, int(tensor.shape[0]), int(tensor.shape[2]))
    return (total_rows * 2, *tuple(int(dim) for dim in tensor.shape[1:]))


def _peer_row_count(tensor: torch.Tensor, *, layout: str) -> int:
    return int(tensor.shape[1] if layout == "head_major" else tensor.shape[0])


def _slice_peer_rows(
    tensor: torch.Tensor,
    start: int,
    end: int,
    *,
    layout: str,
) -> torch.Tensor:
    if layout == "head_major":
        return tensor[:, start:end].movedim(1, 0)
    return tensor[start:end]


def _gather_peer_rows(
    tensor: torch.Tensor,
    ranges: tuple[TokenRange, ...],
    *,
    input_layout: str,
    range_meta_cache: dict[Any, Any] | None,
) -> torch.Tensor:
    if input_layout == "head_major":
        return range_gather_head_major(
            tensor,
            ranges,
            range_meta_cache=range_meta_cache,
        ).movedim(1, 0)
    return range_gather(tensor, ranges, range_meta_cache=range_meta_cache)


def _unpack_packed_tensor_per_peer(
    packed_tensor: torch.Tensor,
    splits: tuple[int, ...],
    *,
    output_layout: str = "token_major",
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_peer_layout(output_layout, context="packed-tensor output")
    if int(packed_tensor.shape[0]) == 0:
        empty = _new_unpacked_peer_tensor(
            packed_tensor,
            total_rows=0,
            output_layout=output_layout,
        )
        return empty, empty
    total_rows = 0
    cursor = 0
    for split in splits:
        if split <= 0:
            continue
        cursor += split * 2
        total_rows += split
    if cursor != int(packed_tensor.shape[0]):
        raise RuntimeError(
            "Packed tensor unpack consumed the wrong number of rows: "
            f"consumed={cursor}, input={int(packed_tensor.shape[0])}"
        )
    left = _new_unpacked_peer_tensor(
        packed_tensor,
        total_rows=total_rows,
        output_layout=output_layout,
    )
    right = _new_unpacked_peer_tensor(
        packed_tensor,
        total_rows=total_rows,
        output_layout=output_layout,
    )
    in_cursor = 0
    out_cursor = 0
    for split in splits:
        if split <= 0:
            continue
        _copy_from_peer_rows(
            left,
            out_cursor,
            packed_tensor[in_cursor : in_cursor + split],
            output_layout=output_layout,
        )
        _copy_from_peer_rows(
            right,
            out_cursor,
            packed_tensor[in_cursor + split : in_cursor + split * 2],
            output_layout=output_layout,
        )
        in_cursor += split * 2
        out_cursor += split
    return left, right


def _new_unpacked_peer_tensor(
    packed_tensor: torch.Tensor,
    *,
    total_rows: int,
    output_layout: str,
) -> torch.Tensor:
    if output_layout == "head_major":
        return packed_tensor.new_empty(
            (packed_tensor.shape[1], total_rows, *packed_tensor.shape[2:])
        )
    return packed_tensor.new_empty((total_rows, *packed_tensor.shape[1:]))


def _copy_from_peer_rows(
    output: torch.Tensor,
    start: int,
    rows: torch.Tensor,
    *,
    output_layout: str,
) -> None:
    if output_layout == "head_major":
        output[:, start : start + int(rows.shape[0])].copy_(rows.movedim(0, 1))
    else:
        output[start : start + int(rows.shape[0])].copy_(rows)
