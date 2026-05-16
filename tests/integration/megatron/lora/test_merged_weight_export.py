from typing import Any, cast

import httpx
import torch

from art.megatron.runtime.jobs import (
    MergedWeightTransferInitInfo,
    MergedWeightTransferSpec,
)
import art.megatron.weights.merged_weight_export as export


def _spec() -> MergedWeightTransferSpec:
    return MergedWeightTransferSpec(
        init_info=MergedWeightTransferInitInfo(
            master_address="127.0.0.1",
            master_port=23456,
            rank_offset=1,
            world_size=3,
        ),
        vllm_base_url="http://runtime.test",
        served_model_name="model@7",
        nccl_so_path="/runtime/libnccl.so.2",
    )


class _OkResponse:
    def raise_for_status(self) -> None:
        return None


def test_ensure_merged_weight_transfer_group_rank_zero_initializes_runtime_and_trainer(
    monkeypatch,
) -> None:
    spec = _spec()
    calls: list[tuple[str, object]] = []

    def fake_trainer_init(init_info: dict[str, object]) -> str:
        calls.append(("trainer_init", init_info))
        return "trainer-group"

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> _OkResponse:
        calls.append(("post", (url, json, timeout)))
        return _OkResponse()

    monkeypatch.setattr(export, "trainer_init", fake_trainer_init)
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(export, "_maybe_distributed_barrier", lambda world_size: None)

    group, init_info = export.ensure_merged_weight_transfer_group(
        rank=0,
        world_size=2,
        merged_weight_transfer_group=None,
        merged_weight_transfer_init_info=None,
        spec=spec,
    )

    assert group == "trainer-group"
    assert init_info == spec.init_info
    assert sorted(calls, key=lambda item: item[0]) == [
        (
            "post",
            (
                "http://runtime.test/init_weight_transfer_engine",
                {"init_info": spec.init_info.model_dump()},
                300.0,
            ),
        ),
        (
            "trainer_init",
            {
                "master_address": "127.0.0.1",
                "master_port": 23456,
                "world_size": 3,
                "nccl_so_path": "/runtime/libnccl.so.2",
            },
        ),
    ]


def test_ensure_merged_weight_transfer_group_non_sender_skips_runtime_init(
    monkeypatch,
) -> None:
    spec = _spec()
    barriers: list[int] = []

    monkeypatch.setattr(
        export,
        "trainer_init",
        lambda init_info: (_ for _ in ()).throw(
            AssertionError("unexpected trainer_init")
        ),
    )
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected post")
        ),
    )
    monkeypatch.setattr(export, "_maybe_distributed_barrier", barriers.append)

    group, init_info = export.ensure_merged_weight_transfer_group(
        rank=1,
        world_size=2,
        merged_weight_transfer_group=None,
        merged_weight_transfer_init_info=None,
        spec=spec,
    )

    assert group is None
    assert init_info == spec.init_info
    assert barriers == []


def test_sync_merged_weights_to_vllm_non_sender_only_drains_export(
    monkeypatch,
) -> None:
    spec = _spec()
    barrier_calls: list[int] = []
    iter_passes: list[int] = []

    monkeypatch.setattr(
        export,
        "ensure_merged_weight_transfer_group",
        lambda **kwargs: (None, spec.init_info),
    )
    monkeypatch.setattr(export, "build_merged_weight_export", lambda **kwargs: object())

    def fake_iter(_weight_export: object):
        iter_passes.append(len(iter_passes) + 1)
        yield ("layer.weight", torch.zeros((2, 3), dtype=torch.float16))
        yield ("layer.bias", torch.zeros((3,), dtype=torch.float32))

    monkeypatch.setattr(export, "iter_merged_vllm_weights", fake_iter)
    monkeypatch.setattr(export, "_maybe_distributed_barrier", barrier_calls.append)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        export,
        "trainer_send_weights",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected send")
        ),
    )
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected http client")),
    )

    group, init_info = export.sync_merged_weights_to_vllm(
        bridge=object(),
        model=cast(Any, object()),
        model_support_handler=object(),
        rank=1,
        world_size=2,
        merged_weight_transfer_group=None,
        merged_weight_transfer_init_info=None,
        spec=spec,
        pause_generation=True,
    )

    assert group is None
    assert init_info == spec.init_info
    assert iter_passes == [1, 2]
    assert barrier_calls == [2]


def test_sync_merged_weights_to_vllm_sender_controls_runtime_and_sends(
    monkeypatch,
) -> None:
    spec = _spec()
    barrier_calls: list[int] = []
    sent_items: list[list[tuple[str, torch.Tensor]]] = []
    posts: list[
        tuple[str, dict[str, object] | None, dict[str, object] | None, float]
    ] = []

    monkeypatch.setattr(
        export,
        "ensure_merged_weight_transfer_group",
        lambda **kwargs: ("trainer-group", spec.init_info),
    )
    monkeypatch.setattr(export, "build_merged_weight_export", lambda **kwargs: object())

    def fake_iter(_weight_export: object):
        yield ("layer.weight", torch.zeros((2, 3), dtype=torch.float16))
        yield ("layer.bias", torch.zeros((3,), dtype=torch.float32))

    def fake_send(iterator, trainer_args):
        sent_items.append(list(iterator))
        assert trainer_args["group"] == "trainer-group"
        assert trainer_args["packed"] is True

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(
            self,
            url: str,
            *,
            json: dict[str, object] | None = None,
            params: dict[str, object] | None = None,
            timeout: float,
        ) -> _OkResponse:
            posts.append((url, json, params, timeout))
            return _OkResponse()

    monkeypatch.setattr(export, "iter_merged_vllm_weights", fake_iter)
    monkeypatch.setattr(export, "trainer_send_weights", fake_send)
    monkeypatch.setattr(export, "_maybe_distributed_barrier", barrier_calls.append)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(httpx, "Client", FakeClient)

    group, init_info = export.sync_merged_weights_to_vllm(
        bridge=object(),
        model=cast(Any, object()),
        model_support_handler=object(),
        rank=0,
        world_size=2,
        merged_weight_transfer_group=None,
        merged_weight_transfer_init_info=None,
        spec=spec,
        pause_generation=True,
    )

    assert group == "trainer-group"
    assert init_info == spec.init_info
    assert [name for name, _ in sent_items[0]] == ["layer.weight", "layer.bias"]
    assert posts == [
        ("http://runtime.test/pause", None, {"mode": "wait"}, 300.0),
        (
            "http://runtime.test/start_weight_update",
            {"is_checkpoint_format": True},
            None,
            300.0,
        ),
        (
            "http://runtime.test/update_weights",
            {
                "update_info": {
                    "names": ["layer.weight", "layer.bias"],
                    "dtype_names": ["float16", "float32"],
                    "shapes": [[2, 3], [3]],
                    "packed": True,
                    "packed_buffer_size_bytes": export.DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                    "packed_num_buffers": export.DEFAULT_PACKED_NUM_BUFFERS,
                }
            },
            None,
            600.0,
        ),
        ("http://runtime.test/finish_weight_update", None, None, 600.0),
        (
            "http://runtime.test/art/set_served_model_name",
            {"name": "model@7"},
            None,
            30.0,
        ),
        ("http://runtime.test/resume", None, None, 30.0),
    ]
    assert barrier_calls == [2]
