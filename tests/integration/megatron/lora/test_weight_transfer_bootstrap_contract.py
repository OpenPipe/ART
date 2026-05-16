from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import art.weight_transfer.nccl as nccl


def test_trainer_nccl_unique_id_round_trips_as_raw_bytes() -> None:
    payload = bytes(range(128))
    unique_id = nccl._nccl_unique_id_from_bytes(payload)
    assert nccl._nccl_unique_id_to_bytes(unique_id) == payload


def test_trainer_nccl_communicator_retains_bootstrap_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bytes(range(128))
    bootstrap_group = SimpleNamespace(
        broadcast_obj=lambda obj, src: obj if obj is not None else payload
    )
    loaded_so_paths: list[str | None] = []

    class FakeNcclLibrary:
        def __init__(self, so_file: str | None = None):
            loaded_so_paths.append(so_file)

        def get_unique_id(self):
            return nccl._nccl_unique_id_from_bytes(payload)

        def init_rank(self, world_size, unique_id, rank):
            assert world_size == 2
            assert rank == 0
            assert nccl._nccl_unique_id_to_bytes(unique_id) == payload
            return "comm"

    monkeypatch.setattr(nccl, "_BootstrapGroup", lambda **kwargs: bootstrap_group)
    monkeypatch.setattr(nccl, "_NcclLibrary", FakeNcclLibrary)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(
        nccl.TrainerNcclCommunicator,
        "all_reduce",
        lambda self, tensor, *, stream=None: None,
    )
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda *args, **kwargs: SimpleNamespace(device=torch.device("cuda:0")),
    )

    communicator = nccl.TrainerNcclCommunicator(
        host="127.0.0.1",
        port=12345,
        rank=0,
        world_size=2,
        device=0,
        nccl_so_path="/runtime/libnccl.so.2",
    )
    assert communicator._bootstrap_group is bootstrap_group
    assert loaded_so_paths == ["/runtime/libnccl.so.2"]


def test_trainer_init_passes_explicit_nccl_so_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_communicator(**kwargs):
        seen.update(kwargs)
        return "communicator"

    monkeypatch.setattr(nccl, "TrainerNcclCommunicator", fake_communicator)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    assert (
        nccl.trainer_init(
            {
                "master_address": "127.0.0.1",
                "master_port": 23456,
                "world_size": 4,
                "nccl_so_path": "/runtime/libnccl.so.2",
            }
        )
        == "communicator"
    )
    assert seen == {
        "host": "127.0.0.1",
        "port": 23456,
        "rank": 0,
        "world_size": 4,
        "device": 3,
        "nccl_so_path": "/runtime/libnccl.so.2",
    }
