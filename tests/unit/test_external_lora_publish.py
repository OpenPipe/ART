from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from safetensors.torch import load
import torch

from art.megatron.lora import LoraShardMeta
from art.megatron.tensor_snapshot import PinnedCpuSnapshotStager
from art.megatron.weights.external_lora_publish import (
    ExternalLoraManifest,
    ExternalLoraObjectRef,
    ExternalLoraPlan,
    ExternalLoraPublication,
    ExternalLoraRankCompletion,
    ExternalLoraShardPlan,
    ExternalLoraTarget,
    ExternalLoraTargetGrant,
    prepare_external_lora,
    publish_external_lora_rank,
)


class _Handler:
    key = "test"

    @staticmethod
    def to_vllm_lora_tensors(tensors, *, adapter_config):
        return tensors, adapter_config

    @staticmethod
    def to_vllm_lora_config(adapter_config):
        return adapter_config


class _Sink:
    def __init__(
        self, *, fail_index: int | None = None, wrong_plan: bool = False
    ) -> None:
        self.fail_index = fail_index
        self.wrong_plan = wrong_plan
        self.events: list[tuple[str, int | None]] = []
        self.objects: dict[int, bytes] = {}
        self.plan: ExternalLoraPlan | None = None

    def _plan(self) -> ExternalLoraPlan:
        assert self.plan is not None
        return self.plan

    def authorize(self, plan) -> ExternalLoraTargetGrant:
        self.plan = plan
        self.events.append(("authorize", None))
        return ExternalLoraTargetGrant(
            authorization_id="grant",
            target_revision="revision",
            plan_sha256="0" * 64 if self.wrong_plan else plan.sha256,
        )

    def put_shard(
        self,
        grant: ExternalLoraTargetGrant,
        shard: ExternalLoraShardPlan,
        chunks,
    ) -> ExternalLoraObjectRef:
        plan = self._plan()
        assert grant.plan_sha256 == plan.sha256
        self.events.append(("shard", shard.index))
        if shard.index == self.fail_index:
            raise RuntimeError("injected upload failure")
        payload = b"".join(chunks)
        self.objects[shard.index] = payload
        return _ref(f"shard/{plan.target.publication_id}/{shard.index}", payload)

    def complete_rank(
        self,
        grant: ExternalLoraTargetGrant,
        completion: ExternalLoraRankCompletion,
    ) -> ExternalLoraPublication:
        plan = self._plan()
        assert grant.plan_sha256 == plan.sha256
        self.events.append(("manifest", None))
        manifest = ExternalLoraManifest(
            plan=plan,
            plan_sha256=plan.sha256,
            shards=completion.shards,
        )
        payload = manifest.canonical_bytes()
        return ExternalLoraPublication(
            manifest=manifest,
            manifest_ref=_ref(f"manifest/{plan.target.publication_id}", payload),
        )

    def abort(
        self,
        grant: ExternalLoraTargetGrant,
        completion: ExternalLoraRankCompletion,
        error: str,
    ) -> None:
        assert grant.plan_sha256 == self._plan().sha256
        assert error
        self.events.append(("abort", completion.rank))
        for receipt in completion.shards:
            self.objects.pop(receipt.index, None)


def _ref(locator: str, payload: bytes) -> ExternalLoraObjectRef:
    return ExternalLoraObjectRef(
        locator=locator,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _target(*, shard_bytes: int) -> ExternalLoraTarget:
    return ExternalLoraTarget(
        tenant_id="tenant",
        run_id="run",
        operation_id="operation",
        training_session_id="session",
        publication_id="publication",
        generation_id="generation",
        model_identity="test/model",
        active_alias="run",
        runtime_fingerprint="1" * 64,
        shard_bytes=shard_bytes,
    )


def _prepared(*, shard_bytes: int = 32):
    key = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
    tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    metadata = LoraShardMeta(
        key=key,
        owner_rank=0,
        shape=tuple(tensor.shape),
        dtype_name="float32",
        manifest={"sharded": False, "shard_world_size": 1, "shard_rank": 0},
        block="layers.0",
    )
    pending = prepare_external_lora(
        target=_target(shard_bytes=shard_bytes),
        source_topology="tp1-cp1-ep1",
        local_tensors={key: tensor},
        local_metadata=[metadata],
        local_packed_tensors={},
        local_packed_metadata=[],
        handler=_Handler(),
        adapter_config={"r": 4, "target_modules": {"q_proj"}},
        exchange_device=torch.device("cpu"),
        stager=PinnedCpuSnapshotStager(),
    )
    return pending.resolve(), key, tensor


def test_external_lora_is_loadable_and_manifest_commits_last() -> None:
    prepared, key, expected = _prepared()
    sink = _Sink()

    publication = publish_external_lora_rank(prepared, sink)

    assert publication is not None
    assert sink.events[0] == ("authorize", None)
    assert sink.events[-1] == ("manifest", None)
    files = {
        item.relative_path: bytearray(item.size_bytes)
        for item in publication.manifest.plan.files
    }
    for shard in publication.manifest.plan.shards:
        payload = sink.objects[shard.index]
        files[shard.relative_path][
            shard.file_offset : shard.file_offset + shard.size_bytes
        ] = payload
    assert json.loads(files["adapter_config.json"]) == {
        "art_lora_format": "vllm",
        "r": 4,
        "target_modules": ["q_proj"],
    }
    tensors = load(bytes(files["adapter_model.safetensors"]))
    assert torch.equal(tensors[key], expected)


def test_external_lora_upload_failure_cleans_committed_shards() -> None:
    prepared, _key, _expected = _prepared(shard_bytes=16)
    sink = _Sink(fail_index=1)

    try:
        publish_external_lora_rank(prepared, sink)
    except RuntimeError as error:
        assert "injected upload failure" in str(error)
    else:
        raise AssertionError("external LoRA publication unexpectedly succeeded")

    assert sink.objects == {}
    assert ("manifest", None) not in sink.events
    assert ("abort", 0) in sink.events


def test_external_lora_rejects_changed_grant_before_writes() -> None:
    prepared, _key, _expected = _prepared()
    sink = _Sink(wrong_plan=True)

    try:
        publish_external_lora_rank(prepared, sink)
    except RuntimeError as error:
        assert "authorization changed its plan" in str(error)
    else:
        raise AssertionError("external LoRA publication accepted a changed plan")

    assert sink.events == [("authorize", None)]
    assert sink.objects == {}


def test_external_lora_nccl_exchanges_rank_owned_blocks() -> None:
    root_value = os.environ.get("ART_EXTERNAL_LORA_NCCL_DIR")
    if root_value is None:
        pytest.skip("set ART_EXTERNAL_LORA_NCCL_DIR under a two-rank torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("external LoRA NCCL conformance requires two ranks")
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl")
    try:
        tensors = {}
        metadata = []
        if rank == 0:
            for layer in range(2):
                key = f"base_model.model.layers.{layer}.self_attn.q_proj.lora_A.weight"
                tensor = torch.arange(
                    layer * 24, (layer + 1) * 24, dtype=torch.float32, device="cuda"
                ).reshape(4, 6)
                tensors[key] = tensor
                metadata.append(
                    LoraShardMeta(
                        key=key,
                        owner_rank=0,
                        shape=tuple(tensor.shape),
                        dtype_name="float32",
                        manifest={
                            "sharded": False,
                            "shard_world_size": 1,
                            "shard_rank": 0,
                        },
                        block=f"layers.{layer}",
                    )
                )
        pending = prepare_external_lora(
            target=_target(shard_bytes=32),
            source_topology="tp1-cp2-ep1",
            local_tensors=tensors,
            local_metadata=metadata,
            local_packed_tensors={},
            local_packed_metadata=[],
            handler=_Handler(),
            adapter_config={"r": 4, "target_modules": {"q_proj"}},
            exchange_device=torch.device("cuda", rank),
            stager=PinnedCpuSnapshotStager(),
        )
        prepared = pending.resolve()
        root = Path(root_value)
        root.mkdir(parents=True, exist_ok=True)
        for index, chunks in prepared.shard_payloads().items():
            (root / str(index)).write_bytes(b"".join(chunks))
        torch.distributed.barrier()
        if rank == 0:
            files = {
                item.relative_path: bytearray(item.size_bytes)
                for item in prepared.plan.files
            }
            for shard in prepared.plan.shards:
                payload = (root / str(shard.index)).read_bytes()
                files[shard.relative_path][
                    shard.file_offset : shard.file_offset + shard.size_bytes
                ] = payload
            result = load(bytes(files["adapter_model.safetensors"]))
            assert len(result) == 2
            for layer in range(2):
                key = f"base_model.model.layers.{layer}.self_attn.q_proj.lora_A.weight"
                assert torch.equal(
                    result[key],
                    torch.arange(
                        layer * 24, (layer + 1) * 24, dtype=torch.float32
                    ).reshape(4, 6),
                )
            for shard in prepared.plan.shards:
                (root / str(shard.index)).unlink()
            root.rmdir()
    finally:
        torch.distributed.destroy_process_group()
