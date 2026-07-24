import json
from pathlib import Path

import pytest
import torch

from art.megatron.model_support.lora_disk import save_vllm_lora_tensors
from art.megatron.runtime.jobs import (
    MegatronLoraPublishJob,
    dump_megatron_job,
    load_megatron_job,
)
from art.megatron.service import MegatronService
from art.megatron.weights.lora_publish import LoraPublishMetrics
from art.megatron.weights.update_replay import (
    inspect_adapter,
    validate_committed_policy_version,
    validate_replay_snapshots,
)


def _write_snapshot(
    path: Path,
    *,
    rank: int = 8,
    experts: int = 256,
    value: float = 1.0,
) -> None:
    save_vllm_lora_tensors(
        path,
        {
            (
                "base_model.model.model.language_model.layers.0."
                "mlp.experts.lora_A.weight"
            ): torch.full((experts, rank, 2), value, dtype=torch.bfloat16),
            (
                "base_model.model.model.language_model.layers.0."
                "mlp.experts.lora_B.weight"
            ): torch.full((experts, 4, rank), value, dtype=torch.bfloat16),
        },
        {
            "base_model_name_or_path": "Qwen/Qwen3.6-35B-A3B",
            "r": rank,
            "lora_alpha": rank,
        },
    )


def test_publish_metrics_use_stable_training_metric_names() -> None:
    metrics = LoraPublishMetrics(
        gather_pack_s=1.0,
        stage_to_cpu_s=2.0,
        write_s=3.0,
        total_s=6.0,
        logical_bytes=10,
        transported_bytes=12,
        tensor_count=4,
    )

    assert metrics.as_training_metrics() == {
        "time/weight_update_trainer_gather_pack_s": 1.0,
        "time/weight_update_trainer_stage_to_cpu_s": 2.0,
        "time/weight_update_trainer_write_s": 3.0,
        "time/weight_update_trainer_publish_s": 6.0,
        "weight_update/logical_bytes": 10,
        "weight_update/transported_bytes": 12,
        "weight_update/tensor_count": 4,
    }


def test_weight_update_event_metrics_exclude_control_fields() -> None:
    assert MegatronService._weight_update_metrics_from_event(
        {
            "event": "lora_ready",
            "step": 7,
            "time/weight_update_trainer_publish_s": 5.0,
            "weight_update/logical_bytes": 1024,
            "diagnostic": "ignored",
        }
    ) == {
        "time/weight_update_trainer_publish_s": 5.0,
        "weight_update/logical_bytes": 1024.0,
    }


def test_replay_snapshot_preserves_layout_bytes_and_integrity(tmp_path: Path) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_snapshot(first_path, value=1.0)
    _write_snapshot(second_path, value=2.0)

    first = inspect_adapter(first_path)
    second = inspect_adapter(second_path)
    validate_replay_snapshots(
        [first, second],
        expected_rank=8,
        expected_experts=256,
        expected_model_substring="Qwen3.6-35B-A3B",
    )

    assert first.manifest_sha256 == second.manifest_sha256
    assert first.file_sha256 != second.file_sha256
    assert first.logical_bytes == 256 * 8 * (2 + 4) * 2
    assert (
        first.transported_bytes
        == (first_path / "adapter_model.safetensors").stat().st_size
    )
    assert first.tensor_count == 2


def test_replay_snapshot_rejects_nonrepresentative_rank(tmp_path: Path) -> None:
    path = tmp_path / "rank-one"
    _write_snapshot(path, rank=1)

    with pytest.raises(ValueError, match="has rank 1; expected 8"):
        validate_replay_snapshots(
            [inspect_adapter(path)],
            expected_rank=8,
            expected_experts=256,
            expected_model_substring="Qwen3.6-35B-A3B",
        )


def test_receiver_replay_rejects_duplicate_snapshot_contents(tmp_path: Path) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    _write_snapshot(first_path, value=1.0)
    _write_snapshot(second_path, value=1.0)

    with pytest.raises(ValueError, match="unique snapshot contents"):
        validate_replay_snapshots(
            [inspect_adapter(first_path), inspect_adapter(second_path)],
            expected_rank=8,
            expected_experts=256,
            expected_model_substring="Qwen3.6-35B-A3B",
        )


def test_lora_publish_job_roundtrips_through_worker_protocol() -> None:
    job = MegatronLoraPublishJob(
        step=9,
        source_lora_path="/checkpoints/seed",
        output_lora_path="/scratch/replay/0009",
        content_version=8,
        allow_unvalidated_arch=True,
        log_path="/scratch/replay/0009.jsonl",
    )

    assert load_megatron_job(dump_megatron_job(job)) == job


def test_policy_version_must_match_committed_runtime_version() -> None:
    validate_committed_policy_version({"policy_version": 12}, expected=12)

    with pytest.raises(RuntimeError, match="committed policy 11; expected 12"):
        validate_committed_policy_version({"policy_version": 11}, expected=12)


def test_snapshot_config_is_machine_readable(tmp_path: Path) -> None:
    path = tmp_path / "snapshot"
    _write_snapshot(path)

    config = json.loads((path / "adapter_config.json").read_text())
    assert config["art_lora_format"] == "vllm"


def test_h200_replay_launcher_preserves_evidence_contract() -> None:
    root = Path(__file__).parents[2]
    launcher = (
        root / "scripts/benchmarks/lora-update-replay-h200.sky.yaml"
    ).read_text()
    service = (root / "scripts/benchmarks/service_lora_update_replay.py").read_text()
    uploader = (root / "scripts/benchmarks/upload_lora_update_replay.py").read_text()

    assert "accelerators: H200:4" in launcher
    assert "name: vivek-dev-api-keys" in launcher
    assert "--mode fixed-load" in launcher
    assert "--mode idle" in launcher
    assert 'max_model_len": 32769' in service
    assert "control_seconds" in service
    for filename in (
        "manifest.json",
        "request-trace.json",
        "tensor-manifest.json",
        "samples.jsonl",
        "summary.json",
        "stdout.log",
        "stderr.log",
    ):
        assert filename in uploader
