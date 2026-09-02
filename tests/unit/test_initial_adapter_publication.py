from pathlib import Path

import pytest

from art.megatron.optimizer_state import (
    ADAPTER_PUBLICATION_ACK,
    new_optimizer_generation,
    optimizer_adapter,
    publish_adapter_checkpoint,
    read_adapter_publication,
    read_latest_adapter_pointer,
)


def _write_adapter(path: Path, *, payload: bytes = b"adapter-weights") -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_bytes(b'{"r": 8}\n')
    (path / "adapter_model.safetensors").write_bytes(payload)


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    canonical = tmp_path / "checkpoints" / "0000"
    staging = tmp_path / "megatron_runtime" / "staging" / "publication"
    return canonical, staging


def test_publish_reuses_exact_bootstrap_step_zero(tmp_path: Path) -> None:
    canonical, staging = _paths(tmp_path)
    _write_adapter(canonical)
    _write_adapter(staging)
    generation = optimizer_adapter(
        canonical,
        0,
        training_session_id="session-1",
    )

    published = publish_adapter_checkpoint(
        staging,
        step=0,
        training_session_id="session-1",
        generation_id=generation.generation_id,
    )

    assert published == generation
    assert not staging.exists()
    assert read_adapter_publication(canonical, step=0) == generation
    assert read_latest_adapter_pointer(tmp_path) == generation

    _write_adapter(staging)
    assert (
        publish_adapter_checkpoint(
            staging,
            step=0,
            training_session_id="session-1",
            generation_id=generation.generation_id,
        )
        == generation
    )
    assert not staging.exists()


def test_publish_refuses_different_step_zero_payload(tmp_path: Path) -> None:
    canonical, staging = _paths(tmp_path)
    _write_adapter(canonical, payload=b"A" * 64)
    _write_adapter(staging, payload=b"B" * 64)
    generation = optimizer_adapter(
        canonical,
        0,
        training_session_id="session-1",
    )

    with pytest.raises(RuntimeError, match="different payload"):
        publish_adapter_checkpoint(
            staging,
            step=0,
            training_session_id="session-1",
            generation_id=generation.generation_id,
        )

    assert staging.exists()
    assert not (canonical / ADAPTER_PUBLICATION_ACK).exists()
    assert read_latest_adapter_pointer(tmp_path) is None


def test_publish_refuses_unacknowledged_step_zero_identity_change(
    tmp_path: Path,
) -> None:
    canonical, staging = _paths(tmp_path)
    _write_adapter(canonical)
    _write_adapter(staging)

    with pytest.raises(RuntimeError, match="different generation"):
        publish_adapter_checkpoint(
            staging,
            step=0,
            training_session_id="session-1",
            generation_id=new_optimizer_generation(0),
        )

    assert staging.exists()
    assert not (canonical / ADAPTER_PUBLICATION_ACK).exists()
    assert read_latest_adapter_pointer(tmp_path) is None
