from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from art import Model
from art.serverless.backend import ServerlessBackend
from art.utils import s3, s3_checkpoint_utils, wandb_sdk


def _source_artifact(filename: str, content: str, step: int = 10) -> MagicMock:
    artifact = MagicMock()
    artifact.aliases = [f"step{step}"]

    def download(*, root: str) -> None:
        Path(root, filename).write_text(content)

    artifact.download.side_effect = download
    return artifact


@pytest.fixture
def fork_io(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with patch("art.serverless.backend.Client") as client_cls:
        client_cls.return_value.api_key = "test-key"
        client_cls.return_value.base_url = "http://serverless.test/v1"
        backend = ServerlessBackend(api_key="test-key")

    api = MagicMock()
    api.default_entity = "default-entity"
    api.artifacts.return_value = [_source_artifact("adapter.json", "checkpoint")]
    api.run.return_value.config = {"wandb.provenance": ["original-training-run"]}
    run = MagicMock()
    destination = MagicMock()
    uploads: list[dict[str, str]] = []
    directories: list[Path] = []

    def add_dir(root: str) -> None:
        directory = Path(root)
        directories.append(directory)
        uploads.append(
            {
                str(path.relative_to(directory)): path.read_text()
                for path in directory.rglob("*")
                if path.is_file()
            }
        )

    destination.add_dir.side_effect = add_dir

    def finish() -> None:
        assert directories[-1].is_dir()

    run.finish.side_effect = finish
    init = MagicMock(return_value=run)
    make_artifact = MagicMock(return_value=destination)
    monkeypatch.setattr(wandb_sdk, "api", MagicMock(return_value=api))
    monkeypatch.setattr(wandb_sdk, "login", MagicMock())
    monkeypatch.setattr(wandb_sdk, "init", init)
    monkeypatch.setattr(wandb_sdk, "settings", MagicMock())
    monkeypatch.setattr(wandb_sdk, "artifact", make_artifact)
    model = Model(name="destination", project="destination-project", entity="target")
    model_run = MagicMock()
    model_run.config = {}
    monkeypatch.setattr(model, "_get_wandb_run", lambda: model_run)
    return SimpleNamespace(
        backend=backend,
        model=model,
        api=api,
        run=run,
        init=init,
        destination=destination,
        make_artifact=make_artifact,
        model_run=model_run,
        uploads=uploads,
        directories=directories,
    )


async def test_fork_checkpoint_uses_source_entity_for_artifacts_and_provenance(
    fork_io: SimpleNamespace,
) -> None:
    selected = _source_artifact("adapter.json", "selected", step=10)
    newer = _source_artifact("adapter.json", "too new", step=20)
    fork_io.api.artifacts.return_value = [newer, selected]

    await fork_io.backend._experimental_fork_checkpoint(
        fork_io.model,
        "source-model",
        from_project="source-project",
        not_after_step=12,
        from_entity="source-entity",
    )

    fork_io.api.artifacts.assert_called_once_with(
        "lora", "source-entity/source-project/source-model"
    )
    fork_io.api.run.assert_called_once_with("source-entity/source-project/source-model")
    assert fork_io.model_run.config == {"wandb.provenance": ["original-training-run"]}
    assert fork_io.init.call_args.kwargs["entity"] == "target"
    assert fork_io.init.call_args.kwargs["project"] == "destination-project"
    fork_io.make_artifact.assert_called_once_with(name="destination", type="lora")
    fork_io.run.log_artifact.assert_called_once_with(
        fork_io.destination, aliases=["step10", "latest"]
    )
    assert fork_io.uploads == [{"adapter.json": "selected"}]
    newer.download.assert_not_called()
    fork_io.run.finish.assert_called_once()


async def test_fork_checkpoint_defaults_to_destination_entity_and_project(
    fork_io: SimpleNamespace,
) -> None:
    await fork_io.backend._experimental_fork_checkpoint(fork_io.model, "source-model")

    fork_io.api.artifacts.assert_called_once_with(
        "lora", "target/destination-project/source-model"
    )
    fork_io.api.run.assert_called_once_with("target/destination-project/source-model")
    assert fork_io.uploads == [{"adapter.json": "checkpoint"}]


@pytest.fixture
def s3_io(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    latest = AsyncMock(return_value=10)
    not_after = AsyncMock(return_value=8)
    ensure_bucket = AsyncMock()

    async def sync(source: str, destination: str, *, verbose: bool) -> None:
        Path(destination, "adapter.json").write_text("s3-checkpoint")

    s3_sync = AsyncMock(side_effect=sync)
    monkeypatch.setattr(
        s3_checkpoint_utils, "get_latest_checkpoint_step_from_s3", latest
    )
    monkeypatch.setattr(
        s3_checkpoint_utils, "get_checkpoint_step_not_after_from_s3", not_after
    )
    monkeypatch.setattr(s3, "ensure_bucket_exists", ensure_bucket)
    monkeypatch.setattr(s3, "s3_sync", s3_sync)
    return SimpleNamespace(
        latest=latest, not_after=not_after, ensure_bucket=ensure_bucket, sync=s3_sync
    )


@pytest.mark.parametrize("limit", [None, 12])
async def test_fork_checkpoint_preserves_positional_s3_arguments(
    fork_io: SimpleNamespace, s3_io: SimpleNamespace, limit: int | None
) -> None:
    await fork_io.backend._experimental_fork_checkpoint(
        fork_io.model, "source-model", "source-project", "bucket", limit, True, "prefix"
    )

    expected = {
        "model_name": "source-model",
        "project": "source-project",
        "s3_bucket": "bucket",
        "prefix": "prefix",
    }
    if limit is None:
        s3_io.latest.assert_awaited_once_with(**expected)
        s3_io.not_after.assert_not_awaited()
        selected_step = 10
    else:
        s3_io.not_after.assert_awaited_once_with(**expected, not_after_step=limit)
        s3_io.latest.assert_not_awaited()
        selected_step = 8
    s3_io.ensure_bucket.assert_awaited_once_with("bucket")
    assert s3_io.sync.await_args.args[0] == (
        f"s3://bucket/prefix/source-project/models/source-model/checkpoints/{selected_step:04d}"
    )
    assert s3_io.sync.await_args.args[1] == str(
        Path(tempfile.gettempdir())
        / "art_fork_checkpoints"
        / "source-project"
        / "source-model"
        / f"{selected_step:04d}"
    )
    assert s3_io.sync.await_args.kwargs == {"verbose": True}
    fork_io.api.artifacts.assert_not_called()
    fork_io.api.run.assert_called_once_with("target/source-project/source-model")
    assert fork_io.uploads == [{"adapter.json": "s3-checkpoint"}]
    fork_io.run.log_artifact.assert_called_once_with(
        fork_io.destination, aliases=[f"step{selected_step}", "latest"]
    )


async def test_fork_checkpoint_source_entity_only_affects_wandb_when_using_s3(
    fork_io: SimpleNamespace, s3_io: SimpleNamespace
) -> None:
    await fork_io.backend._experimental_fork_checkpoint(
        fork_io.model,
        "source-model",
        from_project="source-project",
        from_s3_bucket="bucket",
        from_entity="source-entity",
    )

    assert s3_io.sync.await_args.args[0] == (
        "s3://bucket/source-project/models/source-model/checkpoints/0010"
    )
    fork_io.api.artifacts.assert_not_called()
    fork_io.api.run.assert_called_once_with("source-entity/source-project/source-model")
    assert fork_io.init.call_args.kwargs["entity"] == "target"


async def test_fork_checkpoint_does_not_mix_files_from_different_entities(
    fork_io: SimpleNamespace,
) -> None:
    fork_io.api.artifacts.side_effect = [
        [_source_artifact("first-entity.json", "first")],
        [_source_artifact("second-entity.json", "second")],
    ]
    fork_io.model.entity = "first-entity"
    await fork_io.backend._experimental_fork_checkpoint(fork_io.model, "source-model")
    fork_io.model.entity = "second-entity"
    await fork_io.backend._experimental_fork_checkpoint(fork_io.model, "source-model")

    assert fork_io.uploads == [
        {"first-entity.json": "first"},
        {"second-entity.json": "second"},
    ]
    assert all(not directory.exists() for directory in fork_io.directories)


@pytest.mark.parametrize("limit", [None, 5])
async def test_fork_checkpoint_without_matching_artifacts_does_not_upload(
    fork_io: SimpleNamespace, limit: int | None
) -> None:
    fork_io.api.artifacts.return_value = []

    with pytest.raises(ValueError, match="No checkpoints found"):
        await fork_io.backend._experimental_fork_checkpoint(
            fork_io.model, "source-model", not_after_step=limit
        )

    fork_io.init.assert_not_called()


async def test_fork_checkpoint_cleans_up_after_download_error(
    fork_io: SimpleNamespace,
) -> None:
    artifact = fork_io.api.artifacts.return_value[0]
    artifact.download.side_effect = OSError("download failed")

    with pytest.raises(OSError, match="download failed"):
        await fork_io.backend._experimental_fork_checkpoint(
            fork_io.model, "source-model"
        )

    directory = Path(artifact.download.call_args.kwargs["root"])
    assert not directory.exists()
    fork_io.init.assert_not_called()
