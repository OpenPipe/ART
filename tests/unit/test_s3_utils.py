import asyncio
import importlib
from pathlib import Path
import zipfile

import pytest

s3_utils = importlib.import_module("art.utils.s3")


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_archive_and_presign_step_url_uses_full_s3_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")

    upload_calls: list[tuple[str, str, bool, bool]] = []
    presign_calls: list[tuple[tuple[str, ...], object | None, object | None]] = []
    bucket_calls: list[str | None] = []

    async def fake_ensure_bucket_exists(
        s3_bucket: str | None = None, profile: str | None = None
    ) -> None:
        assert profile is None
        bucket_calls.append(s3_bucket)

    async def fake_s3_sync(
        source: str,
        destination: str,
        *,
        profile: str | None = None,
        verbose: bool = False,
        delete: bool = False,
        exclude: list[s3_utils.ExcludableOption] | None = None,
    ) -> None:
        assert profile is None
        assert exclude is None
        with zipfile.ZipFile(source) as archive:
            assert archive.namelist() == ["adapter_config.json"]
        upload_calls.append((source, destination, verbose, delete))

    async def fake_create_subprocess_exec(*cmd: str, stdout=None, stderr=None):
        presign_calls.append((cmd, stdout, stderr))
        return _FakeProcess(stdout=b"https://signed.example.com/model.zip\n")

    monkeypatch.setattr(s3_utils, "ensure_bucket_exists", fake_ensure_bucket_exists)
    monkeypatch.setattr(s3_utils, "s3_sync", fake_s3_sync)
    monkeypatch.setattr(
        s3_utils.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )

    presigned_url = await s3_utils.archive_and_presign_step_url(
        model_name="demo-model",
        project="demo-project",
        step=7,
        s3_bucket="demo-bucket",
        prefix="exports",
        checkpoint_path=str(checkpoint_path),
    )

    expected_s3_uri = s3_utils.build_s3_zipped_step_path(
        model_name="demo-model",
        project="demo-project",
        step=7,
        s3_bucket="demo-bucket",
        prefix="exports",
    )

    assert presigned_url == "https://signed.example.com/model.zip"
    assert bucket_calls == ["demo-bucket"]
    assert len(upload_calls) == 1
    assert upload_calls[0][1:] == (expected_s3_uri, False, False)
    assert presign_calls == [
        (
            ("aws", "s3", "presign", expected_s3_uri, "--expires-in", "3600"),
            asyncio.subprocess.PIPE,
            asyncio.subprocess.PIPE,
        )
    ]


@pytest.mark.asyncio
async def test_archive_and_presign_step_url_surfaces_presign_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "adapter_model.bin").write_text("weights", encoding="utf-8")

    async def fake_ensure_bucket_exists(
        s3_bucket: str | None = None, profile: str | None = None
    ) -> None:
        return None

    async def fake_s3_sync(
        source: str,
        destination: str,
        *,
        profile: str | None = None,
        verbose: bool = False,
        delete: bool = False,
        exclude: list[s3_utils.ExcludableOption] | None = None,
    ) -> None:
        return None

    async def fake_create_subprocess_exec(*cmd: str, stdout=None, stderr=None):
        return _FakeProcess(returncode=1, stderr=b"invalid S3 URI")

    monkeypatch.setattr(s3_utils, "ensure_bucket_exists", fake_ensure_bucket_exists)
    monkeypatch.setattr(s3_utils, "s3_sync", fake_s3_sync)
    monkeypatch.setattr(
        s3_utils.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )

    with pytest.raises(RuntimeError, match="invalid S3 URI"):
        await s3_utils.archive_and_presign_step_url(
            model_name="demo-model",
            project="demo-project",
            step=3,
            s3_bucket="demo-bucket",
            checkpoint_path=str(checkpoint_path),
        )
